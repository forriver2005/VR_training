#!/usr/bin/env python3
"""Serve a LeRobot ACT checkpoint to the DemoGrasp Isaac Gym process."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from socket_protocol import command_message, read_command, receive_arrays, send_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-action-steps", type=int)
    return parser.parse_args()


class ACTInferenceServer:
    def __init__(self, checkpoint: Path, device: str, n_action_steps: int | None = None) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        checkpoint = checkpoint.expanduser().resolve()
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        if n_action_steps is not None:
            if n_action_steps <= 0 or n_action_steps > policy_config.chunk_size:
                raise ValueError(f"n_action_steps must be in [1, {policy_config.chunk_size}]")
            policy_config.n_action_steps = n_action_steps
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=policy_config).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
        )
        self.input_features = self.policy.config.input_features
        self.action_dim = self.policy.config.output_features["action"].shape[0]
        self.policy.reset()
        print(f"Loaded ACT checkpoint: {checkpoint}", flush=True)
        print(f"Input features: {list(self.input_features)}", flush=True)

    def _tensor_from_request(self, key: str, value: np.ndarray) -> torch.Tensor:
        expected = self.input_features[key]
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if len(expected.shape) == 3:
            if tensor.ndim != 4 or tensor.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Expected NHWC image batch for {key}, got {tuple(tensor.shape)}")
            tensor = tensor[..., :3].permute(0, 3, 1, 2).float().div_(255.0)
        else:
            tensor = tensor.float()
        return tensor.to(self.device, non_blocking=True)

    def predict(self, request: dict[str, np.ndarray]) -> np.ndarray:
        missing = set(self.input_features) - set(request)
        if missing:
            raise ValueError(f"Request is missing policy inputs: {sorted(missing)}")
        batch = {
            key: self._tensor_from_request(key, request[key])
            for key in self.input_features
        }
        batch_sizes = {value.shape[0] for value in batch.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"Inconsistent request batch sizes: {batch_sizes}")

        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(self.preprocessor(batch)))
        if isinstance(action, dict):
            action = action["action"]
        result = action.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape[1] != self.action_dim:
            raise RuntimeError(f"Unexpected policy output shape: {result.shape}")
        if not np.isfinite(result).all():
            raise RuntimeError("Policy returned a non-finite action")
        return result

    def handle_connection(self, connection: socket.socket, address: tuple[str, int]) -> None:
        print(f"Client connected: {address[0]}:{address[1]}", flush=True)
        while True:
            request = receive_arrays(connection)
            command = read_command(request)
            if command == "ping":
                send_arrays(connection, command_message("pong"))
            elif command == "reset":
                self.policy.reset()
                send_arrays(connection, command_message("ok"))
            elif command == "predict":
                action = self.predict(request)
                send_arrays(connection, {"command": np.asarray("action"), "action": action})
            else:
                raise ValueError(f"Unknown command: {command}")


def main() -> None:
    args = parse_args()
    server = ACTInferenceServer(args.checkpoint, args.device, args.n_action_steps)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"Listening on {args.host}:{args.port}", flush=True)
        while True:
            connection, address = listener.accept()
            try:
                with connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    server.handle_connection(connection, address)
            except (ConnectionError, OSError) as exc:
                print(f"Client disconnected: {exc}", flush=True)
                server.policy.reset()
            except Exception as exc:
                print(f"Request failed: {type(exc).__name__}: {exc}", flush=True)
                server.policy.reset()


if __name__ == "__main__":
    main()
