#!/usr/bin/env python3
"""Evaluate a remote LeRobot policy in the existing DemoGrasp Isaac Gym task."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import hydra
from isaacgym import gymapi  # noqa: F401 - Isaac Gym must be imported before torch
import isaacgymenvs
import numpy as np
import torch
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from omegaconf import DictConfig, OmegaConf

import tasks  # noqa: F401 - registers the custom grasp task
from socket_protocol import command_message, read_command, receive_arrays, send_arrays


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


REPLAY_OPTION_KEYS = (
    "scene_selection",
    "object_state",
    "robot_state",
    "camera_state",
    "table_state",
    "visual",
    "lighting",
    "task_config",
)


def resolve_replay_options(cfg: DictConfig, seen_data: bool) -> dict[str, bool]:
    def config_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    options = {
        key: config_bool(cfg.get(f"use_train_{key}", False))
        for key in REPLAY_OPTION_KEYS
    }
    if seen_data:
        options = {key: True for key in REPLAY_OPTION_KEYS}
    elif any(value for key, value in options.items() if key != "scene_selection"):
        options["scene_selection"] = True
    return options


def configure_scene_replay(
    cfg: DictConfig, options: dict[str, bool]
) -> tuple[list[dict], Path]:
    manifest_path = Path(str(cfg.get("scene_manifest", ""))).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training-data replay requires a scene manifest: "
            f"{manifest_path}"
        )
    collection_config_path = Path(
        str(cfg.get("collection_config", manifest_path.with_name("collection_config.json")))
    ).expanduser().resolve()
    if not collection_config_path.is_file():
        raise FileNotFoundError(f"Collection config not found: {collection_config_path}")

    collection = load_json(collection_config_path)
    if int(collection.get("scene_manifest_version", -1)) != 1:
        raise ValueError("Unsupported collection scene manifest version")
    saved_cfg = collection["config"]
    saved_task = OmegaConf.create(saved_cfg["task"])
    if options["task_config"]:
        cfg.task = saved_task
    else:
        # Object slots and actor topology must match source_env_index even when
        # individual state components are freshly randomized.
        cfg.task.env.asset = saved_task.env.asset
        cfg.task.env.camera_config = saved_task.env.camera_config
        cfg.task.env.render.appearance_realistic = saved_task.env.render.appearance_realistic
        cfg.task.env.render.camera_ids = saved_task.env.render.camera_ids
        cfg.task.env.render.resize = saved_task.env.render.resize
        if options["visual"]:
            cfg.task.env.render.randomize = True
            cfg.task.env.render.randomization_params.texture_folder = (
                saved_task.env.render.randomization_params.texture_folder
            )
    if "hand" in saved_cfg:
        cfg.hand = OmegaConf.create(saved_cfg["hand"])
    if options["task_config"]:
        cfg.seed = int(saved_cfg["seed"])
    collection_num_envs = int(saved_cfg["task"]["env"]["numEnvs"])
    cfg.num_envs = collection_num_envs
    cfg.task.env.numEnvs = collection_num_envs

    records = sorted(load_jsonl(manifest_path), key=lambda item: item["episode_index"])
    if not records:
        raise ValueError(f"Scene manifest is empty: {manifest_path}")
    for record in records:
        if int(record.get("scene_manifest_version", -1)) != 1:
            raise ValueError(
                f"Unsupported scene manifest version in episode {record.get('episode_index')}"
            )
        env_index = int(record["source_env_index"])
        if not 0 <= env_index < collection_num_envs:
            raise ValueError(
                f"Episode {record['episode_index']} source_env_index {env_index} is outside "
                f"the collection environment count {collection_num_envs}"
            )
    return records, manifest_path


def group_scene_records(records: list[dict]) -> list[list[dict]]:
    groups: OrderedDict[int, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(int(record["collection_batch"]), []).append(record)
    for collection_batch, group in groups.items():
        env_indices = [int(record["source_env_index"]) for record in group]
        if len(env_indices) != len(set(env_indices)):
            raise ValueError(f"Duplicate source env in collection batch {collection_batch}")
    return list(groups.values())


def scene_config_summary(
    cfg: DictConfig,
    seen_data: bool,
    manifest_path: Path | None,
    replay_options: dict[str, bool],
) -> dict:
    render = cfg.task.env.render
    params = render.randomization_params
    return {
        "seen_data": seen_data,
        "scene_mode": (
            "manifest_replay"
            if seen_data
            else ("manifest_ablation" if manifest_path is not None else "evaluation")
        ),
        "scene_manifest": str(manifest_path) if manifest_path is not None else None,
        "use_train": replay_options,
        "seed": int(cfg.seed),
        "num_envs": int(cfg.task.env.numEnvs),
        "object_list": str(cfg.task.env.asset.multiObjectList),
        "reset_position_range": OmegaConf.to_container(
            cfg.task.env.resetPositionRange, resolve=True
        ),
        "reset_random_rotation": str(cfg.task.env.resetRandomRot),
        "randomize_tracking_reference": bool(cfg.task.env.randomizeTrackingReference),
        "randomize_grasp_pose": bool(cfg.task.env.randomizeGraspPose),
        "render_randomize": bool(render.randomize),
        "camera_position_range": list(params.camera_pos),
        "camera_quaternion_range": list(params.camera_quat),
        "table_position_range": list(params.table_xyz),
        "object_random_texture": bool(params.object_random_texture),
        "light_intensity_range": list(params.light_intensity),
        "light_ambient_range": list(params.light_ambient),
    }


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_s: float, connect_timeout_s: float) -> None:
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self.socket = socket.create_connection((host, port), timeout=timeout_s)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not connect to policy server at {host}:{port}")
                time.sleep(0.5)
        self._round_trip(command_message("ping"), "pong")

    def _round_trip(self, request: dict[str, np.ndarray], expected: str) -> dict[str, np.ndarray]:
        send_arrays(self.socket, request)
        response = receive_arrays(self.socket)
        command = read_command(response)
        if command != expected:
            raise RuntimeError(f"Expected server response {expected!r}, got {command!r}")
        return response

    def reset(self) -> None:
        self._round_trip(command_message("reset"), "ok")

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        request = {"command": np.asarray("predict"), **observation}
        response = self._round_trip(request, "action")
        return response["action"]

    def close(self) -> None:
        self.socket.close()


class EpisodeVideoWriter:
    def __init__(self, path: Path, fps: int, first_frame: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = first_frame.shape[:2]
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video output: {path}")
        self.write(first_frame)

    def write(self, rgb_frame: np.ndarray) -> None:
        self.writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


def policy_observation(env) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    raw = env.compute_real_observation_dict()
    state = np.concatenate([raw["right_arm_eef_pose"], raw["right_hand_qpos"]], axis=-1)
    observation = {"observation.state": state.astype(np.float32, copy=False)}
    camera_frames = []
    for camera_id in env.camera_ids:
        image = raw[f"camera_{camera_id}.rgb"]
        observation[f"observation.camera_{camera_id}.rgb"] = image
        camera_frames.append(image)
    return observation, camera_frames


def make_visualization_frame(
    camera_frames: list[np.ndarray], action: np.ndarray, step: int, env_index: int
) -> np.ndarray:
    views = [frames[env_index] for frames in camera_frames]
    frame = np.concatenate(views, axis=1)
    text = f"step={step:02d} action=[{action[0]:+.2f}, {action[1]:+.2f}, ...]"
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame


def create_isaacgym_env(cfg: DictConfig):
    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        False,
        cfg.force_render,
        cfg,
    )
    return env


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    set_np_formatting()
    seen_data = bool(cfg.get("seen_data", False))
    replay_options = resolve_replay_options(cfg, seen_data)
    replay_requested = replay_options["scene_selection"]
    scene_records = None
    manifest_path = None
    if replay_requested:
        scene_records, manifest_path = configure_scene_replay(cfg, replay_options)
        print(
            f"Manifest replay: {manifest_path} "
            f"({len(scene_records)} scenes, {cfg.task.env.numEnvs} source environments)"
        )
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    if not cfg.task.env.render.enable or "rgb" not in cfg.task.env.render.data_type:
        raise ValueError("Closed-loop ACT evaluation requires RGB rendering")
    if cfg.task.env.armController != "pose":
        raise ValueError("The collected ACT action uses armController=pose")

    host = str(cfg.get("policy_host", "127.0.0.1"))
    port = int(cfg.get("policy_port", 5555))
    num_episodes = int(cfg.get("num_eval_episodes", 10))
    if num_episodes <= 0:
        raise ValueError("num_eval_episodes must be positive")
    if scene_records is not None:
        if num_episodes > len(scene_records):
            raise ValueError(
                f"Requested {num_episodes} episodes, but manifest has {len(scene_records)}"
            )
        scene_batches = group_scene_records(scene_records[:num_episodes])
    else:
        scene_batches = None
    timeout_s = float(cfg.get("policy_timeout_s", 10.0))
    connect_timeout_s = float(cfg.get("connect_timeout_s", 60.0))
    record_video = bool(cfg.get("record_eval_video", True))
    output_root = Path(str(cfg.get("eval_output_dir", "lerobot_policy/outputs/sim_eval"))).resolve()
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    env = create_isaacgym_env(cfg)
    if not hasattr(env, "instructions"):
        env.instructions = [env.instruction_template] * env.num_envs
    client = None
    video_writers: dict[int, EpisodeVideoWriter] = {}
    results = []
    completed = 0
    batch_index = 0
    try:
        client = PolicyClient(host, port, timeout_s, connect_timeout_s)
        while completed < num_episodes:
            env.reset_idx(torch.arange(env.num_envs))
            if scene_batches is not None:
                batch_scenes = scene_batches[batch_index]
                evaluated_env_indices = [
                    int(scene["source_env_index"]) for scene in batch_scenes
                ]
                replay_env_ids = torch.tensor(evaluated_env_indices, dtype=torch.long)
                env.restore_scene_states(
                    replay_env_ids, batch_scenes, replay_options=replay_options
                )
            else:
                batch_scenes = None
                evaluated_env_indices = list(
                    range(min(env.num_envs, num_episodes - completed))
                )
            client.reset()
            video_writers = {}
            extras = None
            for step in range(env.max_episode_length):
                observation, camera_frames = policy_observation(env)
                start = time.perf_counter()
                action = client.predict(observation)
                inference_ms = (time.perf_counter() - start) * 1000
                if action.shape != (env.num_envs, env.num_actions):
                    raise RuntimeError(
                        f"Policy action shape {action.shape} != {(env.num_envs, env.num_actions)}"
                    )
                action_tensor = torch.from_numpy(action).to(env.device)
                _, _, _, extras = env.step(action_tensor)

                if record_video:
                    for result_offset, env_index in enumerate(evaluated_env_indices):
                        frame = make_visualization_frame(
                            camera_frames, action[env_index], step, env_index
                        )
                        writer = video_writers.get(env_index)
                        if writer is None:
                            writer = EpisodeVideoWriter(
                                run_dir
                                / (
                                    f"episode_{completed + result_offset:06d}_"
                                    f"env_{env_index:03d}.mp4"
                                ),
                                round(1 / env.dt / env.decimation),
                                frame,
                            )
                            video_writers[env_index] = writer
                        else:
                            writer.write(frame)
                print(
                    f"batch={batch_index} step={step + 1}/{env.max_episode_length} "
                    f"inference={inference_ms:.1f}ms",
                    end="\r",
                    flush=True,
                )
            print()
            for writer in video_writers.values():
                writer.close()
            video_writers = {}
            if extras is None:
                raise RuntimeError("Episode finished without environment metrics")

            successes = (extras["current_successes"] > 0.5).detach().cpu().numpy()
            for result_offset, env_index in enumerate(evaluated_env_indices):
                scene = batch_scenes[result_offset] if batch_scenes is not None else None
                result = {
                    "episode": completed,
                    "batch": batch_index,
                    "env_index": env_index,
                    "success": bool(successes[env_index]),
                }
                if scene is not None:
                    result.update({
                        "dataset_episode": int(scene["episode_index"]),
                        "collection_batch": int(scene["collection_batch"]),
                        "object_asset": scene["object_asset"],
                    })
                results.append(result)
                completed += 1
            success_rate = np.mean([item["success"] for item in results])
            print(f"Completed {completed}/{num_episodes}; success rate: {success_rate:.1%}")
            batch_index += 1
    finally:
        for writer in video_writers.values():
            writer.close()
        if client is not None:
            client.close()

    summary = {
        "num_episodes": len(results),
        "num_successes": sum(item["success"] for item in results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "scene_config": scene_config_summary(
            cfg, seen_data, manifest_path, replay_options
        ),
        "episodes": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    print(f"Saved evaluation to {run_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Gym Preview can segfault while Python tears down live GPU camera tensors.
    os._exit(0)


if __name__ == "__main__":
    main()
