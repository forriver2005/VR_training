#!/usr/bin/env python3
"""Run an ACT checkpoint on a recorded episode without commanding a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_batch_dimension(item: dict, input_keys: set[str]) -> dict:
    batch = {}
    for key in input_keys:
        value = item[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        batch[key] = value.unsqueeze(0)
    return batch


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset.expanduser().resolve(),
        episodes=[args.episode],
    )
    policy = ACTPolicy.from_pretrained(checkpoint).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
    )
    policy.eval()
    policy.reset()

    input_keys = set(policy.config.input_features)
    predictions = []
    targets = []
    with torch.inference_mode():
        for frame_index in range(len(dataset)):
            item = dataset[frame_index]
            batch = add_batch_dimension(item, input_keys)
            action = postprocessor(policy.select_action(preprocessor(batch)))
            if isinstance(action, dict):
                action = action["action"]
            predictions.append(action.squeeze(0).detach().cpu().numpy())
            targets.append(item["action"].detach().cpu().numpy())

    predictions_array = np.stack(predictions)
    targets_array = np.stack(targets)
    mae_per_dimension = np.abs(predictions_array - targets_array).mean(axis=0)
    summary = {
        "episode": args.episode,
        "frames": len(dataset),
        "mean_absolute_error": float(mae_per_dimension.mean()),
        "mae_per_action_dimension": mae_per_dimension.tolist(),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predicted_action=predictions_array, target_action=targets_array)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
