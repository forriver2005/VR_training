#!/usr/bin/env python3
"""Build a one-episode RGB+state LeRobot dataset for Diffusion Policy overfit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK = "Pick up the red cube and stack it on the blue cube."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/isaacsim_trajectorysuccessful2_fr3v2_strict_friction12_20260927/episode_000000.npz"),
    )
    p.add_argument("--output", type=Path, default=Path("data/datasets/demograsp_fr3v2_rgb_state_overfit_episode0_20261009"))
    p.add_argument("--repo-id", default="demograsp_fr3v2_rgb_state_overfit_episode0")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)

    with np.load(source) as data:
        required = ["state", "action", "camera_1_rgb", "camera_2_rgb"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing keys in {source}: {missing}")
        state = np.asarray(data["state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        cam1 = np.asarray(data["camera_1_rgb"], dtype=np.uint8)
        cam2 = np.asarray(data["camera_2_rgb"], dtype=np.uint8)

    if state.shape != (40, 8) or action.shape != (40, 8):
        raise ValueError(f"Expected state/action shape (40, 8), got {state.shape}/{action.shape}")
    if cam1.shape != (40, 256, 256, 3) or cam2.shape != (40, 256, 256, 3):
        raise ValueError(f"Expected RGB shape (40, 256, 256, 3), got {cam1.shape}/{cam2.shape}")
    for name, array in (("state", state), ("action", action)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        "action": {"dtype": "float32", "shape": (8,), "names": [f"action_{i}" for i in range(8)]},
        "observation.camera_1.rgb": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.camera_2.rgb": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=args.fps,
        robot_type="fr3v2_panda_gripper",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=1,
    )
    try:
        for i in range(40):
            dataset.add_frame({
                "observation.state": state[i],
                "action": action[i],
                "observation.camera_1.rgb": cam1[i],
                "observation.camera_2.rgb": cam2[i],
                "task": TASK,
            })
        dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()

    manifest = {
        "source_npz": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "episode_index": 0,
        "frames": 40,
        "fps": args.fps,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "image_augmentation": False,
        "random_crop": False,
        "cross_episode_windows": False,
        "normalization": "dataset statistics computed from this episode only",
        "simulator": "Isaac Sim",
        "robot": "FR3V2 panda two-finger gripper",
        "task": TASK,
    }
    (output / "meta" / "overfit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": 40, "stats": str(output / "meta/stats.json")}, indent=2))


if __name__ == "__main__":
    main()
