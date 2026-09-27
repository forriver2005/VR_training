#!/usr/bin/env python3
"""Encode successful Isaac Gym raw episodes as a LeRobot v3 dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    root = Path("data/isaacgym_trajectorysuccessful2_fr3_v3_box_20260927_raw").resolve()
    manifest = Path("data/datasets/trajectorysuccessful2_fr3_v3_strict_friction12_20260927/meta/scenes.jsonl").resolve()
    output = Path("data/datasets/trajectorysuccessful2_fr3_v3_isaacgym_box_20260927").resolve()
    if output.exists():
        shutil.rmtree(output)
    rows = {int(row["episode_index"]): row for row in
            (json.loads(line) for line in manifest.read_text().splitlines() if line.strip())}
    files = sorted(root.glob("episode_*.npz"))
    if len(files) < 80:
        raise RuntimeError(f"Expected at least 80 successful Isaac Gym episodes, found {len(files)}")
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        "action": {"dtype": "float32", "shape": (8,), "names": [f"action_{i}" for i in range(8)]},
        "observation.camera_1.rgb": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.camera_2.rgb": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="trajectorysuccessful2_fr3_v3_isaacgym_box_20260927",
        root=output, fps=10, robot_type="fr3v2_panda_gripper",
        features=features, use_videos=True, vcodec="h264", streaming_encoding=True,
    )
    scene_path = output / "meta/scenes.jsonl"
    for new_index, path in enumerate(files):
        source_index = int(path.stem.split("_")[-1])
        episode = np.load(path)
        if episode["state"].shape != (40, 8) or episode["action"].shape != (40, 8):
            raise ValueError(f"Invalid episode shape: {path}")
        for frame in range(40):
            dataset.add_frame({
                "observation.state": episode["state"][frame],
                "action": episode["action"][frame],
                "observation.camera_1.rgb": episode["camera_1_rgb"][frame],
                "observation.camera_2.rgb": episode["camera_2_rgb"][frame],
                "task": "Pick up the red cube and stack it on the blue cube.",
            })
        dataset.save_episode()
        record = {"episode_index": new_index, "source_episode_index": source_index,
                  "collection_backend": "Isaac Gym", "success": True,
                  **rows[source_index]}
        record["episode_index"] = new_index
        with scene_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    metadata = {
        "task": "cube_stacking", "simulator": "Isaac Gym", "fps": 10,
        "frames_per_episode": 40, "observation_state_dim": 8, "action_dim": 8,
        "camera_resolution": [256, 256], "camera_features": ["observation.camera_1.rgb", "observation.camera_2.rgb"],
        "demo_grasp_modality": "eefpose+finger_position; no joint angles as policy input",
        "robot": "FR3V2 panda two-finger gripper; eef_link TCP", "static_box_geometry": True,
        "contact_friction": {"static": 1.2, "dynamic": 1.0, "restitution": 0.0},
        "source_manifest": str(manifest), "successful_episode_count": len(files),
    }
    (output / "meta/collection_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(files), "frames": len(files) * 40}, indent=2))


if __name__ == "__main__":
    main()
