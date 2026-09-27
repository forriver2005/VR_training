#!/usr/bin/env python3
"""Collect current boxed demonstrations with the original Isaac Gym task."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from isaacgym import gymtorch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lerobot_policy.isaacgym_v3_closed_loop import (  # noqa: E402
    configure, create_env, environment_action, metric, observation, scene_rows, set_scene,
)
import tasks  # noqa: F401, E402
from isaacgymenvs.utils.utils import set_np_formatting, set_seed  # noqa: E402


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg) -> None:
    manifest = Path(str(cfg.get("v3_manifest"))).expanduser().resolve()
    output = Path(str(cfg.get("v3_output"))).expanduser().resolve()
    limit = cfg.get("v3_limit")
    limit = None if limit in (None, "", "null") else int(limit)
    set_np_formatting()
    rows = scene_rows(manifest, limit)
    start = int(cfg.get("v3_start", 0))
    stride = int(cfg.get("v3_stride", 1))
    rows = rows[start::stride]
    configure(cfg)
    cfg.seed = set_seed(42, torch_deterministic=True, rank=0)
    output.mkdir(parents=True, exist_ok=True)
    env = create_env(cfg)
    results = []
    try:
        for row in rows:
            set_scene(env, row)
            states, actions, camera_1, camera_2 = [], [], [], []
            source_plan = np.asarray(row["expert_plan"], dtype=np.float32)
            if source_plan.shape != (40, 8):
                raise ValueError(f"episode {row['episode_index']} expert_plan has shape {source_plan.shape}")
            # Generate a fresh Isaac Gym expert at the actual scene positions.
            # Repeating each waypoint makes the demonstration independent of
            # the source simulator's IK interpolation and gives the fingers
            # several control intervals to establish contact.
            red = np.asarray(row["cube_a_initial_xyz"], dtype=np.float32) + np.asarray([0.61, 0, 0], dtype=np.float32)
            blue = np.asarray(row["cube_b_initial_xyz"], dtype=np.float32) + np.asarray([0.61, 0, 0], dtype=np.float32)
            q = source_plan[0, 3:7]
            plan = np.zeros((40, 8), dtype=np.float32)
            waypoints = [
                (0, 8, [red[0], red[1], 0.10], 0.03),
                (8, 13, [red[0], red[1], 0.011], 0.03),
                (13, 20, [red[0], red[1], 0.011], 0.0),
                (20, 24, [red[0], red[1], 0.10], 0.0),
                (24, 31, [blue[0], blue[1], 0.10], 0.0),
                (31, 35, [blue[0], blue[1], 0.045], 0.0),
                (35, 37, [blue[0], blue[1], 0.045], 0.03),
                (37, 40, [blue[0], blue[1], 0.12], 0.03),
            ]
            for start, end, position, finger in waypoints:
                plan[start:end, :3] = np.asarray(position, dtype=np.float32)
                plan[start:end, 3:7] = q
                plan[start:end, 7] = finger
            expert_plan = plan
            for frame in range(40):
                obs, frames = observation(env)
                if os.environ.get("IGYM_TRACE") == "1" and frame in (0, 10, 20, 31, 39):
                    eef = obs["observation.state"][0, :3].tolist()
                    print(f"trace episode={row['episode_index']} frame={frame} eef={eef} finger={obs['observation.state'][0,7]:.4f} target={expert_plan[frame, :3].tolist()} cube={row['cube_a_initial_xyz']}", flush=True)
                states.append(obs["observation.state"][0].copy())
                actions.append(expert_plan[frame].copy())
                camera_1.append(frames[0][0].copy())
                camera_2.append(frames[1][0].copy())
                env.step(environment_action(env, expert_plan[frame:frame + 1]))
            for _ in range(120):
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            result = {"episode_index": int(row["episode_index"]), **metric(env)}
            results.append(result)
            if result["success"]:
                np.savez_compressed(
                    output / f"episode_{int(row['episode_index']):06d}.npz",
                    state=np.asarray(states, dtype=np.float32),
                    action=np.asarray(actions, dtype=np.float32),
                    camera_1_rgb=np.asarray(camera_1, dtype=np.uint8),
                    camera_2_rgb=np.asarray(camera_2, dtype=np.uint8),
                )
            print(json.dumps(result), flush=True)
    finally:
        pass
    summary = {
        "num_episodes": len(results),
        "num_successes": sum(bool(item["success"]) for item in results),
        "success_rate": sum(bool(item["success"]) for item in results) / max(1, len(results)),
        "manifest": str(manifest),
        "physics_backend": "Isaac Gym FR3V2 panda gripper with fixed box obstacles",
        "episodes": results,
    }
    (output / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
