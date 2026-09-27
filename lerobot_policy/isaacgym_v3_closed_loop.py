#!/usr/bin/env python3
"""Run ACT closed-loop episodes for the current boxed v3 scene manifest in Isaac Gym.

The old ACT Isaac Gym task is used as the physics backend.  Episode positions are
read from the current v3 manifest, while policy inputs remain the DemoGrasp
8-dimensional eef pose plus finger position and two RGB cameras.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import isaacgymenvs
import tasks  # noqa: F401
from lerobot_policy.isaacgym_closed_loop import PolicyClient


def scene_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("v3 scene manifest episode indices must be contiguous")
    return rows[:limit] if limit is not None else rows


def configure(cfg) -> None:
    hand = OmegaConf.load(ROOT / "tasks/hand/fr3_panda_gripper.yaml")
    with open_dict(cfg):
        # The v3 Isaac Sim data records the URDF's ee_link TCP. The historical
        # ACT task used fr3_link8 (the wrist), which is about 11 cm higher.
        hand.eef_link = "ee_link"
        cfg.hand = hand
        cfg.num_envs = 1
        cfg.task.env.numEnvs = 1
        cfg.task.env.episodeLength = 1000
        cfg.task.env.randomEpisodeLength = False
        cfg.task.env.armController = "pose"
        cfg.task.env.actionsMaxAngVelArm = 100.0
        cfg.task.env.actionsMaxAngVelHand = 100.0
        # Match the 10 Hz DemoGrasp frame interval in Isaac Gym (60 Hz sim).
        cfg.task.env.controlFrequencyInv = 6
        cfg.task.env.observationType = "eefpose+handdof"
        cfg.task.env.enablePointCloud = False
        cfg.task.env.resetDofPosRandomInterval = 0.0
        cfg.task.env.resetHandDofPosFullRange = False
        cfg.task.env.resetRandomRot = "fixed"
        cfg.task.env.tableHeightRange = [0.0, 0.0]
        cfg.task.env.enableRobotTableCollision = True
        cfg.task.env.objectFriction = 1.2
        cfg.task.env.gripperFriction = 1.2
        cfg.task.env.asset.multiObject = False
        cfg.task.env.asset.objectAssetFile = "stack_cube/cube_a.urdf"
        cfg.task.env.asset.useDistractorObjects = True
        cfg.task.env.asset.numDistractorObjects = 1
        cfg.task.env.asset.randomRemoveDistractorObjects = 0.0
        cfg.task.env.asset.distractorObjectAssetFile = "stack_cube/cube_b.urdf"
        cfg.task.env.asset.useFixedBoxObstacles = True
        cfg.task.env.asset.fixedBoxAssetFile = "stack_box/box.urdf"
        cfg.task.env.asset.fixedBoxDefaultPositions = [[0.0, 0.0, 0.005]] * 3
        cfg.task.env.render.enable = True
        cfg.task.env.enableCameraSensors = True
        cfg.task.env.render.appearance_realistic = True
        cfg.task.env.render.randomize = False
        cfg.task.env.render.camera_ids = [1, 2]
        cfg.task.env.render.data_type = "rgb"
        cfg.task.env.render.resize = [256, 256]
        cfg.force_render = True
        cfg.headless = True


def create_env(cfg):
    return isaacgymenvs.make(cfg.seed, cfg.task_name, cfg.task.env.numEnvs,
                             cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                             cfg.headless, cfg.multi_gpu, False, cfg.force_render, cfg)


def set_scene(env, row: dict) -> None:
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids)
    # v3 source coordinates use the FR3 base frame. Isaac Gym's stack task uses
    # the same table frame offset as the original ACT collection.
    offset = np.array([0.61, 0.0, 0.0], dtype=np.float32)
    a = np.asarray(row["cube_a_initial_xyz"], dtype=np.float32) + offset
    b = np.asarray(row["cube_b_initial_xyz"], dtype=np.float32) + offset
    object_index = env.object_indices[0]
    target_index = env.distractor_object_indices[0, 0]
    env.root_state_tensor[object_index, :3] = torch.from_numpy(a).to(env.device)
    env.root_state_tensor[target_index, :3] = torch.from_numpy(b).to(env.device)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    env.root_state_tensor[object_index, 3:7] = quat
    env.root_state_tensor[target_index, 3:7] = quat
    env.root_state_tensor[object_index, 7:] = 0.0
    env.root_state_tensor[target_index, 7:] = 0.0
    indices = torch.tensor([object_index, target_index], dtype=torch.int32, device=env.device)
    box_positions = [row["boxes"][f"box{i}"]["position"] for i in (1, 2, 3)]
    box_indices = []
    for box_index, position in zip(env.fixed_box_indices[0], box_positions):
        # Box transforms are authored in the Isaac scene/world frame, unlike
        # cube and EEF telemetry which use the FR3 base frame.
        world_position = np.asarray(position, dtype=np.float32)
        env.root_state_tensor[box_index, :3] = torch.from_numpy(world_position).to(env.device)
        env.root_state_tensor[box_index, 3:7] = quat
        env.root_state_tensor[box_index, 7:] = 0.0
        box_indices.append(box_index)
    if box_indices:
        indices = torch.cat([indices, torch.tensor(box_indices, dtype=torch.int32, device=env.device)])
    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_state_tensor),
                                                gymtorch.unwrap_tensor(indices), len(indices))
    for handle_name, color in (("object", (0.85, 0.12, 0.08)), ("distractor", (0.08, 0.20, 0.85))):
        handle = env.gym.find_actor_handle(env.envs[0], handle_name)
        env.gym.set_rigid_body_color(env.envs[0], handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
    env.instructions = ["Pick up the red cube and stack it on the blue cube."]
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.compute_observations()


def observation(env) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    raw = env.compute_real_observation_dict()
    state = np.concatenate([raw["right_arm_eef_pose"], raw["right_hand_qpos"]], axis=-1).astype(np.float32)
    result = {"observation.state": state}
    frames = []
    for camera_id in env.camera_ids:
        frame = raw[f"camera_{camera_id}.rgb"]
        result[f"observation.camera_{camera_id}.rgb"] = frame
        frames.append(frame)
    return result, frames


def environment_action(env, action: np.ndarray) -> torch.Tensor:
    """Convert DemoGrasp metres for the finger to Isaac Gym's [-1, 1] input."""
    value = np.asarray(action, dtype=np.float32).copy()
    lower = float(env.robot_dof_lower_limits[env.active_hand_dof_indices[0]].item())
    upper = float(env.robot_dof_upper_limits[env.active_hand_dof_indices[0]].item())
    value[..., 7] = np.clip(2.0 * (value[..., 7] - lower) / (upper - lower) - 1.0, -1.0, 1.0)
    return torch.from_numpy(value).to(env.device)


def metric(env) -> dict:
    a = env.root_state_tensor[env.object_indices[0]]
    b = env.root_state_tensor[env.distractor_object_indices[0, 0]]
    xy = float(torch.linalg.vector_norm(a[:2] - b[:2]).item())
    z_delta = float((a[2] - b[2]).item())
    speed_a = float(torch.linalg.vector_norm(a[7:10]).item())
    speed_b = float(torch.linalg.vector_norm(b[7:10]).item())
    return {"success": xy <= 0.009 and abs(z_delta - 0.025) <= 0.004 and
            float(a[2]) >= 0.031 and speed_a <= 0.05 and speed_b <= 0.05,
            "xy_error_m": xy, "z_delta_m": z_delta,
            "cube_a_speed_mps": speed_a, "cube_b_speed_mps": speed_b}


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg) -> None:
    manifest = Path(str(cfg.get("v3_manifest"))).expanduser().resolve()
    checkpoint = Path(str(cfg.get("v3_checkpoint"))).expanduser().resolve()
    output = Path(str(cfg.get("v3_output"))).expanduser().resolve()
    limit = int(cfg.get("v3_limit", 3))
    policy_port = int(cfg.get("v3_policy_port", 5555))
    set_np_formatting()
    rows = scene_rows(manifest, limit)
    configure(cfg)
    cfg.seed = set_seed(42, torch_deterministic=True, rank=0)
    output.mkdir(parents=True, exist_ok=True)
    env = create_env(cfg)
    client = PolicyClient("127.0.0.1", policy_port, 30.0, 60.0)
    results = []
    try:
        for row in rows:
            set_scene(env, row)
            client.reset()
            last = None
            for step in range(40):
                obs, frames = observation(env)
                action = client.predict(obs)
                if action.shape != (1, 8):
                    raise RuntimeError(f"unexpected action shape {action.shape}")
                last = environment_action(env, action)
                env.step(last)
            # Continue physics with the final target so the stack settles. The
            # enlarged episodeLength prevents an automatic reset at step 40.
            for _ in range(120):
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            result = {"episode_index": int(row["episode_index"]), **metric(env)}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        client.close()
    summary = {"num_episodes": len(results), "num_successes": sum(r["success"] for r in results),
               "success_rate": sum(r["success"] for r in results) / max(1, len(results)),
               "manifest": str(manifest), "checkpoint": str(checkpoint),
               "physics_backend": "Isaac Gym ACT task with fixed box obstacles", "static_box_geometry": True,
               "episodes": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
