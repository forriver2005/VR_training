#!/usr/bin/env python3
"""Replay, validate, and collect the raw cube-stacking demonstrations in Isaac Gym."""

from __future__ import annotations

import json
import os
import sys
import hashlib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from isaacgym import gymapi, gymtorch  # Isaac Gym must be imported before torch.
import isaacgymenvs
import numpy as np
import torch
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from omegaconf import DictConfig, OmegaConf, open_dict

import tasks  # noqa: F401 - registers the DemoGrasp task
from data.dataset_utils import LerobotDatasetWriter
from lerobot_policy.stack_data import (
    StackTrajectory,
    infer_geometry,
    load_all_trajectories,
    load_selected_trajectories,
    perturb_trajectory,
)


RAW_ROOT = Path(
    os.environ.get("STACK_RAW_ROOT", REPO_ROOT / "houshengyuan_data")
).expanduser().resolve()
SELECTION_PATH = Path(os.environ.get("STACK_SELECTION_PATH", REPO_ROOT / "lerobot_policy/stack_artifacts/selected_successful_100.json"))
V2_DATASET_NAME = os.environ.get("STACK_V2_DATASET_NAME", "stack_cube_hsy_v2")
AUGMENTED_V2_DATASET_NAME = os.environ.get(
    "STACK_AUGMENTED_V2_NAME", "stack_cube_hsy_augmented_v2"
)
AUGMENTED_MANIFEST_PATH = Path(
    os.environ.get(
        "STACK_AUGMENTED_MANIFEST",
        REPO_ROOT / "lerobot_policy/stack_artifacts/augmented_successful_300.json",
    )
)
NUM_STEPS = 40
NUM_SCENES = 10
PER_SCENE = 10
TABLE_FRAME_X_M = 0.61
DEFAULT_GRASP_WORLD_Z_OFFSET_M = -0.03
DEFAULT_PLACE_WORLD_Z_OFFSET_M = -0.015
CUBE_A_RGB = [0.85, 0.12, 0.08]
CUBE_B_RGB = [0.08, 0.20, 0.85]


def configure_stack_task(cfg: DictConfig, num_envs: int, render: bool) -> None:
    with open_dict(cfg):
        cfg.num_envs = num_envs
        cfg.task.env.numEnvs = num_envs
        cfg.task.env.episodeLength = NUM_STEPS
        cfg.task.env.randomEpisodeLength = False
        cfg.task.env.armController = "pose"
        cfg.task.env.observationType = "eefpose+handdof+objpose"
        cfg.task.env.enablePointCloud = False
        cfg.task.env.resetDofPosRandomInterval = 0.0
        cfg.task.env.resetHandDofPosFullRange = False
        cfg.task.env.resetRandomRot = "fixed"
        cfg.task.env.tableHeightRange = [0.0, 0.0]
        cfg.task.env.enableRobotTableCollision = True
        cfg.task.env.useObjectVhacd = False
        cfg.task.env.asset.multiObject = False
        cfg.task.env.asset.objectAssetFile = "stack_cube/cube.urdf"
        cfg.task.env.asset.useDistractorObjects = True
        cfg.task.env.asset.numDistractorObjects = 1
        cfg.task.env.asset.randomRemoveDistractorObjects = 0.0
        cfg.task.env.asset.distractorObjectAssetFile = "stack_cube/cube.urdf"
        cfg.task.env.render.enable = render
        cfg.task.env.enableCameraSensors = render
        cfg.task.env.render.appearance_realistic = True
        cfg.task.env.render.randomize = False
        cfg.task.env.render.camera_ids = [1, 2]
        cfg.task.env.render.data_type = "rgb"
        cfg.task.env.render.resize = [256, 256]
        cfg.task.env.render.instruction_template = (
            "Pick up the red cube and stack it on the blue cube."
        )
        cfg.task.env.enableCameraSensors = render
        cfg.force_render = render


def create_env(cfg: DictConfig):
    return isaacgymenvs.make(
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


def normalized_actions(env, trajectories: list[StackTrajectory]) -> torch.Tensor:
    actions = np.stack([item.actions for item in trajectories]).astype(np.float32)
    grasp_z_offset = float(
        os.environ.get("STACK_GRASP_WORLD_Z_OFFSET_M", DEFAULT_GRASP_WORLD_Z_OFFSET_M)
    )
    place_z_offset = float(
        os.environ.get("STACK_PLACE_WORLD_Z_OFFSET_M", DEFAULT_PLACE_WORLD_Z_OFFSET_M)
    )
    z_offsets = np.zeros(NUM_STEPS, dtype=np.float32)
    z_offsets[10:12] = grasp_z_offset
    z_offsets[12:32] = np.linspace(grasp_z_offset, place_z_offset, 20)
    z_offsets[32:34] = place_z_offset
    actions[..., 2] += z_offsets[None, :]
    lower = float(env.robot_dof_lower_limits[env.active_hand_dof_indices[0]].item())
    upper = float(env.robot_dof_upper_limits[env.active_hand_dof_indices[0]].item())
    actions[..., 7] = 2.0 * (actions[..., 7] - lower) / (upper - lower) - 1.0
    actions[..., 7] = np.clip(actions[..., 7], -1.0, 1.0)
    print(
        "Isaac Gym TCP world-z calibration: "
        f"grasp={grasp_z_offset:+.4f} m, place={place_z_offset:+.4f} m"
    )
    return torch.from_numpy(actions).to(env.device)


def set_stack_scenes(env, trajectories: list[StackTrajectory]) -> None:
    if len(trajectories) != env.num_envs:
        raise ValueError(f"Expected {env.num_envs} trajectories, got {len(trajectories)}")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    cube_a = np.stack([item.cube_a for item in trajectories]).astype(np.float32)
    cube_b = np.stack([item.cube_b for item in trajectories]).astype(np.float32)
    cube_a[:, 0] += TABLE_FRAME_X_M
    cube_b[:, 0] += TABLE_FRAME_X_M

    object_indices = env.object_indices[env_ids]
    target_indices = env.distractor_object_indices[env_ids, 0]
    env.root_state_tensor[object_indices, :3] = torch.from_numpy(cube_a).to(env.device)
    env.root_state_tensor[target_indices, :3] = torch.from_numpy(cube_b).to(env.device)
    identity_quaternion = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device
    ).unsqueeze(0).expand(env.num_envs, -1)
    env.root_state_tensor[object_indices, 3:7] = identity_quaternion
    env.root_state_tensor[target_indices, 3:7] = identity_quaternion
    env.root_state_tensor[object_indices, 7:] = 0.0
    env.root_state_tensor[target_indices, 7:] = 0.0
    actor_indices = torch.cat([object_indices, target_indices]).to(torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_state_tensor),
        gymtorch.unwrap_tensor(actor_indices),
        len(actor_indices),
    )

    for env_index, env_ptr in enumerate(env.envs):
        object_handle = env.gym.find_actor_handle(env_ptr, "object")
        target_handle = env.gym.find_actor_handle(env_ptr, "distractor")
        env.gym.set_rigid_body_color(
            env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*CUBE_A_RGB)
        )
        env.gym.set_rigid_body_color(
            env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*CUBE_B_RGB)
        )
    env.instructions = [env.instruction_template] * env.num_envs
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.compute_observations()


def settle(env, physics_steps: int = 45) -> None:
    for _ in range(physics_steps):
        env.gym.set_dof_position_target_tensor(
            env.sim, gymtorch.unwrap_tensor(env.cur_targets)
        )
        env.gym.simulate(env.sim)
        if env.device == "cpu":
            env.gym.fetch_results(env.sim, True)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def stack_metrics(env, cube_size_m: float) -> tuple[np.ndarray, list[dict]]:
    a = env.root_state_tensor[env.object_indices]
    b = env.root_state_tensor[env.distractor_object_indices[:, 0]]
    xy_error = torch.linalg.vector_norm(a[:, :2] - b[:, :2], dim=1)
    z_delta = a[:, 2] - b[:, 2]
    speed = torch.linalg.vector_norm(a[:, 7:10], dim=1)
    target_speed = torch.linalg.vector_norm(b[:, 7:10], dim=1)
    success = (
        (xy_error <= cube_size_m * 0.45)
        & (torch.abs(z_delta - cube_size_m) <= cube_size_m * 0.30)
        & (a[:, 2] >= cube_size_m * 1.25)
        & (torch.abs(b[:, 2] - cube_size_m / 2.0) <= cube_size_m * 0.25)
        & (speed <= 0.12)
        & (target_speed <= 0.12)
    )
    details = [
        {
            "success": bool(success[index].item()),
            "xy_error_m": float(xy_error[index].item()),
            "z_delta_m": float(z_delta[index].item()),
            "cube_a_height_m": float(a[index, 2].item()),
            "cube_b_height_m": float(b[index, 2].item()),
            "cube_a_speed_mps": float(speed[index].item()),
            "cube_b_speed_mps": float(target_speed[index].item()),
        }
        for index in range(env.num_envs)
    ]
    return success.detach().cpu().numpy(), details


def replay(
    env,
    trajectories: list[StackTrajectory],
    collect_observations: bool = False,
    trace: bool = False,
):
    actions = normalized_actions(env, trajectories)
    buffers = []
    trace_env = int(os.environ.get("STACK_TRACE_ENV", "0"))
    for step in range(NUM_STEPS):
        if collect_observations:
            observation = env.compute_real_observation_dict()
            observation["action"] = actions[:, step].detach().cpu().numpy()
            buffers.append(observation)
        env.step(actions[:, step])
        if trace and step in {0, 6, 7, 8, 9, 10, 11, 12, 20, 30, NUM_STEPS - 1}:
            eef = env.rigid_body_states.view(-1, 13)[env.eef_idx[trace_env], :7]
            object_state = env.root_state_tensor[env.object_indices[trace_env], :3]
            hand = env.robot_dof_pos[trace_env, env.active_hand_dof_indices]
            robot_handle = env.gym.find_actor_handle(env.envs[trace_env], "robot")
            left_index = env.gym.find_actor_rigid_body_index(
                env.envs[trace_env], robot_handle, "panda_leftfinger", gymapi.DOMAIN_SIM
            )
            right_index = env.gym.find_actor_rigid_body_index(
                env.envs[trace_env], robot_handle, "panda_rightfinger", gymapi.DOMAIN_SIM
            )
            rigid_states = env.rigid_body_states.view(-1, 13)
            print(
                f"trace env={trace_env} step={step} target={actions[trace_env, step].tolist()} "
                f"eef={eef.tolist()} hand={hand.tolist()} object={object_state.tolist()} "
                f"left={rigid_states[left_index, :3].tolist()} "
                f"right={rigid_states[right_index, :3].tolist()}"
            )
    settle(env)
    return buffers


def select_successful(env, cube_size_m: float) -> list[StackTrajectory]:
    candidates = load_all_trajectories(RAW_ROOT, num_steps=NUM_STEPS)
    by_scene: dict[int, list[StackTrajectory]] = defaultdict(list)
    for item in candidates:
        by_scene[item.scene_index].append(item)
    selected: dict[int, list[tuple[StackTrajectory, dict]]] = defaultdict(list)

    for batch in range(10):
        batch_records = []
        for scene_index in range(NUM_SCENES):
            start = batch * PER_SCENE
            batch_records.extend(by_scene[scene_index][start : start + PER_SCENE])
        set_stack_scenes(env, batch_records)
        trace = batch == 0 and os.environ.get("STACK_TRACE") == "1"
        replay(env, batch_records, trace=trace)
        successes, details = stack_metrics(env, cube_size_m)
        for record, is_success, detail in zip(batch_records, successes, details):
            if is_success and len(selected[record.scene_index]) < PER_SCENE:
                selected[record.scene_index].append((record, detail))
        counts = [len(selected[index]) for index in range(NUM_SCENES)]
        print(f"selection batch {batch + 1}/10: successes per scene={counts}")
        if trace:
            trace_env = int(os.environ.get("STACK_TRACE_ENV", "0"))
            print(f"selection batch 1 final metrics env={trace_env}: {details[trace_env]}")
        if os.environ.get("STACK_DIAGNOSTIC_ONLY") == "1":
            break
        if all(count == PER_SCENE for count in counts):
            break

    missing = {index: PER_SCENE - len(selected[index]) for index in range(NUM_SCENES)}
    missing = {index: count for index, count in missing.items() if count}
    if missing:
        raise RuntimeError(f"Could not find 10 physical successes in every scene: {missing}")
    flat = [item for scene_index in range(NUM_SCENES) for item, _ in selected[scene_index]]
    manifest = {
        "selection_version": 1,
        "raw_root": str(RAW_ROOT),
        "selection_rule": "10 physically successful trajectories from each of 10 scenes",
        "success_thresholds": {
            "max_xy_error_m": cube_size_m * 0.45,
            "max_z_error_m": cube_size_m * 0.30,
            "max_linear_speed_mps": 0.12,
        },
        "trajectories": [
            {**item.manifest_record(), "replay_metrics": detail}
            for scene_index in range(NUM_SCENES)
            for item, detail in selected[scene_index]
        ],
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved 100 selected demonstrations to {SELECTION_PATH}")
    return flat


def collect_v2(env, cfg: DictConfig, geometry) -> Path:
    writer = LerobotDatasetWriter(
        output_path=V2_DATASET_NAME,
        camera_ids=env.camera_ids,
        data_type=env.render_data_type,
        action_dim=env.num_actions,
        state_dim=7 + env.num_active_hand_dofs,
        image_shape=(*env.render_cfg["resize"], 3),
        fps=round(1 / env.dt / env.decimation),
    )
    writer.write_collection_metadata(
        {
            "scene_manifest_version": 1,
            "task": "cube_stacking",
            "raw_geometry": geometry.as_dict(),
            "stack_visuals": {"cube_a_rgb": CUBE_A_RGB, "cube_b_rgb": CUBE_B_RGB},
            "table_frame_x_m": TABLE_FRAME_X_M,
            "grasp_world_z_offset_m": DEFAULT_GRASP_WORLD_Z_OFFSET_M,
            "place_world_z_offset_m": DEFAULT_PLACE_WORLD_Z_OFFSET_M,
            "selected_trajectory_manifest": str(SELECTION_PATH),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    candidates = load_all_trajectories(RAW_ROOT, num_steps=NUM_STEPS)
    by_scene: dict[int, list[StackTrajectory]] = defaultdict(list)
    for item in candidates:
        by_scene[item.scene_index].append(item)
    selected: dict[int, list[tuple[StackTrajectory, dict, int]]] = defaultdict(list)
    saved_episodes = 0

    num_batches = max(len(items) for items in by_scene.values()) // PER_SCENE
    for collection_batch in range(num_batches):
        batch_records = []
        for scene_index in range(NUM_SCENES):
            start = collection_batch * PER_SCENE
            batch_records.extend(by_scene[scene_index][start : start + PER_SCENE])
        if not batch_records:
            continue
        if len(batch_records) != env.num_envs:
            raise ValueError(
                f"Collection batch has {len(batch_records)} trajectories for "
                f"{env.num_envs} environments"
            )
        set_stack_scenes(env, batch_records)
        env_ids = torch.arange(env.num_envs, device=env.device)
        scene_states = env.capture_scene_states(env_ids)
        buffers = replay(env, batch_records, collect_observations=True)
        successes, details = stack_metrics(env, geometry.cube_size_m)

        for env_index, (trajectory, scene, metric, is_success) in enumerate(
            zip(batch_records, scene_states, details, successes)
        ):
            scene_index = trajectory.scene_index
            if not is_success or len(selected[scene_index]) >= PER_SCENE:
                continue
            dataset_episode = saved_episodes
            actual_plan = [
                frame["action"][env_index].astype(np.float32).tolist() for frame in buffers
            ]
            for step, frame in enumerate(buffers):
                writer.append_step(
                    {key: value[env_index : env_index + 1] for key, value in frame.items()},
                    episode_end=step == NUM_STEPS - 1,
                    episode_metadata={
                        **scene,
                        "collection_batch": collection_batch,
                        "source_env_index": env_index,
                        "raw_trajectory": trajectory.manifest_record(),
                        "replay_metrics": metric,
                        "expert_plan": actual_plan,
                    }
                    if step == NUM_STEPS - 1
                    else None,
                )
            selected[scene_index].append((trajectory, metric, dataset_episode))
            saved_episodes += 1
            print(
                f"Saved v2 episode {saved_episodes}/100 from collection batch "
                f"{collection_batch}, env {env_index}"
            )

        counts = [len(selected[index]) for index in range(NUM_SCENES)]
        print(
            f"RGB collection batch {collection_batch + 1}/10: "
            f"successful episodes per scene={counts}"
        )
        if all(count == PER_SCENE for count in counts):
            break

    missing = {index: PER_SCENE - len(selected[index]) for index in range(NUM_SCENES)}
    missing = {index: count for index, count in missing.items() if count}
    if missing:
        raise RuntimeError(f"Could not collect 10 RGB successes in every scene: {missing}")

    manifest = {
        "selection_version": 2,
        "raw_root": str(RAW_ROOT),
        "selection_rule": "10 RGB-rendered physical successes from each of 10 scenes",
        "success_thresholds": {
            "max_xy_error_m": geometry.cube_size_m * 0.45,
            "max_z_error_m": geometry.cube_size_m * 0.30,
            "max_linear_speed_mps": 0.12,
        },
        "trajectories": [
            {
                **item.manifest_record(),
                "dataset_episode": dataset_episode,
                "replay_metrics": detail,
            }
            for scene_index in range(NUM_SCENES)
            for item, detail, dataset_episode in selected[scene_index]
        ],
    }
    SELECTION_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dataset_root = REPO_ROOT / "data/datasets" / V2_DATASET_NAME
    print(f"Created LeRobot v2 dataset at {dataset_root}")
    return dataset_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_augmented_v2(env, cfg: DictConfig, geometry) -> Path:
    """Collect original, scene-jittered, and corrective successful trajectories."""
    writer = LerobotDatasetWriter(
        output_path=AUGMENTED_V2_DATASET_NAME,
        camera_ids=env.camera_ids,
        data_type=env.render_data_type,
        action_dim=env.num_actions,
        state_dim=7 + env.num_active_hand_dofs,
        image_shape=(*env.render_cfg["resize"], 3),
        fps=round(1 / env.dt / env.decimation),
    )
    writer.write_collection_metadata(
        {
            "scene_manifest_version": 1,
            "task": "cube_stacking",
            "dataset_role": "augmented_training_only",
            "raw_geometry": geometry.as_dict(),
            "stack_visuals": {"cube_a_rgb": CUBE_A_RGB, "cube_b_rgb": CUBE_B_RGB},
            "table_frame_x_m": TABLE_FRAME_X_M,
            "grasp_world_z_offset_m": DEFAULT_GRASP_WORLD_Z_OFFSET_M,
            "place_world_z_offset_m": DEFAULT_PLACE_WORLD_Z_OFFSET_M,
            "frozen_benchmark_manifest": str(SELECTION_PATH),
            "frozen_benchmark_manifest_sha256": _sha256(SELECTION_PATH),
            "augmentation": {
                "kinds": ["scene_jitter", "correction"],
                "scene_xy_max_m": 0.008,
                "approach_correction_xy_max_m": 0.012,
                "physical_success_filter": True,
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    originals = load_selected_trajectories(RAW_ROOT, SELECTION_PATH, num_steps=NUM_STEPS)
    all_by_scene: dict[int, list[StackTrajectory]] = defaultdict(list)
    for trajectory in load_all_trajectories(RAW_ROOT, num_steps=NUM_STEPS):
        all_by_scene[trajectory.scene_index].append(trajectory)

    targets = {"original": PER_SCENE, "scene_jitter": PER_SCENE, "correction": PER_SCENE}
    counts = {
        kind: {scene_index: 0 for scene_index in range(NUM_SCENES)}
        for kind in targets
    }
    saved_records = []
    successful_originals: dict[int, list[StackTrajectory]] = defaultdict(list)
    collection_batch = 0

    def collect_batch(batch_records: list[StackTrajectory], kinds: list[str]) -> None:
        nonlocal collection_batch
        set_stack_scenes(env, batch_records)
        env_ids = torch.arange(env.num_envs, device=env.device)
        scene_states = env.capture_scene_states(env_ids)
        buffers = replay(env, batch_records, collect_observations=True)
        successes, details = stack_metrics(env, geometry.cube_size_m)
        for env_index, (trajectory, kind, scene, metric, is_success) in enumerate(
            zip(batch_records, kinds, scene_states, details, successes)
        ):
            scene_index = trajectory.scene_index
            if not is_success or counts[kind][scene_index] >= targets[kind]:
                continue
            dataset_episode = len(saved_records)
            actual_plan = [
                frame["action"][env_index].astype(np.float32).tolist() for frame in buffers
            ]
            for step, frame in enumerate(buffers):
                writer.append_step(
                    {key: value[env_index : env_index + 1] for key, value in frame.items()},
                    episode_end=step == NUM_STEPS - 1,
                    episode_metadata={
                        **scene,
                        "collection_batch": collection_batch,
                        "source_env_index": env_index,
                        "raw_trajectory": trajectory.manifest_record(),
                        "augmentation_kind": kind,
                        "replay_metrics": metric,
                        "expert_plan": actual_plan,
                    }
                    if step == NUM_STEPS - 1
                    else None,
                )
            counts[kind][scene_index] += 1
            if kind == "original":
                successful_originals[scene_index].append(trajectory)
            saved_records.append(
                {
                    **trajectory.manifest_record(),
                    "augmentation_kind": kind,
                    "dataset_episode": dataset_episode,
                    "replay_metrics": metric,
                }
            )
        print(
            f"Augmented collection batch {collection_batch}: "
            + ", ".join(
                f"{kind}={[counts[kind][scene] for scene in range(NUM_SCENES)]}"
                for kind in targets
            )
        )
        collection_batch += 1

    collect_batch(originals, ["original"] * len(originals))

    # A trajectory that succeeded during the historical collection can fail after
    # a fresh simulator initialization. Fill every scene from raw JSON candidates
    # and use only successes from this exact run as augmentation bases.
    for retry_round in range(10):
        if all(counts["original"][scene] >= PER_SCENE for scene in range(NUM_SCENES)):
            break
        retry_records = []
        for scene_index in range(NUM_SCENES):
            start = retry_round * PER_SCENE
            retry_records.extend(all_by_scene[scene_index][start : start + PER_SCENE])
        collect_batch(retry_records, ["original"] * len(retry_records))

    missing_originals = {
        scene: PER_SCENE - counts["original"][scene]
        for scene in range(NUM_SCENES)
        if counts["original"][scene] < PER_SCENE
    }
    if missing_originals:
        raise RuntimeError(
            f"Could not collect 10 fresh unperturbed successes per scene: {missing_originals}"
        )

    max_rounds = 12
    for round_index in range(max_rounds):
        batch_records = []
        kinds = []
        for scene_index in range(NUM_SCENES):
            bases = successful_originals[scene_index]
            for slot in range(PER_SCENE):
                kind = "scene_jitter" if slot < PER_SCENE // 2 else "correction"
                base = bases[(round_index * PER_SCENE + slot) % len(bases)]
                seed = 10_000 + round_index * 1_000 + scene_index * 100 + slot
                batch_records.append(perturb_trajectory(base, seed=seed, kind=kind))
                kinds.append(kind)
        collect_batch(batch_records, kinds)
        if all(
            counts[kind][scene] >= target
            for kind, target in targets.items()
            for scene in range(NUM_SCENES)
        ):
            break

    missing = {
        f"{kind}/scene_{scene}": target - counts[kind][scene]
        for kind, target in targets.items()
        for scene in range(NUM_SCENES)
        if counts[kind][scene] < target
    }
    if missing:
        raise RuntimeError(f"Could not collect balanced augmented successes: {missing}")
    if len(saved_records) != 300:
        raise RuntimeError(f"Expected 300 augmented training episodes, got {len(saved_records)}")

    manifest = {
        "selection_version": 3,
        "dataset_role": "augmented_training_only",
        "frozen_benchmark_manifest": str(SELECTION_PATH),
        "frozen_benchmark_manifest_sha256": _sha256(SELECTION_PATH),
        "counts": {kind: counts[kind] for kind in targets},
        "trajectories": saved_records,
    }
    AUGMENTED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUGMENTED_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    dataset_root = REPO_ROOT / "data/datasets" / AUGMENTED_V2_DATASET_NAME
    print(f"Created augmented LeRobot v2 dataset at {dataset_root}")
    return dataset_root


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    set_np_formatting()
    mode = str(cfg.get("stack_mode", "select"))
    if mode not in {"select", "collect", "collect_augmented"}:
        raise ValueError(f"Unsupported stack_mode: {mode}")
    geometry = infer_geometry(RAW_ROOT)
    expected_asset_size = 0.06
    if not np.isclose(geometry.cube_size_m, expected_asset_size, atol=1e-6):
        raise ValueError(
            f"cube.urdf is {expected_asset_size} m but JSON implies {geometry.cube_size_m} m"
        )
    select_with_render = os.environ.get("STACK_SELECT_WITH_RENDER") == "1"
    configure_stack_task(
        cfg,
        num_envs=100,
        render=mode in {"collect", "collect_augmented"} or select_with_render,
    )
    cfg.seed = set_seed(cfg.seed, torch_deterministic=True, rank=0)
    print(f"JSON-derived geometry: {geometry.as_dict()}")
    env = create_env(cfg)
    if mode == "select":
        select_successful(env, geometry.cube_size_m)
    elif mode == "collect":
        collect_v2(env, cfg, geometry)
    else:
        collect_augmented_v2(env, cfg, geometry)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
