#!/usr/bin/env python3
"""Parse and resample the raw cube-stacking trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAIN_TRAJECTORY_RE = re.compile(r"^Trajectory_(\d+)\.json$")


@dataclass(frozen=True)
class StackTrajectory:
    source_file: str
    scene_index: int
    trajectory_id: int
    cube_a: np.ndarray
    cube_b: np.ndarray
    actions: np.ndarray
    augmentation: dict | None = None

    def manifest_record(self) -> dict:
        record = {
            "source_file": self.source_file,
            "scene_index": self.scene_index,
            "trajectory_id": self.trajectory_id,
        }
        if self.augmentation is not None:
            record["augmentation"] = self.augmentation
        return record


@dataclass(frozen=True)
class StackGeometry:
    cube_size_m: float
    gripper_open_m: float
    gripper_closed_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cube_size_m": self.cube_size_m,
            "gripper_open_m": self.gripper_open_m,
            "gripper_closed_m": self.gripper_closed_m,
        }


def _xyz(value: dict) -> np.ndarray:
    return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float32)


def _euler_xyz_to_quaternion(eulers: np.ndarray) -> np.ndarray:
    half = eulers * 0.5
    cx, cy, cz = (np.cos(half[:, axis]) for axis in range(3))
    sx, sy, sz = (np.sin(half[:, axis]) for axis in range(3))
    return np.column_stack(
        (
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
    ).astype(np.float32)


def trajectory_files(raw_root: Path) -> list[Path]:
    files = []
    for path in raw_root.iterdir():
        match = MAIN_TRAJECTORY_RE.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if len(files) != 10:
        raise ValueError(f"Expected 10 trajectory files in {raw_root}, found {len(files)}")
    return [path for _, path in files]


def infer_geometry(raw_root: Path) -> StackGeometry:
    """Infer and cross-check task geometry encoded by every source file."""
    half_extents = []
    open_angles = []
    closed_angles = []
    for path in trajectory_files(raw_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        half_extents.extend(
            [
                float(payload["defaultCubeAStartPosition"]["z"]),
                float(payload["defaultCubeBStartPosition"]["z"]),
            ]
        )
        open_angles.append(float(payload["gripperOpenAngle"]))
        closed_angles.append(float(payload["gripperClosedAngle"]))
    if not np.allclose(half_extents, half_extents[0], atol=1e-6):
        raise ValueError("Cube center heights are inconsistent across source files")
    if not np.allclose(open_angles, open_angles[0], atol=1e-6):
        raise ValueError("Gripper open angles are inconsistent across source files")
    if not np.allclose(closed_angles, closed_angles[0], atol=1e-6):
        raise ValueError("Gripper closed angles are inconsistent across source files")
    cube_size = 2.0 * half_extents[0]
    if not np.isclose(2.0 * open_angles[0], cube_size, atol=1e-6):
        raise ValueError(
            "The cube center-height and two-sided gripper opening imply different sizes"
        )
    return StackGeometry(
        cube_size_m=cube_size,
        gripper_open_m=open_angles[0],
        gripper_closed_m=closed_angles[0],
    )


def _resample_actions(
    trajectory: dict, num_steps: int, gripper_open_m: float, gripper_closed_m: float
) -> np.ndarray:
    points = trajectory["points"]
    rotations = trajectory["rotations"]
    gripper_states = trajectory["gripperStates"]
    if not (len(points) == len(rotations) == len(gripper_states)):
        raise ValueError(f"Trajectory {trajectory['trajectoryId']} has inconsistent lengths")
    if len(points) < 2:
        raise ValueError(f"Trajectory {trajectory['trajectoryId']} is too short")

    # The source stores xyz Euler angles in the x/y/z fields. The w field is a
    # constant serialization placeholder, not the scalar component of a quaternion.
    if num_steps != 40:
        raise ValueError("Cube stacking replay currently requires 40 output steps")
    transitions = [
        index
        for index in range(1, len(gripper_states))
        if bool(gripper_states[index]) != bool(gripper_states[index - 1])
    ]
    if len(transitions) != 2 or not gripper_states[transitions[0]] or gripper_states[transitions[1]]:
        raise ValueError(
            f"Trajectory {trajectory['trajectoryId']} must contain one close and one open event"
        )
    close_index, open_index = transitions

    # Preserve the manipulation events when compressing 50 Hz source paths to the
    # 40-step DemoGrasp collection horizon. Dwells at grasp and release let the
    # Isaac Gym controller reach the event pose before the next path segment.
    source_samples = np.concatenate(
        [
            np.linspace(0, close_index, 10),
            np.full(2, close_index),
            np.linspace(close_index + 1, open_index - 1, 20),
            np.full(2, open_index),
            np.linspace(min(open_index + 1, len(points) - 1), len(points) - 1, 6),
        ]
    )
    sampled_gripper = np.asarray(
        [gripper_open_m] * 11
        + [gripper_closed_m] * 21
        + [gripper_open_m] * 8,
        dtype=np.float32,
    )[:, None]
    source_times = np.arange(len(points), dtype=np.float64)
    positions = np.asarray([_xyz(point) for point in points], dtype=np.float32)
    eulers = np.asarray(
        [[rotation[axis] for axis in "xyz"] for rotation in rotations], dtype=np.float64
    )
    quaternions = _euler_xyz_to_quaternion(eulers)

    sampled_positions = np.column_stack(
        [np.interp(source_samples, source_times, positions[:, axis]) for axis in range(3)]
    ).astype(np.float32)
    nearest = np.rint(source_samples).astype(np.int64)
    sampled_quaternions = quaternions[nearest]
    actions = np.concatenate(
        [sampled_positions, sampled_quaternions, sampled_gripper], axis=1
    )
    if actions.shape != (num_steps, 8) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid resampled action array: {actions.shape}")
    return actions


def load_all_trajectories(raw_root: Path, num_steps: int = 40) -> list[StackTrajectory]:
    records = []
    for scene_index, path in enumerate(trajectory_files(raw_root)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trajectories = payload["trajectories"]
        if payload.get("totalTrajectories") not in (None, len(trajectories)):
            raise ValueError(
                f"Declared trajectory count does not match {path}: "
                f"{payload.get('totalTrajectories')} != {len(trajectories)}"
            )
        if not trajectories:
            raise ValueError(f"No trajectories in {path}")
        cube_a = _xyz(payload["defaultCubeAStartPosition"])
        cube_b = _xyz(payload["defaultCubeBStartPosition"])
        gripper_open_m = float(payload["gripperOpenAngle"])
        gripper_closed_m = float(payload["gripperClosedAngle"])
        for expected_id, trajectory in enumerate(trajectories):
            trajectory_id = int(trajectory["trajectoryId"])
            if trajectory_id != expected_id:
                raise ValueError(f"Non-contiguous trajectory ids in {path}")
            records.append(
                StackTrajectory(
                    source_file=path.name,
                    scene_index=scene_index,
                    trajectory_id=trajectory_id,
                    cube_a=cube_a.copy(),
                    cube_b=cube_b.copy(),
                    actions=_resample_actions(
                        trajectory, num_steps, gripper_open_m, gripper_closed_m
                    ),
                )
            )
    return records


def load_selected_trajectories(
    raw_root: Path, selection_path: Path, num_steps: int = 40
) -> list[StackTrajectory]:
    all_records = load_all_trajectories(raw_root, num_steps=num_steps)
    by_key = {(item.source_file, item.trajectory_id): item for item in all_records}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = []
    for record in selection["trajectories"]:
        key = (record["source_file"], int(record["trajectory_id"]))
        if key not in by_key:
            raise ValueError(f"Selection references unknown trajectory: {key}")
        selected.append(by_key[key])
    if len(selected) != 100 or len({(x.source_file, x.trajectory_id) for x in selected}) != 100:
        raise ValueError("Selection must contain 100 unique trajectories")
    return selected


def perturb_trajectory(
    trajectory: StackTrajectory,
    *,
    seed: int,
    kind: str,
    scene_xy_m: float = 0.008,
    correction_xy_m: float = 0.012,
) -> StackTrajectory:
    """Create an action-consistent scene perturbation or corrective expert path."""
    if kind not in {"scene_jitter", "correction"}:
        raise ValueError(f"Unsupported perturbation kind: {kind}")
    rng = np.random.default_rng(seed)
    cube_a_delta = np.r_[rng.uniform(-scene_xy_m, scene_xy_m, 2), 0.0].astype(
        np.float32
    )
    cube_b_delta = np.r_[rng.uniform(-scene_xy_m, scene_xy_m, 2), 0.0].astype(
        np.float32
    )

    actions = trajectory.actions.copy()
    place_blend = np.zeros(len(actions), dtype=np.float32)
    place_blend[12:32] = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    place_blend[32:] = 1.0
    path_delta = (
        (1.0 - place_blend[:, None]) * cube_a_delta[None, :]
        + place_blend[:, None] * cube_b_delta[None, :]
    )
    actions[:, :3] += path_delta

    correction_delta = np.zeros(3, dtype=np.float32)
    if kind == "correction":
        correction_delta[:2] = rng.uniform(-correction_xy_m, correction_xy_m, 2)
        # Deliberately deviate during approach, then return to the correct grasp pose.
        recovery_profile = np.zeros(len(actions), dtype=np.float32)
        recovery_profile[:6] = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        recovery_profile[6:11] = np.linspace(1.0, 0.0, 5, dtype=np.float32)
        actions[:, :3] += recovery_profile[:, None] * correction_delta[None, :]

    augmentation = {
        "kind": kind,
        "seed": seed,
        "cube_a_xy_delta_m": cube_a_delta[:2].tolist(),
        "cube_b_xy_delta_m": cube_b_delta[:2].tolist(),
        "approach_correction_xy_m": correction_delta[:2].tolist(),
    }
    return StackTrajectory(
        source_file=trajectory.source_file,
        scene_index=trajectory.scene_index,
        trajectory_id=trajectory.trajectory_id,
        cube_a=trajectory.cube_a + cube_a_delta,
        cube_b=trajectory.cube_b + cube_b_delta,
        actions=actions,
        augmentation=augmentation,
    )
