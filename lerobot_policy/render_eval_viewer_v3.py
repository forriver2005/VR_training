#!/usr/bin/env python3
"""Render selected v3 ACT episodes from the Isaac Gym viewer, not policy cameras."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks  # noqa: F401,E402
from lerobot_policy.isaacgym_v3_closed_loop import (  # noqa: E402
    configure,
    create_env,
    environment_action,
    observation,
    scene_rows,
    set_scene,
)
from lerobot_policy.isaacgym_closed_loop import PolicyClient  # noqa: E402


def write_viewer_frame(env, path: Path) -> None:
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)
    env.gym.write_viewer_image_to_file(env.viewer, str(path))


def encode(frames: Path, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
        "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg) -> None:
    manifest = Path(str(cfg.v3_manifest)).expanduser().resolve()
    checkpoint = Path(str(cfg.v3_checkpoint)).expanduser().resolve()
    output = Path(str(cfg.v3_output)).expanduser().resolve()
    selected = [int(value) for value in str(cfg.get("v3_selected", "0,17,1,9")).split(",")]
    rows = scene_rows(manifest, None)
    by_index = {int(row["episode_index"]): row for row in rows}
    missing = [index for index in selected if index not in by_index]
    if missing:
        raise ValueError(f"Selected scene indices are missing: {missing}")

    set_np_formatting()
    configure(cfg)
    cfg.v3_headless = False
    cfg.headless = False
    cfg.seed = set_seed(42, torch_deterministic=True, rank=0)
    env = create_env(cfg)
    env.gym.viewer_camera_look_at(
        env.viewer, None, gymapi.Vec3(1.8, -3.2, 2.7), gymapi.Vec3(0.6, 0.05, 0.10)
    )
    client = PolicyClient("127.0.0.1", int(cfg.get("v3_policy_port", 5555)), 30.0, 60.0)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index in selected:
            row = by_index[index]
            frame_dir = output / f"scene_{index:06d}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            set_scene(env, row)
            client.reset()
            write_viewer_frame(env, frame_dir / "frame_0000.png")
            for step in range(40):
                obs, _ = observation(env)
                action = client.predict(obs)
                env.step(environment_action(env, action))
                write_viewer_frame(env, frame_dir / f"frame_{step + 1:04d}.png")
            for _ in range(120):
                env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
            write_viewer_frame(env, frame_dir / "frame_0041.png")
            encode(frame_dir, output / f"episode_{index:06d}_physical_viewer.mp4")
            print(json.dumps({"episode_index": index, "video": str(output / f"episode_{index:06d}_physical_viewer.mp4")}), flush=True)
    finally:
        client.close()
    print(json.dumps({"selected": selected, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
