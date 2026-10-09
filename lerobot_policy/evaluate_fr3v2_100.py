#!/usr/bin/env python3
"""Evaluate each recorded FR3V2 episode in its own Isaac Sim process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_PYTHON = Path("/home/houshengyuan/anaconda3/envs/lerobot/bin/python")
ISAAC_PYTHON = Path("/home/houshengyuan/env_isaacsim60/bin/python")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_for_server(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"policy server on port {port} did not start")


def evaluate_one(index: int, gpu: int, port: int, checkpoint: Path,
                 manifest: Path, output: Path, source: dict,
                 configured_scene: Path, urdf: Path,
                 static_friction: float, dynamic_friction: float, restitution: float) -> dict:
    episode_dir = output / f"episode_{index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA="YES",
               OMNI_KIT_ALLOW_ROOT="1")
    server = None
    try:
        with (episode_dir / "policy.log").open("w") as policy_log:
            server = subprocess.Popen(
                [str(LEROBOT_PYTHON), str(ROOT / "lerobot_policy/policy_server.py"),
                 "--checkpoint", str(checkpoint), "--port", str(port), "--seed", str(index),
                 "--n-action-steps", "1"],
                cwd=ROOT, env=env, stdout=policy_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_server(server, port)
            with (episode_dir / "simulation.log").open("w") as sim_log:
                simulation = subprocess.Popen(
                    [str(ISAAC_PYTHON), str(ROOT / "isaacsim_replay_collect.py"),
                     "--eval-manifest", str(manifest), "--eval-index", str(index),
                     "--policy-port", str(port), "--output", str(episode_dir / "simulation"),
                     "--review-video-dir", str(episode_dir / "video"), "--num-gpus", "1",
                     "--usd", str(configured_scene), "--urdf", str(urdf),
                     "--static-friction", str(static_friction),
                     "--dynamic-friction", str(dynamic_friction),
                     "--restitution", str(restitution)],
                    cwd=ROOT, env=env, stdout=sim_log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = simulation.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    stop_process(simulation)
                    raise TimeoutError(f"episode {index} exceeded 30 minutes")
                if code:
                    raise RuntimeError(f"Isaac Sim exited with {code}; see simulation.log")
        result_path = episode_dir / "simulation/replay_manifest.json"
        result = json.loads(result_path.read_text())["episodes"]
        if len(result) != 1 or result[0]["source_file"] != source["source_file"] or \
                result[0]["trajectory_id"] != source["trajectory_id"]:
            raise RuntimeError(f"episode {index} did not match its source trajectory")
        return {"episode_index": index, "source_file": source["source_file"],
                "trajectory_id": source["trajectory_id"], "success": result[0]["success"],
                "xy_error_m": result[0]["xy_error_m"], "z_delta_m": result[0]["z_delta_m"],
                "grasp_xy_error_m": result[0].get("grasp_xy_error_m", float("inf")),
                "result": str(result_path)}
    finally:
        if server is not None:
            stop_process(server)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--train-pid", type=int, help="Wait for this training process before evaluating")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 6, 7])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N ordered episodes for the physical gate")
    parser.add_argument("--configured-scene", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--static-friction", type=float, default=1.2)
    parser.add_argument("--dynamic-friction", type=float, default=1.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    args = parser.parse_args()
    if len(args.gpus) > 6 or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("Use at most six distinct GPUs")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 100 or [row["episode_index"] for row in rows] != list(range(100)):
        raise ValueError("Expected exactly 100 ordered source episodes")
    if args.limit is not None and not 1 <= args.limit <= 100:
        raise ValueError("--limit must be between 1 and 100")
    rows = rows[:args.limit] if args.limit is not None else rows
    if args.train_pid is not None:
        while process_alive(args.train_pid):
            time.sleep(15)
    checkpoint_dir = args.training_output / "checkpoints/last"
    step_info = json.loads((checkpoint_dir / "training_state/training_step.json").read_text())
    if step_info["step"] != args.steps:
        raise RuntimeError(f"Training stopped at {step_info['step']}, expected {args.steps}")
    checkpoint = (checkpoint_dir / "pretrained_model").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluation_config.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "training_step": args.steps,
        "source_manifest": str(args.manifest.resolve()), "gpus": args.gpus,
        "isolation": "one Isaac Sim process and one policy server per episode",
        "success_criteria": "same as replay collector physical stacking check",
        "episode_limit": args.limit,
    }, indent=2) + "\n")
    results = []
    result_lock = Lock()

    def evaluate_gpu(gpu: int, slot: int) -> None:
        for index in range(slot, len(rows), len(args.gpus)):
            try:
                row = evaluate_one(index, gpu, 5600 + gpu, checkpoint,
                                   args.manifest.resolve(), args.output.resolve(), rows[index],
                                   args.configured_scene.resolve(), args.urdf.resolve(),
                                   args.static_friction, args.dynamic_friction, args.restitution)
            except Exception as exc:
                row = {"episode_index": index, "source_file": rows[index]["source_file"],
                       "trajectory_id": rows[index]["trajectory_id"], "error": str(exc)}
            with result_lock:
                results.append(row)
                print(f"completed {len(results)}/{len(rows)} episode={index} result={row.get('success', row.get('error'))}", flush=True)
                (args.output / "progress.json").write_text(
                    json.dumps(sorted(results, key=lambda x: x["episode_index"]), indent=2) + "\n"
                )

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(evaluate_gpu, gpu, slot) for slot, gpu in enumerate(args.gpus)]
        for future in as_completed(futures):
            future.result()
    results.sort(key=lambda x: x["episode_index"])
    failures = [row for row in results if "error" in row]
    successes = sum(row.get("success", False) for row in results)
    summary = {"total": len(results), "completed": len(results) - len(failures),
               "successes": successes, "errors": len(failures),
               "success_rate": successes / len(results) if not failures else None,
               "episodes": results}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} episodes failed to evaluate; no success rate reported")


if __name__ == "__main__":
    main()
