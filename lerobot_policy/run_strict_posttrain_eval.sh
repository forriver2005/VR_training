#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PID="${1:?training PID is required}"
TRAIN_OUTPUT="${ROOT}/lerobot_policy/outputs/trajectorysuccessful2_fr3_v3_imitation_strict_friction12_20260927_v2"
DATASET="${ROOT}/data/datasets/trajectorysuccessful2_fr3_v3_strict_friction12_20260927"
REPLAY="${ROOT}/data/isaacsim_trajectorysuccessful2_fr3v2_strict_friction12_20260927"
MANIFEST="${DATASET}/meta/scenes.jsonl"
SCENE="${ROOT}/data/isaacsim_trajectorysuccessful2_fr3v2_friction12_20260927/configured_scene.usda"
URDF="${ROOT}/assets/fr3_gripper/fr3_panda_gripper.urdf"
CHECKPOINT="${TRAIN_OUTPUT}/checkpoints/last/pretrained_model"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do sleep 60; done
test -f "${CHECKPOINT}/config.json"

PATH="/home/houshengyuan/anaconda3/envs/lerobot/bin:${PATH}" \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python \
  "${ROOT}/lerobot_policy/offline_inference.py" \
  --dataset "${DATASET}" --repo-id trajectorysuccessful2_fr3_v3_strict_friction12_20260927 \
  --checkpoint "${CHECKPOINT}" --policy-type diffusion --episode 0 --device cuda \
  --output "${TRAIN_OUTPUT}/training_episode_000000_reproduction.npz"

GATE_OUTPUT="${TRAIN_OUTPUT}/physical_gate3_v3"
PATH="/home/houshengyuan/anaconda3/envs/lerobot/bin:${PATH}" \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python \
  "${ROOT}/lerobot_policy/evaluate_fr3v2_100.py" \
  --training-output "${TRAIN_OUTPUT}" --manifest "${MANIFEST}" --output "${GATE_OUTPUT}" \
  --configured-scene "${SCENE}" --urdf "${URDF}" --steps 100000 --limit 3 --gpus 0 \
  --static-friction 1.2 --dynamic-friction 1.0 --restitution 0.0

PATH="/home/houshengyuan/anaconda3/envs/lerobot/bin:${PATH}" \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python - "${GATE_OUTPUT}/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
rows = summary["episodes"]
if len(rows) != 3 or any(
    not row.get("success", False)
    or row.get("xy_error_m", 1.0) >= 0.01
    or row.get("grasp_xy_error_m", 1.0) >= 0.01
    for row in rows
):
    raise SystemExit(f"physical gate failed: {summary}")
print("physical gate passed: 3/3, final and grasp XY errors < 1 cm")
PY

EVAL_OUTPUT="${TRAIN_OUTPUT}/eval100_v3"
PATH="/home/houshengyuan/anaconda3/envs/lerobot/bin:${PATH}" \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python \
  "${ROOT}/lerobot_policy/evaluate_fr3v2_100.py" \
  --training-output "${TRAIN_OUTPUT}" --manifest "${MANIFEST}" --output "${EVAL_OUTPUT}" \
  --configured-scene "${SCENE}" --urdf "${URDF}" --steps 100000 \
  --gpus 0 1 2 3 4 6 --static-friction 1.2 --dynamic-friction 1.0 --restitution 0.0
