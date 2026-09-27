#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/houshengyuan/anaconda3/envs/lerobot/bin/python"
TRAIN="/home/houshengyuan/anaconda3/envs/lerobot/bin/lerobot-train"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/datasets/trajectorysuccessful2_fr3_v3_isaacgym_box_20260927}"
REPO_ID="${REPO_ID:-trajectorysuccessful2_fr3_v3_isaacgym_box_20260927}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/lerobot_policy/outputs/trajectorysuccessful2_fr3_v3_act_isaacgym_box_20260927}"

test -f "${DATASET_ROOT}/meta/info.json"
test -f "${DATASET_ROOT}/meta/scenes.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="/home/houshengyuan/anaconda3/envs/lerobot/bin:${PATH}"

exec "${TRAIN}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=40 \
  --policy.n_obs_steps=1 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=trajectorysuccessful2_fr3_v3_act_overfit_box_20260927 \
  --steps="${STEPS:-20000}" \
  --resume="${RESUME:-false}" \
  --batch_size="${BATCH_SIZE:-16}" \
  --num_workers="${NUM_WORKERS:-16}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ:-2000}" \
  --log_freq="${LOG_FREQ:-100}" \
  --eval_freq=0 \
  --wandb.enable=false
