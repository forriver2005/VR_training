#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/dataset_env.sh"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
NUM_PROCESSES="${NUM_PROCESSES:-${NUM_GPUS:-1}}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Dataset not found at ${DATASET_ROOT}. Run prepare_dataset.sh first." >&2
  exit 1
fi

if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}" >&2
  exit 1
fi

# --policy.chunk_size="${CHUNK_SIZE}"
# --policy.n_action_steps="${N_ACTION_STEPS}"

TRAIN_ARGS=(
  --dataset.repo_id="${REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend=pyav
  --policy.type=act
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_DIR}"
  --job_name="${TRAIN_NAME}"
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --eval_freq=0
  --wandb.enable=true
)

if ! LEROBOT_TRAIN="$(command -v lerobot-train)"; then
  echo "lerobot-train not found. Activate the lerobot environment first." >&2
  exit 1
fi

if (( NUM_PROCESSES > 1 )); then
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate not found. Install it in the active lerobot environment." >&2
    exit 1
  fi
  accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_PROCESSES}" \
    --num_machines=1 \
    --mixed_precision="${MIXED_PRECISION}" \
    --dynamo_backend=no \
    "${LEROBOT_TRAIN}" \
    "${TRAIN_ARGS[@]}"
else
  "${LEROBOT_TRAIN}" "${TRAIN_ARGS[@]}"
fi
