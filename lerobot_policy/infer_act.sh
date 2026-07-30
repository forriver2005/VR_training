#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/dataset_env.sh"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${OUTPUT_DIR}}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_OUTPUT}/checkpoints/last/pretrained_model}"
INFERENCE_OUTPUT="${INFERENCE_OUTPUT:-${SCRIPT_DIR}/outputs/inference/${TRAIN_NAME}/episode_000000.npz}"
EPISODE="${EPISODE:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "ACT checkpoint not found at ${CHECKPOINT}. Run train_act.sh first." >&2
  exit 1
fi

conda run --no-capture-output -n lerobot python "${SCRIPT_DIR}/offline_inference.py" \
  --dataset "${DATASET_ROOT}" \
  --repo-id "${REPO_ID}" \
  --checkpoint "${CHECKPOINT}" \
  --episode "${EPISODE}" \
  --device "${DEVICE}" \
  --output "${INFERENCE_OUTPUT}"
