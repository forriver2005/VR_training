#!/usr/bin/env bash

# Shared dataset/run naming for preparation, training, inference, and evaluation.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_EPISODES="${DATASET_EPISODES:-100}"

if ! [[ "${DATASET_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DATASET_EPISODES must be a positive integer, got: ${DATASET_EPISODES}" >&2
  return 1 2>/dev/null || exit 1
fi

DATASET_NAME="${DATASET_NAME:-demograsp_act${DATASET_EPISODES}}"
TRAIN_NAME="${TRAIN_NAME:-act_demograsp${DATASET_EPISODES}}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/artifacts/datasets/${DATASET_NAME}}"
REPO_ID="${REPO_ID:-local/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${TRAIN_NAME}}"

export DATASET_EPISODES DATASET_NAME TRAIN_NAME DATASET_ROOT REPO_ID OUTPUT_DIR
