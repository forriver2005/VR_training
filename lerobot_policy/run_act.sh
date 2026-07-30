#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_EPISODES="${DATASET_EPISODES:-${NUM_EPISODES:-100}}"
source "${SCRIPT_DIR}/dataset_env.sh"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  "${SCRIPT_DIR}/prepare_dataset.sh"
fi
"${SCRIPT_DIR}/train_act.sh"
"${SCRIPT_DIR}/infer_act.sh"
