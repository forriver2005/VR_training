#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATASET_EPISODES="${DATASET_EPISODES:-${NUM_EPISODES:-100}}"
source "${SCRIPT_DIR}/dataset_env.sh"

SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/data/datasets/fr3_inspire_tac_2026-07-30_18-54}"
START_EPISODE="${START_EPISODE:-0}"

args=(
  conda run --no-capture-output -n lerobot python "${SCRIPT_DIR}/prepare_dataset.py"
  --source "${SOURCE_DATASET}"
  --output "${DATASET_ROOT}"
  --repo-id "${REPO_ID}"
  --num-episodes "${DATASET_EPISODES}"
  --start-episode "${START_EPISODE}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
"${args[@]}"
