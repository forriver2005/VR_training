#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/dataset_env.sh"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-5555}"
POLICY_GPU="${POLICY_GPU:-0}"
SIM_GPU="${SIM_GPU:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-10}"
HEADLESS="${HEADLESS:-False}"
RECORD_VIDEO="${RECORD_VIDEO:-True}"
RENDER_RANDOMIZE="${RENDER_RANDOMIZE:-True}"
SEEN_DATA="${SEEN_DATA:-False}"

# 训练场景的分项恢复开关；全部默认 False，即使用本次 eval 新随机出的值。
# USE_TRAIN_SCENE_SELECTION: 使用 manifest 中的训练 episode、物体 ID 和原始 env 槽位。
# USE_TRAIN_OBJECT_STATE: 恢复物体初始位置/旋转/速度，以及干扰物状态。
# USE_TRAIN_ROBOT_STATE: 恢复机器人 root、关节状态和位置 target。
# USE_TRAIN_CAMERA_STATE: 恢复相机载体 pose 和 depth range。
# USE_TRAIN_TABLE_STATE: 恢复桌子、垫子、墙面 actor pose 和桌面高度。
# USE_TRAIN_VISUAL: 恢复物体/背景纹理和颜色。
# USE_TRAIN_LIGHTING: 恢复全局灯光强度、环境光和方向。
# USE_TRAIN_TASK_CONFIG: 使用采集时保存的完整 task 配置。
# SEEN_DATA=True 会覆盖以下全部开关，相当于全部设为 True。
# 任一 USE_TRAIN_*=True 都需要 manifest，并会自动使用采集时的原始 env 数量。
# 示例：USE_TRAIN_OBJECT_STATE=True USE_TRAIN_VISUAL=True ./run_isaacgym_eval.sh
USE_TRAIN_SCENE_SELECTION="${USE_TRAIN_SCENE_SELECTION:-False}"
USE_TRAIN_OBJECT_STATE="${USE_TRAIN_OBJECT_STATE:-False}"
USE_TRAIN_ROBOT_STATE="${USE_TRAIN_ROBOT_STATE:-False}"
USE_TRAIN_CAMERA_STATE="${USE_TRAIN_CAMERA_STATE:-True}"
USE_TRAIN_TABLE_STATE="${USE_TRAIN_TABLE_STATE:-True}"
USE_TRAIN_VISUAL="${USE_TRAIN_VISUAL:-True}"
USE_TRAIN_LIGHTING="${USE_TRAIN_LIGHTING:-True}"
USE_TRAIN_TASK_CONFIG="${USE_TRAIN_TASK_CONFIG:-False}" # 初步发现这一项开启时会报错

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/sim_eval}"
OBJECT_LIST="${OBJECT_LIST:-union_ycb_unidex/union_ycb_debugset.yaml}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${DATASET_ROOT}/meta/scenes.jsonl}"
COLLECTION_CONFIG="${COLLECTION_CONFIG:-${DATASET_ROOT}/meta/collection_config.json}"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "ACT checkpoint not found at ${CHECKPOINT}" >&2
  exit 1
fi

replay_requested=False
for replay_option in \
  "${SEEN_DATA}" \
  "${USE_TRAIN_SCENE_SELECTION}" \
  "${USE_TRAIN_OBJECT_STATE}" \
  "${USE_TRAIN_ROBOT_STATE}" \
  "${USE_TRAIN_CAMERA_STATE}" \
  "${USE_TRAIN_TABLE_STATE}" \
  "${USE_TRAIN_VISUAL}" \
  "${USE_TRAIN_LIGHTING}" \
  "${USE_TRAIN_TASK_CONFIG}"; do
  case "${replay_option,,}" in
    true|1|yes) replay_requested=True ;;
  esac
done

case "${replay_requested}" in
  True)
    if [[ ! -f "${SCENE_MANIFEST}" ]]; then
      echo "Training-data replay requires scene manifest: ${SCENE_MANIFEST}" >&2
      exit 1
    fi
    if [[ ! -f "${COLLECTION_CONFIG}" ]]; then
      echo "Training-data replay requires collection config: ${COLLECTION_CONFIG}" >&2
      exit 1
    fi
    ;;
esac

if [[ "${HOST}" == "127.0.0.1" || "${HOST}" == "localhost" ]] && \
  ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  echo "Policy server port ${HOST}:${PORT} is already in use" >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -- "-${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
setsid env CUDA_VISIBLE_DEVICES="${POLICY_GPU}" conda run --no-capture-output -n lerobot \
  python "${SCRIPT_DIR}/policy_server.py" \
  --checkpoint "${CHECKPOINT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device cuda &
server_pid=$!

CUDA_VISIBLE_DEVICES="${SIM_GPU}" conda run --no-capture-output -n demograsp \
  python "${SCRIPT_DIR}/isaacgym_closed_loop.py" \
  task=grasp \
  task.env.asset.multiObjectList="${OBJECT_LIST}" \
  task.env.armController=pose \
  task.env.render.enable=True \
  task.env.enableCameraSensors=True \
  task.env.render.camera_ids="[1,2]" \
  task.env.render.data_type=rgb \
  task.env.render.randomize="${RENDER_RANDOMIZE}" \
  task.env.episodeLength=40 \
  num_envs="${NUM_ENVS}" \
  headless="${HEADLESS}" \
  force_render=True \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  +policy_host="${HOST}" \
  +policy_port="${PORT}" \
  +seen_data="${SEEN_DATA}" \
  +use_train_scene_selection="${USE_TRAIN_SCENE_SELECTION}" \
  +use_train_object_state="${USE_TRAIN_OBJECT_STATE}" \
  +use_train_robot_state="${USE_TRAIN_ROBOT_STATE}" \
  +use_train_camera_state="${USE_TRAIN_CAMERA_STATE}" \
  +use_train_table_state="${USE_TRAIN_TABLE_STATE}" \
  +use_train_visual="${USE_TRAIN_VISUAL}" \
  +use_train_lighting="${USE_TRAIN_LIGHTING}" \
  +use_train_task_config="${USE_TRAIN_TASK_CONFIG}" \
  +scene_manifest="${SCENE_MANIFEST}" \
  +collection_config="${COLLECTION_CONFIG}" \
  +num_eval_episodes="${NUM_EPISODES}" \
  +record_eval_video="${RECORD_VIDEO}" \
  +eval_output_dir="${EVAL_OUTPUT_DIR}"
