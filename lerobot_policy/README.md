# DemoGrasp LeRobot ACT baseline

These scripts use LeRobot 0.4.4 without importing the DemoGrasp runtime. Activate
the `lerobot` environment before running `train_act.sh`; the training script invokes
the active environment's `lerobot-train` command directly. The source data is
LeRobot v2.0, so the first step creates an independent LeRobot v3.0 subset. The
source dataset is never modified.

## Run

From this directory:

```bash
conda activate lerobot
export DATASET_EPISODES=100
./prepare_dataset.sh
./train_act.sh
./infer_act.sh
```

Or run all three stages:

```bash
DATASET_EPISODES=100 ./run_act.sh
```

The defaults are:

- source: `../data/datasets/fr3_inspire_tac_2026-07-30_13-58`
- dataset size: `DATASET_EPISODES` (default: 100)
- v3 subset: `artifacts/datasets/demograsp_act${DATASET_EPISODES}`
- episodes: 0 through `DATASET_EPISODES - 1` by default (40 frames each)
- replay metadata: `meta/scenes.jsonl` and `meta/collection_config.json` are retained
  when they exist in the source dataset
- ACT training: 20,000 steps, batch size 128 per process, 40-step chunks, one CUDA device
- inference execution: one action per observation (closed-loop replanning)
- checkpoint: `outputs/act_demograsp${DATASET_EPISODES}/checkpoints/last/pretrained_model`
- inference: offline replay of episode 0, with predictions saved as NPZ and MAE as JSON

Common overrides are environment variables:

```bash
CUDA_VISIBLE_DEVICES=1 STEPS=50000 BATCH_SIZE=64 ./train_act.sh
EPISODE=10 DEVICE=cuda ./infer_act.sh
```

Set `DATASET_EPISODES` on every standalone stage, or export it once. Paths, repo id,
job name, and checkpoint location are derived from it consistently:

```bash
export DATASET_EPISODES=100
./prepare_dataset.sh
./train_act.sh
./infer_act.sh
```

`DATASET_NAME`, `TRAIN_NAME`, `DATASET_ROOT`, `REPO_ID`, and `OUTPUT_DIR` remain
individually overridable when custom naming is needed. `NUM_EPISODES` is retained as
a compatibility alias for dataset preparation; prefer `DATASET_EPISODES` because
`run_isaacgym_eval.sh` uses `NUM_EPISODES` for the number of evaluation episodes.

Multi-GPU training uses LeRobot's native Accelerate/DDP support. For example, this
runs one process on each of four visible GPUs:

```bash
DATASET_EPISODES=100 CUDA_VISIBLE_DEVICES=1,2,3,4,5 NUM_PROCESSES=5 ./train_act.sh
```

`NUM_GPUS` is accepted as an alias for `NUM_PROCESSES`. `BATCH_SIZE` is per process,
so the effective global batch size in the example above is `NUM_GPUS * BATCH_SIZE`.
Mixed precision is disabled by default; set `MIXED_PRECISION=fp16` or `bf16` to
enable it on supported GPUs.

To intentionally recreate the converted subset:

```bash
DATASET_EPISODES=100 OVERWRITE=1 ./prepare_dataset.sh
```

For a fast end-to-end smoke test, use a separate output directory:

```bash
STEPS=1 BATCH_SIZE=2 NUM_WORKERS=0 SAVE_FREQ=1 \
OUTPUT_DIR=outputs/act_smoke ./train_act.sh
CHECKPOINT=outputs/act_smoke/checkpoints/last/pretrained_model \
INFERENCE_OUTPUT=outputs/inference/smoke.npz ./infer_act.sh
```

`offline_inference.py` only evaluates recorded observations. Connecting the policy
to DemoGrasp simulation or real hardware requires an adapter that returns the same
two RGB observations and 13-dimensional state, then applies the predicted
13-dimensional action through the target controller.

## Isaac Gym closed-loop evaluation

The closed-loop evaluator keeps Isaac Gym in the `demograsp` conda environment and
runs ACT in the `lerobot` environment. A local TCP connection transfers batched raw
RGB observations and states; no extra package is required in either environment.

After formal training, run a headed evaluation with one environment:

```bash
DATASET_EPISODES=100 ./run_isaacgym_eval.sh
```

Evaluation defaults to 10 episodes and writes camera videos plus `summary.json` to
`outputs/sim_eval/<timestamp>`. Common overrides are:

```bash
DATASET_EPISODES=100 CHECKPOINT=/home/zhaolizhi/DexResearch/DemoGrasp/lerobot_policy/outputs/act_demograsp100/checkpoints/last/pretrained_model POLICY_GPU=1 SIM_GPU=0 NUM_EPISODES=100 HEADLESS=True NUM_ENVS=100 SEEN_DATA=False ./run_isaacgym_eval.sh

SEEN_DATA=True # replay recorded training scenes exactly
```

`OBJECT_LIST` can select another object YAML for targeted evaluation. Keep the
default debug set when measuring the same object distribution used for collection.

Set `SEEN_DATA=True` to replay recorded training scenes exactly:

```bash
SEEN_DATA=True NUM_EPISODES=20 ./run_isaacgym_eval.sh
```

This mode requires `meta/scenes.jsonl` and `meta/collection_config.json` in
`DATASET_ROOT`. It restores the first `NUM_EPISODES` recorded scenes, groups them by
their original collection batch, uses their original source environment slots, and
restores object/robot states, cameras, table actors, textures, colors, and global
lights. The evaluator automatically uses the original collection environment count;
`NUM_ENVS` is ignored in this mode. The manifest path and each source dataset episode
are written to `summary.json`. Older datasets without replay metadata are rejected.

Use `SCENE_MANIFEST` and `COLLECTION_CONFIG` to replay metadata stored outside
`DATASET_ROOT`.

For replay ablations, each scene component can independently use its recorded value
or a fresh reset value. Every option defaults to `False`:

```bash
USE_TRAIN_SCENE_SELECTION=True  # Recorded episode/source-env selection only.
USE_TRAIN_OBJECT_STATE=True     # Object pose/velocity and distractor states.
USE_TRAIN_ROBOT_STATE=True      # Robot root, DOF state, and targets.
USE_TRAIN_CAMERA_STATE=True     # Camera pads and depth ranges.
USE_TRAIN_TABLE_STATE=True      # Table/mat/wall poses and table height.
USE_TRAIN_VISUAL=True           # Object/background textures and colors.
USE_TRAIN_LIGHTING=True         # Global light intensity, ambient, and direction.
USE_TRAIN_TASK_CONFIG=True      # Full task config from collection_config.json.
```

Selecting any training component automatically enables recorded scene selection and
the original source environment count so object slots remain valid. For example,
this restores only object state and visual appearance while freshly resetting robot,
camera, and table state:

```bash
USE_TRAIN_OBJECT_STATE=True USE_TRAIN_VISUAL=True \
NUM_EPISODES=100 ./run_isaacgym_eval.sh
```

`SEEN_DATA=True` overrides all `USE_TRAIN_*` values and restores every component.

`NUM_ENVS=1` is recommended for viewer visualization. Batched headless evaluation
uses one ACT request for all environments and reports DemoGrasp's existing grasp
success metric. When video is enabled, it records both camera views for every
evaluated episode as `episode_<episode>_env_<env>.mp4`. The evaluator
requires the same `armController=pose`, two RGB cameras, 256x256 resize, and
40-step episode configuration used during collection.
