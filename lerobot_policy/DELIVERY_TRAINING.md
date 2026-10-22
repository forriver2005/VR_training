# Delivery diffusion workflow

The old `train_diffusion_overfit.py` / `run_diffusion_closed_loop.py` pipeline is
retained for old checkpoints. Use the delivery entry points for the new 4 cm / 6 cm task.

All commands run from the DemoGrasp repository root. Output directories must be new.

```bash
# Already frozen: 25 validation trajectories, all from the validation scene.
/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/select_quick_validation.py

# Expert Cartesian commands through the same IK/controller as the policy.
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \
  /home/houshengyuan/env_isaacsim60/bin/python lerobot_policy/run_delivery_episode.py \
  --list data/training_manifests/fr3v2_cube40_cube60_20261021/overfit_4hz.jsonl \
  --mode expert_ik --output lerobot_policy/outputs/new_expert_check

# Single-trajectory baseline. For a documented clean-action diffusion ablation,
# add --prediction-type sample and use a separate output directory.
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/train_delivery_diffusion.py \
  --list data/training_manifests/fr3v2_cube40_cube60_20261021/overfit_4hz.jsonl \
  --output lerobot_policy/outputs/new_delivery_overfit \
  --steps 20000 --save-every 2000 --batch-size 8

# Offline teacher-forced action reconstruction, 10 seeds by default.
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  /home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/evaluate_delivery_diffusion.py \
  --checkpoint lerobot_policy/outputs/new_delivery_overfit/checkpoints/020000/ema \
  --list data/training_manifests/fr3v2_cube40_cube60_20261021/overfit_4hz.jsonl \
  --output lerobot_policy/outputs/new_delivery_offline

# Independent closed loop, latest live observations, strict 5 mm XY gate.
/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/run_delivery_trials.py \
  --checkpoint lerobot_policy/outputs/new_delivery_overfit/checkpoints/020000/ema \
  --list data/training_manifests/fr3v2_cube40_cube60_20261021/overfit_4hz.jsonl \
  --output lerobot_policy/outputs/new_delivery_closed_loop \
  --server-gpu 2 --sim-gpu 0 --count 10
```

For multi-episode training, replace the input with `train_4hz.jsonl`; start with
batch 32 and an explicitly frozen schedule. Evaluate checkpoints on
`validation_quick_4hz.jsonl` (e.g. `--seeds 0`) before broader validation. Do not use
the test list to select checkpoints. `run_delivery_trials.py` evaluates the first
entry; `run_delivery_episode.py --index` selects one entry, and
`run_delivery_sweep.py --indices ... --seeds ...` performs a multi-scene sweep in
independent processes. `monitor_delivery_validation.py` evaluates a fixed
validation list at saved checkpoints and freezes a selection by validation
mean position error.

The shared contract uses 4 Hz, H=2/K=16/A=8, EMA DDPM100, a history validity bit,
and a timer derived only from already executed commands. `--timer-mode anchor`
uses the original predicate anchored to the first command of the current hold,
with position 1 mm, orientation 1 degree, and finger 0.5 mm tolerances.
`--timer-mode bounded` uses adjacent-command tolerances of position 2.5 mm and
finger 0.5 mm, with cumulative anchor limits of position 5 mm and finger 1 mm;
orientation remains within 1 degree of the anchor. This prevents indefinite
accumulation when targets drift slowly. The timer never changes the executed
action. Timer input is clipped to
[0,2] seconds and divided by 2. Initial value is zero. This is a measured-command
condition, not an expert phase label.

For a new experiment using clean-action prediction and bounded hold timing, add
`--prediction-type sample --timer-mode bounded` to the training command and use
a new output directory. Always derive offline timer features with the saved
checkpoint's rule, as `evaluate_delivery_diffusion.py` does.

The pilot `sample20k_hold_v2/ema` is an explicitly documented inference ablation
of the sample-20k weights. All 40 pilot training timer inputs and all 10 offline
predictions were verified identical under the two timer rules; weights were
copied without retraining. This equivalence is not assumed for other episodes.
For multi-episode training compute the bounded timer on all new data and start a
new run. The inference-ablation directory has no resumable optimizer state.

Point clouds are reconstructed from wrist depth with fixed base-frame crop,
raster reduction to at most 4096 candidates, then index-0 FPS; squared distances
are rounded to 12 decimal places to explicitly resolve equal-distance choices.
Derived caches are keyed by source hash and preprocessing-code hash, stored in
`data/derived/delivery_pointcloud_v2/`. Source NPZ and package checksums are unchanged.

Position/action normalization uses fixed simulation bounds X=[.15,.95],
Y=[-.55,.55], Z=[0,.95] metres; quaternions retain [-1,1] components and finger
targets use [0,.04] metres. These numerical bounds do not substitute for IK and
collision validation. The runtime uses the archived FR3V2/TCP and driver setup,
but evaluates original strict XY=5 mm, height difference=50±4 mm, drift<1 mm,
speed<50 mm/s, release/contact and IK conditions.

The four selected training scenes and one held-out validation scene have now
passed local representative expert replay. The runner uses the validated latest
archived runtime for all scenes, verifies the shared dynamics contract and code
hashes, and loads each package's own scene, URDF and reset data. Vulkan uses the
physical GPU index in CUDA_VISIBLE_DEVICES; the CUDA physics ordinal is zero.
Other scenes and episodes still require appropriate replay checks. No new data
may be silently clipped into the old bounds.

The 20-trajectory experiment achieved 19/20 on training layouts and 0/10 on the
held-out validation layout; do not treat it as a generalization success. See
[the full report](../reports/multiscene_pilot20_quality_20261021/report.md).

## Observation and action ablations

`--pointcloud-source fused_depth` constructs 1024 XYZ points using fixed-camera
quotas 256/256 and wrist quota 512. Empty views are omitted and their quota is
assigned to remaining measured views; no object masks or synthetic object points
are used. Global/wrist geometry remains in the same FR3 base frame and crop.
`wrist_depth` remains the default. Offline evaluation and live capture dispatch
from the saved checkpoint contract; caches are isolated by source and recipe.

`--relative-actions` changes only XYZ output labels to displacement from the
planning TCP. Observations remain in base coordinates. The decoder adds that
anchor exactly once before enqueuing absolute targets; rotation/finger semantics
and the controller interface remain unchanged. Relative output normalization is
stored separately from observation normalization and absolute execution limits.

The fused absolute-action trial achieved 8/20 seen and 0/10 held-out; fused with
relative XYZ output achieved 5/20 and 0/10. Neither replaces the original baseline.
See [the ablation report](../reports/fused_pilot20_quality_20261021/report.md).

`run_delivery_sweep.py --tasks <json>` accepts a frozen list of `{index, seed}`
pairs, allowing different seed counts by scene in one sweep. Repeat a GPU in
`--sim-gpus` only when it has sufficient memory for independent simulator
processes. Completed runs require the result, NPZ, GIF and completion marker;
all policy failures and runtime errors remain in the summary.

Resuming requires the same data, normalization, preprocessing, diffusion target,
seed, batch size and total schedule:

```bash
# Add this to the same training command:
# --resume <output>/checkpoints/010000/training_state.pt
```

The checkpoint saves raw/EMA weights, optimizer, scheduler, all RNG state and the
full data/processing contract. CPU interrupted/uninterrupted continuation was
checked for exact raw/EMA equality on a short run. DDP is not implemented.

## Existing-data optimization (2026-10-21)

The user now owns further data generation. The local 16-layout collection was
stopped during independent replay and is excluded from the existing-data study.
The frozen `train_4hz.jsonl` remains the only training source (398 trajectories,
4 backgrounds); original validation and test manifests remain unchanged.

`train_delivery_diffusion.py --relative-actions --observation-frame current_tcp`
centers both H=2 point clouds and both measured XYZ states at the latest planning
TCP. Axes remain parallel to the FR3 base. This differs from the earlier
output-only relative-action trial. The common model conditioning implements the
same transform during training and inference. Raw depth/point clouds are not
changed or synthesized.

`--encoder spatial_tokens` is a local experimental encoder: 128 deterministic
point landmarks, 16 neighbors per landmark, local shared features, positional
encoding, and two 128D Transformer layers. The readout remains 64D and the
Diffusion Policy action model is retained. This is not a full reproduction of
RISE/iDP3 and does not establish SOTA. Both flags default to the old behavior;
loading uses the saved run contract, with strict weight matching.

`run_existing398_study.py` freezes three 40k/batch32/seed42 experiments and evaluates
the original validation25 at 10k intervals, followed by independent closed loops.
The two TCP-centered fusion variants isolate the encoder choice. Comparing either
against wrist/base changes both source and representation and is not a
single-factor attribution.

`derive_action_steps_checkpoint.py --action-steps 2` makes an explicit inference
ablation with identical copied weights, H=2/K=16, and more frequent replanning.
It preserves the parent checkpoint and records its hash; it does not retrain.

## Measured goals and consistent execution

The latest candidates use a measured RGB-D target plus actual robot state and
initial-state memory. The `proprioceptive` encoder is an explicit ablation with no
learned point-cloud features; it must not be described as a PointNet result.
The original wrist PointNet baseline and all comparison checkpoints are retained.

`rgbd_goal_geometry_v2.py` estimates the green 6cm target from depth surface
normals and depth-supported silhouettes. Known color, size and axis alignment
are task priors. Flat color reflections do not establish an object pose. The
estimator receives RGB-D and camera/base calibration only. Simulator object
poses appear only in post-rollout diagnostics and success evaluation.

The network uses actual `observation[0].state` as its initial robot reference.
A learned transport classifier changes conditioning to the measured target frame.
The optional monotonic inference memory retains a predicted transport event, or
an actually returned nonzero transport anchor, until an explicit episode reset.
This prevents the reference frame from reverting after opening the fingers.
Expert phase labels supervise this classifier and the extra anchor channel during
training; no expert phase or absolute frame index is supplied at inference.

The nine latent outputs decode to eight physical action channels. The optional
valid-action projection fixes the orientation and binary gripper alphabet only
after checking every training trajectory supports these constraints. Goal-end
constraints use the measured target XY and the known stack height. A separate
release guard delays a **learned** opening request until the nominal placement
command and measured robot pose have settled, then holds position while opening.
It does not inspect object poses, contacts or rewards, and it does not declare
stacking success; the original independent physics evaluator does that.

An optional 40ms first-order joint/gripper reference filter runs at120Hz, after
the original reset. It preserves static drive gains and the desired4Hz action
interface, while changing inner-loop actuation transients. Its use must be
reported as a controller ablation. Joint and gripper reference traces are saved.
All these options are checkpoint contracts; defaults preserve older models.

The first frozen test under the new state/goal architecture scored2/20 and was
subsequently used for debugging. Later runs on that layout are regression tests.
Additional physical layouts were reserved from the existing8Hz packages: only
their assets and measured reset are used for a4Hz/40-action scene evaluation.
Their raw demonstrations are not relabelled or used for training. Evaluation-only
manifests are rejected by the trainer, and source-rate mismatch is explicit in
rollout metadata. Check `frozen_final_candidate.json` and the final report for the
selected candidate and genuinely fresh evaluation results once completed.

New data must retain consistent action/TCP semantics. The current anchor labels
verify the40-frame expert phase schedule. If phase lengths or grasp/place TCP
height change, update label derivation and execution-contract validation from
metadata before training; do not silently reuse the existing phase constants.

The current `height_aware_policy` adds a transport-clearance guard based on the
measured destination height. It raises before horizontal carry when needed,
preserves the learned horizontal direction with bounded steps, and allows
vertical placement after reaching the target column. It consumes robot/goal
observations and predicted transport/release events, not obstacle truth maps.
The two formerly fresh layout packages were used to diagnose the elevated-goal
failure, so their subsequent results are regression results. The final report
must retain both earlier frozen-test outcomes (2/20 and19/40) and distinguish
local combination checks from genuinely new obstacle backgrounds.
