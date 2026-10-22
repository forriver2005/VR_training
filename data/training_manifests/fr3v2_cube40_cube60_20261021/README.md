# FR3V2 4 cm / 6 cm training candidate lists

Prepared 2026-10-21 from the complete delivery audit. Source NPZ files are unchanged.

| List | Episodes | Scenes | Purpose |
|---|---:|---:|---|
| train_4hz.jsonl | 398 | 4 | Training candidates |
| validation_4hz.jsonl | 100 | 1 | Checkpoint selection |
| test_4hz.jsonl | 84 | 1 | Held-out evaluation after freezing the model |
| overfit_4hz.jsonl | 1 | 1 | Intentional subset of train_4hz |
| reserve_8hz.jsonl | 194 | 2 | Separate 8 Hz protocol |
| excluded.jsonl | 124 | — | 23 rejected + 100 replay-quality failures + 1 FPS discrepancy |

`npz`, `package_root`, and `diagnostics` paths are relative to the DemoGrasp repository root.
`replay_report` is relative to `package_root`. Use `group/episode_id` as the unique identity.
The stored NPZ hashes come from the complete delivery audit and must be checked by the loader before use.

The legacy single-episode trainer does **not** accept these lists. The new
`lerobot_policy/train_delivery_diffusion.py` verifies hashes and loads these lists,
rebuilds wrist point clouds into a separate versioned cache, and includes history
validity and a command-history timer. The matching evaluator is
`lerobot_policy/run_delivery_trials.py`. Per-scene local replay and policy quality
gates are still required before full training. Do not combine reserve_8hz with
the 4 Hz lists in the baseline.

`validation_quick_4hz.jsonl` contains a fixed subset of 25 of the 100 validation
trajectories, selected by action-path coverage. The remaining 75 stay in validation.
See `quick_validation_selection.json` for the selection rule and identities.

Exact FPS reconstruction was sampled at 54 frames; all successful episodes were checked
for geometric consistency with wrist depth. These are candidates, not evidence of a
fresh local physics replay or trained-policy success.

See [network review and startup workflow](../../../reports/network_training_readiness_20261021/report.md).
