"""Measure synchronized policy latency on recorded observations, without simulation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_overfit_common import Inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Explicit observation whitelist: cube poses, contacts and success are not inputs.
    with np.load(args.rollout, allow_pickle=False) as archive:
        observations = {k: archive[k] for k in
                        ['state', 'point_cloud', 'elapsed_same_target_s', 'goal_features']}
        recorded_actions = archive['action']
    model = Inference(args.checkpoint, 'cuda', seed=0)
    rows = []
    reference_difference = None
    for repeat in range(-1, args.repeats):
        model.reset(0)
        actions = []
        for frame in range(len(observations['state'])):
            previous_calls = model.inference_calls
            torch.cuda.synchronize()
            start = time.perf_counter()
            action, _ = model.predict(
                observations['state'][frame:frame+1],
                point_cloud=observations['point_cloud'][frame:frame+1],
                elapsed_same_target_s=observations['elapsed_same_target_s'][frame:frame+1],
                goal_features=observations['goal_features'][frame:frame+1])
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            actions.append(action[0])
            if repeat >= 0:
                rows.append(dict(repeat=repeat, frame=frame, duration_s=duration,
                                 replanned=model.inference_calls > previous_calls))
        reference_difference = float(np.max(np.abs(np.asarray(actions) - recorded_actions)))
    def stats(values):
        return dict(count=len(values), mean_ms=float(np.mean(values)*1000),
                    p50_ms=float(np.percentile(values, 50)*1000),
                    p95_ms=float(np.percentile(values, 95)*1000), max_ms=max(values)*1000)
    result = dict(checkpoint=str(args.checkpoint.resolve()), rollout=str(args.rollout.resolve()),
                  device=torch.cuda.get_device_name(), action_samples=model.action_samples,
                  cuda_synchronized=True, warmup_episodes=1, repetitions=args.repeats,
                  model_replanning=stats([r['duration_s'] for r in rows if r['replanned']]),
                  queued_action=stats([r['duration_s'] for r in rows if not r['replanned']]),
                  all_calls=stats([r['duration_s'] for r in rows]),
                  reference_action_max_absolute_difference=reference_difference,
                  scope='Policy only; excludes camera capture, RGB-D geometry, IPC, IK and physics. '
                        'Synchronous simulator pauses physics while waiting; simulated4Hz is not guaranteed wall-clock4Hz.',
                  timings=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'timings'}, indent=2))


if __name__ == '__main__':
    main()
