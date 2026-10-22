"""Portable entry points for checking, training and evaluating the released policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'lerobot_policy'))
from repro_paths import learning_python, isaac_python

TRAIN_LIST = ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021/train_4hz.jsonl'
EVAL_LIST = ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021/complete_regression_scenes.jsonl'


def launch(script, arguments, **env):
    subprocess.run([learning_python(), str(ROOT/'lerobot_policy'/script), *map(str, arguments)],
                   cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', **env), check=True)


def verify_source():
    manifest = json.loads((ROOT/'repro/source_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:raise ValueError('Released file changed: '+name)
    print('Source and checkpoint hashes verified.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    sub.add_parser('unit')
    for name in ['smoke', 'regression']:
        p = sub.add_parser(name)
        p.add_argument('--checkpoint', type=Path, default=ROOT/'repro/checkpoint')
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--sim-gpus', nargs='+', default=['0'])
        p.add_argument('--server-gpu', default='0')
    p = sub.add_parser('train')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('derive')
    p.add_argument('--checkpoint', type=Path, required=True, help='Selected training EMA directory')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'doctor':
        verify_source()
        for name, python, versions in [
            ('learning', learning_python(), {'numpy':'2.2.6','torch':'2.10.0','lerobot':'0.4.4','diffusers':'0.35.2'}),
            ('simulation', isaac_python(), {'numpy':'2.3.1','isaacsim':'6.0.0.0','ikpy':'3.4.2'})]:
            code = ('import importlib.metadata as m,json,sys; expected='+repr(versions)+'; '
                    'actual={k:m.version(k) for k in expected}; print(sys.executable,json.dumps(actual)); '
                    'assert all(actual[k].split("+")[0]==v for k,v in expected.items()), "Dependency version mismatch"')
            print('CHECK', name, flush=True);subprocess.run([python, '-c', code], check=True)
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv'], check=True)
    elif args.command == 'unit':
        # Synthetic contract tests need no original raw dataset or running simulator.
        modules = ['test_canonical_goal','test_episode_anchor','test_goal_anchor_action','test_goal_gate',
                   'test_goal_geometry_v2','test_goal_memory','test_goal_terminal_constraints',
                   'test_inference_constraints','test_monotonic_goal_gate','test_position_loss',
                   'test_relative_action','test_release_guard','test_smooth_joint_targets',
                   'test_spatial_pointcloud','test_transport_clearance_guard','test_layout_variants',
                   'test_repro_artifacts']
        subprocess.run([learning_python(), '-m', 'unittest', *modules], cwd=ROOT/'lerobot_policy',
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1'), check=True)
    elif args.command in ['smoke','regression']:
        isaac_python()  # Fail before allocating workers if the simulator is missing.
        gpus = set(args.sim_gpus + [args.server_gpu])
        if len(gpus)>6:raise ValueError('At most six physical GPUs are allowed')
        jobs = ['--indices', 0, 6, 23, '--seeds', 0] if args.command=='smoke' else ['--tasks', ROOT/'repro/regression_tasks.json']
        launch('run_delivery_sweep.py', ['--list', EVAL_LIST, *jobs, '--mode', 'policy',
               '--checkpoint', args.checkpoint.resolve(), '--output', args.output.resolve(),
               '--sim-gpus', *args.sim_gpus, '--server-gpu', args.server_gpu, '--port-base', 15000])
        result = json.loads((args.output/'summary.json').read_text())
        print(json.dumps({k:v for k,v in result.items() if k not in ['trials','groups']}, indent=2))
        if result['runtime_errors']:raise SystemExit('Incomplete experiment: runtime errors retained in summary')
    elif args.command == 'train':
        launch('train_delivery_diffusion.py', ['--list', TRAIN_LIST, '--output', args.output.resolve(),
            '--steps', args.steps, '--save-every', args.save_every, '--batch-size', args.batch_size, '--seed', args.seed,
            '--prediction-type', 'sample', '--timer-mode', 'bounded', '--pointcloud-source', 'wrist_depth',
            '--observation-frame', 'initial_tcp', '--encoder', 'proprioceptive', '--goal-memory', '--goal-recipe', 'v1',
            '--episode-goal-3d-actions', '--goal-state-switch', '--learned-goal-gate', '--position-loss-weight', 6],
            CUDA_VISIBLE_DEVICES=args.gpu)
    else:
        target=args.output.resolve()
        if target.exists():raise FileExistsError(target)
        stages=target.with_name(target.name+'-derivation');stages.mkdir(parents=True, exist_ok=False)
        parent=args.checkpoint.resolve()
        steps=[('derive_goal_recipe_checkpoint.py', []), ('derive_monotonic_gate_checkpoint.py', []),
               ('derive_action_steps_checkpoint.py', ['--action-steps',8,'--action-samples',4,
                 '--binary-gripper','--fixed-down-quaternion','--anchor-snap-epsilon',.05,'--goal-terminal-constraints']),
               ('derive_release_guard_checkpoint.py', []), ('derive_smooth_controller_checkpoint.py', ['--tau',.04]),
               ('derive_transport_guard_checkpoint.py', [])]
        for index,(script,extra) in enumerate(steps):
            output=target if index==len(steps)-1 else stages/str(index)
            launch(script, ['--checkpoint',parent,'--output',output,*extra]);parent=output
        print('Derived policy:', target)


if __name__ == '__main__':
    main()
