#!/usr/bin/env python3
"""Run ten fixed-seed Isaac Sim closed-loop trials for the RGB overfit checkpoint."""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()

def wait_server(port, process):
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError(f'server exited {process.returncode}')
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=1): return
        except OSError: time.sleep(.5)
    raise TimeoutError('policy server did not start')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/datasets/trajectorysuccessful2_fr3_v3_strict_friction12_20260927/meta/scenes.jsonl')
    p.add_argument('--usd',type=Path,default=ROOT/'data/isaacsim_trajectorysuccessful2_fr3v2_friction12_20260927/configured_scene.usda')
    p.add_argument('--urdf',type=Path,default=ROOT/'assets/fr3_gripper/fr3_panda_gripper.urdf')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isaac-python',default='/home/houshengyuan/env_isaacsim60/bin/python')
    p.add_argument('--lerobot-python',default='/home/houshengyuan/anaconda3/envs/lerobot/bin/python')
    p.add_argument('--server-device',default='cuda:0')
    p.add_argument('--server-gpu',default='1')
    p.add_argument('--sim-gpu',default='0')
    p.add_argument('--first-seed',type=int,default=0)
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--episode-npz',type=Path,default=ROOT/'data/isaacsim_trajectorysuccessful2_fr3v2_strict_friction12_20260927/episode_000000.npz')
    p.add_argument('--restore-recorded-joints',action='store_true',help='diagnostic only; standard reset is default')
    p.add_argument('--pointcloud',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads((args.checkpoint/'run_config.json').read_text())
    if (contract.get('modality','rgb')=='pointcloud')!=args.pointcloud:
        raise ValueError('Checkpoint modality and --pointcloud disagree')
    control_dt=contract.get('control_dt_s',.125)
    physics_steps=round(control_dt*120)
    if abs(physics_steps/120-control_dt)>1e-8:
        raise ValueError('Control period is not an integer number of physics steps')
    results=[]
    for seed in range(args.first_seed,args.first_seed+args.count):
        trial=args.output/f'seed_{seed:02d}'; trial.mkdir(parents=True,exist_ok=True)
        port=5600+seed; server=sim=log=sim_log=None
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.server_gpu,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1')
            log=(trial/'policy.log').open('w')
            server=subprocess.Popen([args.lerobot_python,str(ROOT/'lerobot_policy/diffusion_policy_server.py'),
                '--checkpoint',str(args.checkpoint),'--port',str(port),'--device',args.server_device,'--seed',str(seed)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            wait_server(port,server)
            sim_log=(trial/'simulation.log').open('w')
            sim=subprocess.Popen([args.isaac_python,str(ROOT/'isaacsim_replay_collect.py'),
                '--eval-manifest',str(args.manifest),'--eval-index','0','--policy-port',str(port),
                '--output',str(trial/'simulation'),'--usd',str(args.usd),'--urdf',str(args.urdf),
                '--policy-rgb-codec','raw',
                '--physics-steps-per-frame',str(physics_steps),
                '--reset-settle-steps',str(contract.get('reset_settle_steps',123)),
                *(['--eval-episode-npz',str(args.episode_npz)] if args.restore_recorded_joints else []),
                *(['--save-rgbd-pcd'] if args.pointcloud else []),
                '--num-gpus','1','--headless'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.sim_gpu,
                OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1'),stdout=sim_log,stderr=subprocess.STDOUT,
                start_new_session=True)
            code=sim.wait(timeout=1800)
            sim_log.close(); log.close()
            if code: raise RuntimeError(f'Isaac Sim exited {code}')
            report=json.loads((trial/'simulation'/'replay_manifest.json').read_text())['episodes'][0]
            import numpy as np
            if not np.allclose(np.diff(report['observation_time_s']),control_dt,rtol=0,atol=1e-6):
                raise RuntimeError('Closed-loop observation timing mismatch')
            result={'seed':seed,'success':bool(report['success']),'xy_error_m':report['xy_error_m'],
                'z_delta_m':report['z_delta_m'],'ik_failures':report['ik_failures'],
                'action_clipping_count':report['policy_action_clipping_count'],
                'quaternion_fallback_count':report['policy_quaternion_fallback_count'],
                'inference_calls':report['policy_inference_calls'],
                'result':str((trial/'simulation'/'replay_manifest.json').resolve())}
            results.append(result); print(json.dumps(result),flush=True)
        except Exception as exc:
            result={'seed':seed,'success':False,'error':repr(exc)}
            results.append(result); print(json.dumps(result),flush=True)
        finally:
            stop(sim)
            stop(server)
            if log is not None: log.close()
            if sim_log is not None: sim_log.close()
    summary={'checkpoint':str(args.checkpoint.resolve()),'count':len(results),'successes':sum(r.get('success',False) for r in results),
        'success_rate':sum(r.get('success',False) for r in results)/max(1,len(results)),'trials':results,
        'fixed_scene_manifest':str(args.manifest.resolve()),'policy_contract':'H=2 K=16 A=8 EMA DDPM100',
        'control_dt_s':control_dt,'reset':'recorded joint telemetry diagnostic' if args.restore_recorded_joints else 'standard IK and settle',
        'runtime_errors':sum('error' in r for r in results)}
    (args.output/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
