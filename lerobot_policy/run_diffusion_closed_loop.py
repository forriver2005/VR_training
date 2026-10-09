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
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in range(args.first_seed,args.first_seed+args.count):
        trial=args.output/f'seed_{seed:02d}'; trial.mkdir(parents=True,exist_ok=True)
        port=5600+seed; server=None
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
                '--policy-rgb-codec','raw','--eval-episode-npz',str(args.episode_npz),
                '--num-gpus','1','--headless'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.sim_gpu,
                OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1'),stdout=sim_log,stderr=subprocess.STDOUT,
                start_new_session=True)
            code=sim.wait(timeout=1800)
            sim_log.close(); log.close()
            if code: raise RuntimeError(f'Isaac Sim exited {code}')
            report=json.loads((trial/'simulation'/'replay_manifest.json').read_text())['episodes'][0]
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
            stop(server)
    summary={'checkpoint':str(args.checkpoint.resolve()),'count':len(results),'successes':sum(r.get('success',False) for r in results),
        'success_rate':sum(r.get('success',False) for r in results)/max(1,len(results)),'trials':results,
        'fixed_scene_manifest':str(args.manifest.resolve()),'policy_contract':'H=2 K=16 A=8 EMA DDPM100'}
    (args.output/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
