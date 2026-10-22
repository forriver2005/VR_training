"""Fixed-seed closed-loop trials through the new task's actual IK/physics path."""
import argparse,json,os,signal,socket,subprocess,time,secrets
from pathlib import Path
from delivery_policy_contract import ROOT
from socket_protocol import send_arrays,receive_arrays,command_message,read_command

def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--list',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=10);p.add_argument('--first-seed',type=int,default=0)
    p.add_argument('--server-gpu',default='2');p.add_argument('--sim-gpu',default='0')
    p.add_argument('--port-base',type=int,default=6300)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    contract=json.loads((args.checkpoint/'run_config.json').read_text())
    rows=[]
    for seed in range(args.first_seed,args.first_seed+args.count):
        server=sim=None;reserved_listener=None;trial=args.output/f'seed_{seed:02d}';trial.mkdir()
        port=args.port_base+seed
        env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',OPENBLAS_NUM_THREADS='1')
        try:
            with (trial/'policy.log').open('w') as log,(trial/'simulation.log').open('w') as sim_log:
                reserved_listener=socket.socket()
                try:reserved_listener.bind(('127.0.0.1',port))
                except OSError:
                    reserved_listener.close();reserved_listener=socket.socket();reserved_listener.bind(('127.0.0.1',0))
                port=reserved_listener.getsockname()[1];reserved_listener.listen(8);token=secrets.token_hex(16)
                server=subprocess.Popen(['/home/houshengyuan/anaconda3/envs/lerobot/bin/python',str(ROOT/'lerobot_policy/diffusion_policy_server.py'),
                    '--checkpoint',str(args.checkpoint.resolve()),'--port',str(port),'--seed',str(seed),'--listen-fd',str(reserved_listener.fileno()),'--ready-token',token],cwd=ROOT,
                    env=dict(env,CUDA_VISIBLE_DEVICES=args.server_gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(reserved_listener.fileno(),))
                deadline=time.monotonic()+120
                while True:
                    if server.poll() is not None:raise RuntimeError('Policy server exited')
                    try:
                        with socket.create_connection(('127.0.0.1',port),timeout=1) as ready:
                            send_arrays(ready,command_message('ping'));reply=receive_arrays(ready)
                            if read_command(reply)!='pong' or str(reply.get('ready_token',''))!=token:raise RuntimeError('Policy server identity mismatch')
                        break
                    except OSError:
                        if time.monotonic()>deadline:raise TimeoutError('Policy startup')
                        time.sleep(.5)
                reserved_listener.close();reserved_listener=None
                sim=subprocess.Popen(['/home/houshengyuan/env_isaacsim60/bin/python',str(ROOT/'lerobot_policy/run_delivery_episode.py'),
                    '--mode','policy','--list',str(args.list.resolve()),'--output',str((trial/'simulation').resolve()),
                    '--port',str(port),'--seed',str(seed),'--contract',str((args.checkpoint/'run_config.json').resolve()),'--server-token',token],cwd=ROOT,
                    env=dict(env,CUDA_VISIBLE_DEVICES=args.sim_gpu),stdout=sim_log,stderr=subprocess.STDOUT,start_new_session=True)
                if sim.wait(timeout=600)!=0:raise RuntimeError('Simulation exited with an error')
                if (trial/'simulation/error.json').exists() or not all((trial/'simulation'/name).is_file() for name in ['completed.json','result.json','rollout.npz','rollout.gif']):raise RuntimeError('Incomplete simulation artifacts')
                result=json.loads((trial/'simulation/result.json').read_text())
                row={k:result[k] for k in ['seed','success','failure_reasons','final_xy_error_m','final_z_error_m',
                    'stable_window_max_xy_error_m','ik_failures','action_clipping_count','quaternion_fallback_count','inference_calls','lifted']}
                row['result']=str((trial/'simulation/result.json').resolve())
                row['port']=port
        except Exception as exc:row=dict(seed=seed,success=False,error=repr(exc))
        finally:
            stop(sim);stop(server)
            if reserved_listener is not None:reserved_listener.close()
        rows.append(row);print(json.dumps(row),flush=True)
        summary=dict(checkpoint=str(args.checkpoint.resolve()),completed=len(rows),requested=args.count,
            successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
            runtime_errors=sum('error' in r for r in rows),protocol=f"4 Hz; H2 K16 A{contract['A']}; EMA DDPM100; 40 actions; strict XY5mm",trials=rows)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
