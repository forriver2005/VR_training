"""Parallel independent episode processes; retain all failures and partial progress."""
import argparse,json,os,socket,subprocess,time,secrets
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from delivery_policy_contract import ROOT,load_entries
from run_delivery_trials import stop
from socket_protocol import send_arrays,receive_arrays,command_message,read_command

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',type=Path,required=True)
    jobs=p.add_mutually_exclusive_group(required=True)
    jobs.add_argument('--indices',type=int,nargs='+');jobs.add_argument('--tasks',type=Path,help='JSON list of index/seed pairs')
    p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--mode',choices=['expert_ik','expert_joint','policy'],default='expert_ik')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-gpus',nargs='+',default=['0','3']);p.add_argument('--server-gpu',default='2')
    p.add_argument('--port-base',type=int,default=6600);args=p.parse_args()
    if args.mode=='policy' and args.checkpoint is None:raise ValueError('Policy needs a checkpoint')
    entries=load_entries(args.list)
    tasks=([(int(r['index']),int(r['seed'])) for r in json.loads(args.tasks.read_text())] if args.tasks
           else [(i,s) for i in args.indices for s in args.seeds])
    if not tasks or len(tasks)!=len(set(tasks)) or any(i<0 or i>=len(entries) or s<0 for i,s in tasks):
        raise ValueError('Invalid or duplicate trial specification')
    indices=sorted({i for i,s in tasks});args.output.mkdir(parents=True,exist_ok=False);completed=[]
    def run_queue(gpu,jobs):
        out=[]
        for job,(idx,seed) in jobs:
            entry=entries[idx];trial=args.output/f'index_{idx:03d}_seed_{seed:02d}';trial.mkdir()
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1')
            server=sim=None;reserved_listener=None;port=args.port_base+job
            row=dict(index=idx,seed=seed,uid=entry['uid'],scene_id=entry['scene_id'],sim_gpu=gpu,mode=args.mode)
            try:
                with (trial/'simulation.log').open('w') as log,(trial/'policy.log').open('w') as policy_log:
                    if args.mode=='policy':
                        # Reserve the listener BEFORE loading Torch/checkpoint.
                        # A plain successful TCP connection is not evidence that
                        # our server owns a port; other local HTTP services may.
                        requested_port=port;reserved_listener=socket.socket()
                        try:reserved_listener.bind(('127.0.0.1',port))
                        except OSError:
                            reserved_listener.close();reserved_listener=socket.socket();reserved_listener.bind(('127.0.0.1',0))
                        port=reserved_listener.getsockname()[1];reserved_listener.listen(8)
                        token=secrets.token_hex(16);row.update(requested_port=requested_port,port=port)
                        server=subprocess.Popen(['/home/houshengyuan/anaconda3/envs/lerobot/bin/python',str(ROOT/'lerobot_policy/diffusion_policy_server.py'),
                            '--checkpoint',str(args.checkpoint.resolve()),'--port',str(port),'--seed',str(seed),'--listen-fd',str(reserved_listener.fileno()),'--ready-token',token],cwd=ROOT,
                            env=dict(env,CUDA_VISIBLE_DEVICES=args.server_gpu),stdout=policy_log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(reserved_listener.fileno(),))
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
                    command=['/home/houshengyuan/env_isaacsim60/bin/python',str(ROOT/'lerobot_policy/run_delivery_episode.py'),
                        '--list',str(args.list.resolve()),'--index',str(idx),'--mode',args.mode,'--seed',str(seed),
                        '--output',str((trial/'simulation').resolve()),'--port',str(port)]
                    if args.mode=='policy':command+=['--contract',str((args.checkpoint/'run_config.json').resolve()),'--server-token',token]
                    sim=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    if sim.wait(timeout=600)!=0:raise RuntimeError('Simulation exited with an error')
                    if (trial/'simulation/error.json').exists() or not all((trial/'simulation'/name).is_file() for name in ['completed.json','result.json','rollout.npz','rollout.gif']):
                        raise RuntimeError('Incomplete simulation artifacts')
                    result=json.loads((trial/'simulation/result.json').read_text())
                    row.update({k:result[k] for k in ['success','failure_reasons','final_xy_error_m','final_z_error_m',
                        'ik_failures','action_clipping_count','quaternion_fallback_count','inference_calls','lifted','max_observation_position_difference_m']})
                    row['result']=str((trial/'simulation/result.json').resolve())
            except Exception as exc:row.update(success=False,error=repr(exc))
            finally:
                stop(sim);stop(server)
                if reserved_listener is not None:reserved_listener.close()
            (trial/'trial.json').write_text(json.dumps(row,indent=2)+'\n');out.append(row)
            print(json.dumps(row),flush=True)
        return out
    queues=[[] for _ in args.sim_gpus]
    for j,t in enumerate(tasks):queues[j%len(queues)].append((j,t))
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures=[pool.submit(run_queue,g,q) for g,q in zip(args.sim_gpus,queues)]
        for future in as_completed(futures):completed.extend(future.result())
    completed.sort(key=lambda r:(r['index'],r['seed']))
    groups={}
    for idx in indices:
        rows=[r for r in completed if r['index']==idx]
        groups[str(idx)]=dict(uid=entries[idx]['uid'],count=len(rows),successes=sum(r['success'] for r in rows))
    summary=dict(mode=args.mode,checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        requested=len(tasks),completed=len(completed),successes=sum(r['success'] for r in completed),
        runtime_errors=sum('error' in r for r in completed),groups=groups,
        failure_categories=dict(Counter(x for r in completed for x in r.get('failure_reasons',[]))),trials=completed)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)

if __name__=='__main__':main()
