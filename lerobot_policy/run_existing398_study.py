"""Full existing-data comparison; no new scenes or synthetic observations."""
from repro_paths import learning_python, isaac_python
import argparse,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from delivery_policy_contract import ROOT,sha256
PYTHON=learning_python()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    manifests=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    experiments=[dict(name='wrist_absolute',gpu=1,source='wrist_depth',encoder='pointnet',frame='base',relative=False),
                 dict(name='fused_tcp_pointnet',gpu=2,source='fused_depth',encoder='pointnet',frame='current_tcp',relative=True),
                 dict(name='fused_tcp_spatial',gpu=0,source='fused_depth',encoder='spatial_tokens',frame='current_tcp',relative=True)]
    plan=dict(experiments=experiments,training_episodes=398,training_backgrounds=4,steps=40000,batch_size=32,seed=42,
        H=2,K=16,A=8,ddpm_steps=100,prediction_type='sample',timer='bounded',
        selection='each model: minimum original validation25 offline position error among10k/20k/30k/40k; then compare closed loop on original4train layouts x5 and original validation layout x10',
        scope='use only original delivered train_4hz; local generated layout pilot excluded; no test use',
        interpretation='wrist->fused TCP changes source and coordinate representation jointly; fused TCP pointnet->spatial isolates encoder',
        references=['https://3d-diffusion-policy.github.io/','https://rise-policy.github.io/'],
        method_note='Local spatial token experiment inspired by spatial 3D encoding; not a RISE/iDP3 reproduction or SOTA claim',
        manifest_hashes={n:sha256(manifests/n) for n in ['train_4hz.jsonl','validation_quick_4hz.jsonl','test_4hz.jsonl','pilot_scene_checks_4hz.jsonl']})
    (out/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def run(label,script,argv,gpu=None):
        print('START',label,flush=True);env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (out/f'{label}.log').open('w') as log:
            subprocess.run([PYTHON,str(ROOT/'lerobot_policy'/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',label,flush=True)
    deadline=time.monotonic()+3600
    while not (out/'cache_manifest.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('Preprocessing did not finish')
        time.sleep(5)
    def experiment(e):
        name=e['name'];train=out/(name+'_train40k');val=out/(name+'_validation25')
        argv=['--list',manifests/'train_4hz.jsonl','--output',train,'--steps',40000,'--save-every',10000,'--batch-size',32,'--seed',42,
            '--prediction-type','sample','--timer-mode','bounded','--pointcloud-source',e['source'],'--encoder',e['encoder'],'--observation-frame',e['frame']]
        if e['relative']:argv+=['--relative-actions']
        run(name+'_training','train_delivery_diffusion.py',argv,e['gpu'])
        run(name+'_validation','monitor_delivery_validation.py',['--training',train,'--list',manifests/'validation_quick_4hz.jsonl','--output',val,'--gpu',e['gpu']])
        checkpoint=Path((val/'selected_checkpoint.txt').read_text().strip())
        run(name+'_offline_train20','evaluate_delivery_diffusion.py',['--checkpoint',checkpoint,'--list',manifests/'pilot20_4hz.jsonl','--output',out/(name+'_offline_train20'),'--seeds',0],e['gpu'])
        return name,checkpoint
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(experiment,e) for e in experiments]
        models=dict(f.result() for f in futures)
    tasks=[dict(index=i,seed=s) for i in range(4) for s in range(5)]+[dict(index=4,seed=s) for s in range(10)]
    task_file=out/'closed_tasks.json';task_file.write_text(json.dumps(tasks,indent=2)+'\n')
    for name,checkpoint in models.items():
        run(name+'_closed','run_delivery_sweep.py',['--list',manifests/'pilot_scene_checks_4hz.jsonl','--tasks',task_file,'--mode','policy','--checkpoint',checkpoint,
            '--output',out/(name+'_closed'),'--sim-gpus',0,0,1,1,'--server-gpu',2,'--port-base',7600])
    (out/'models_complete.json').write_text(json.dumps({k:str(v) for k,v in models.items()},indent=2)+'\n')

if __name__=='__main__':main()
