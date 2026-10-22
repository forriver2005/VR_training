"""Finite, logged collection-to-evaluation study. Source deliveries remain immutable."""
from repro_paths import learning_python, isaac_python
import argparse,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from delivery_policy_contract import ROOT,load_entries,load_episode,BOUNDED_TIMER
PYTHON=learning_python()


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);args=p.parse_args();out=args.collection.resolve()
    original=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    def run(name,script,argv,gpu=None):
        print('START',name,flush=True)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (out/f'{name}.log').open('w') as log:
            subprocess.run([PYTHON,str(ROOT/'lerobot_policy'/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,flush=True)
    deadline=time.monotonic()+1800
    while not (out/'collected_candidates.jsonl').is_file():
        if time.monotonic()>deadline:raise TimeoutError('Collection did not complete')
        time.sleep(5)
    entries=load_entries(out/'collected_candidates.jsonl')
    run('independent_replay','run_delivery_sweep.py',['--list',out/'collected_candidates.jsonl','--indices',*range(len(entries)),
        '--mode','expert_ik','--seeds',0,'--sim-gpus',0,0,'--output',out/'independent_replay'])
    run('admission','finalize_layout_pilot.py',['--collection',out,'--replay',out/'independent_replay'])
    run('paired_audit','audit_layout_pairs.py',['--collection',out])
    train_list=out/'manifests/augmented_train.jsonl';entries=load_entries(train_list)
    cache=[]
    # Warm isolated caches before starting GPU timing; no test or validation labels fitted.
    for source in ['wrist_depth','fused_depth']:
        for e in entries:
            d=load_episode(e,BOUNDED_TIMER,source);cache.append(dict(uid=e['uid'],source=source,path=d['derived_point_cloud_path']))
    (out/'cache_manifest.json').write_text(json.dumps(cache,indent=2)+'\n')
    def train_validate(source,gpu):
        label='wrist' if source=='wrist_depth' else 'fused'
        run(label+'_train','train_delivery_diffusion.py',['--list',train_list,'--output',out/(label+'_train40k'),
            '--steps',40000,'--save-every',10000,'--batch-size',16,'--seed',42,'--prediction-type','sample','--timer-mode','bounded','--pointcloud-source',source],gpu)
        run(label+'_validation','monitor_delivery_validation.py',['--training',out/(label+'_train40k'),'--list',original/'validation_quick_4hz.jsonl',
            '--output',out/(label+'_validation25'),'--gpu',gpu])
        checkpoint=Path((out/(label+'_validation25')/'selected_checkpoint.txt').read_text().strip())
        run(label+'_offline_train','evaluate_delivery_diffusion.py',['--checkpoint',checkpoint,'--list',train_list,'--output',out/(label+'_offline_train'),'--seeds',0],gpu)
        return label,checkpoint
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_validate,s,g) for s,g in [('wrist_depth',1),('fused_depth',2)]]
        models=dict(f.result() for f in futures)
    baseline=ROOT/'lerobot_policy/outputs/multiscene_pilot20_20261021/train_sample40k/checkpoints/030000/ema'
    for label,checkpoint in [('baseline',baseline),*models.items()]:
        tasks=out/'manifests'/('baseline_new_tasks.json' if label=='baseline' else 'closed_tasks.json')
        run(label+'_closed','run_delivery_sweep.py',['--list',out/'manifests/closed_checks.jsonl','--tasks',tasks,'--mode','policy','--checkpoint',checkpoint,
            '--output',out/(label+'_closed'),'--sim-gpus',0,0,1,1,'--server-gpu',2,'--port-base',7200])
    (out/'study_complete.json').write_text(json.dumps(dict(models={k:str(v) for k,v in models.items()},test_used=False),indent=2)+'\n')

if __name__=='__main__':main()
