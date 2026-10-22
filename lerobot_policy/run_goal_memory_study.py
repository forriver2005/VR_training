"""Existing RGB-D measured-goal conditioning; no new data or scripted expert actions."""
from repro_paths import learning_python, isaac_python
import argparse,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from delivery_policy_contract import ROOT,sha256
from rgbd_object_features import GOAL_RECIPE
PYTHON=learning_python()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    manifest=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    plan=dict(recipe=GOAL_RECIPE,source='only existing fixed RGB-D and calibration',known_task_priors='green target, 6cm cube, initially static target',
        policy='PointNet Diffusion, H2/K16/A8, current TCP origin, relative XYZ output; all actions produced by network',
        experiments=['goal_wrist_tcp','goal_fused_tcp'],steps=40000,batch_size=32,seed=42,
        comparison='goal_fused_tcp vs fused_tcp_pointnet isolates added goal memory; wrist vs fused goal experiments isolate point-cloud source',
        train_sha256=sha256(manifest/'train_4hz.jsonl'),validation_sha256=sha256(manifest/'validation_quick_4hz.jsonl'),
        goal_code_sha256=sha256(ROOT/'lerobot_policy/rgbd_object_features.py'),test_used=False)
    (out/'goal_experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def run(label,script,argv,gpu=None):
        print('START',label,flush=True);env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (out/f'{label}.log').open('w') as log:subprocess.run([PYTHON,str(ROOT/'lerobot_policy'/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',label,flush=True)
    def experiment(name,source,gpu):
        train=out/(name+'_train40k');val=out/(name+'_validation25')
        run(name+'_training','train_delivery_diffusion.py',['--list',manifest/'train_4hz.jsonl','--output',train,'--steps',40000,'--save-every',10000,'--batch-size',32,
            '--prediction-type','sample','--timer-mode','bounded','--pointcloud-source',source,'--relative-actions','--observation-frame','current_tcp','--goal-memory'],gpu)
        run(name+'_validation','monitor_delivery_validation.py',['--training',train,'--list',manifest/'validation_quick_4hz.jsonl','--output',val,'--gpu',gpu])
        checkpoint=Path((val/'selected_checkpoint.txt').read_text().strip())
        run(name+'_offline_train20','evaluate_delivery_diffusion.py',['--checkpoint',checkpoint,'--list',manifest/'pilot20_4hz.jsonl','--output',out/(name+'_offline_train20'),'--seeds',0],gpu)
        return name,checkpoint
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(experiment,'goal_wrist_tcp','wrist_depth',1),pool.submit(experiment,'goal_fused_tcp','fused_depth',2)]
        models=dict(f.result() for f in fs)
    deadline=time.monotonic()+7200
    while not (out/'models_complete.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('Base closed-loop study did not complete')
        time.sleep(5)
    for name,checkpoint in models.items():
        run(name+'_closed','run_delivery_sweep.py',['--list',manifest/'pilot_scene_checks_4hz.jsonl','--tasks',out/'closed_tasks.json','--mode','policy','--checkpoint',checkpoint,
            '--output',out/(name+'_closed'),'--sim-gpus',0,0,1,1,'--server-gpu',2,'--port-base',8000])
    (out/'goal_models_complete.json').write_text(json.dumps({k:str(v) for k,v in models.items()},indent=2)+'\n')

if __name__=='__main__':main()
