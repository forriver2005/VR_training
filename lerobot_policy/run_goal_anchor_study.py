"""Learned target-anchored residual diffusion; evaluate with fixed existing splits."""
from repro_paths import learning_python, isaac_python
import json,os,subprocess,time
from pathlib import Path
from delivery_policy_contract import ROOT,sha256
PYTHON=learning_python()
out=ROOT/'lerobot_policy/outputs/existing398_20261021'
manifest=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
name='goal_anchor_proprio'
plan=dict(name=name,training_episodes=398,encoder='no point-cloud features; measured goal and proprioception only',
    action='9 learned outputs: residual xyz, xyzw, finger, anchor weight; decoded to8 physical action channels',
    label='extra anchor supervision from archived expert phase; phase/frame index never supplied at inference',
    comparison='bundled representation/conditioning change; does not isolate causal benefit of each change',
    steps=40000,batch_size=32,seed=42,H=2,K=16,A=8,test_used=False,
    goal_code_sha256=sha256(ROOT/'lerobot_policy/rgbd_object_features.py'),anchor_code_sha256=sha256(ROOT/'lerobot_policy/goal_anchor_action.py'))
(out/(name+'_plan.json')).write_text(json.dumps(plan,indent=2)+'\n')

def run(label,script,argv,gpu=None):
    print('START',label,flush=True);env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (out/f'{label}.log').open('w') as log:subprocess.run([PYTHON,str(ROOT/'lerobot_policy'/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

train=out/(name+'_train40k');val=out/(name+'_validation25')
run(name+'_training','train_delivery_diffusion.py',['--list',manifest/'train_4hz.jsonl','--output',train,'--steps',40000,'--save-every',10000,'--batch-size',32,
    '--prediction-type','sample','--timer-mode','bounded','--observation-frame','current_tcp','--encoder','proprioceptive','--goal-memory','--goal-anchor-actions'],0)
run(name+'_validation','monitor_delivery_validation.py',['--training',train,'--list',manifest/'validation_quick_4hz.jsonl','--output',val,'--gpu',0])
checkpoint=Path((val/'selected_checkpoint.txt').read_text().strip())
run(name+'_offline_train20','evaluate_delivery_diffusion.py',['--checkpoint',checkpoint,'--list',manifest/'pilot20_4hz.jsonl','--output',out/(name+'_offline_train20'),'--seeds',0],0)
deadline=time.monotonic()+7200
while not (out/'goal_models_complete.json').is_file():
    if time.monotonic()>deadline:raise TimeoutError('Earlier goal-memory sweeps incomplete')
    time.sleep(5)
run(name+'_closed','run_delivery_sweep.py',['--list',manifest/'pilot_scene_checks_4hz.jsonl','--tasks',out/'closed_tasks.json','--mode','policy','--checkpoint',checkpoint,
    '--output',out/(name+'_closed'),'--sim-gpus',0,0,1,1,'--server-gpu',2,'--port-base',8400])
(out/(name+'_complete.json')).write_text(json.dumps(dict(checkpoint=str(checkpoint)),indent=2)+'\n')
