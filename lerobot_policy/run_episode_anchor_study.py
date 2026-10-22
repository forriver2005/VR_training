"""Fixed measured references + learned goal gate, including closed-loop selection."""
from repro_paths import learning_python, isaac_python
import argparse,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from delivery_policy_contract import ROOT,sha256
parser=argparse.ArgumentParser();parser.add_argument('--canonical-3d',action='store_true');args=parser.parse_args()
prefix='canonical' if args.canonical_3d else 'episode'
names=[prefix+'_gate_proprio',prefix+'_gate_wrist']
action_flags=['--episode-goal-3d-actions','--goal-state-switch'] if args.canonical_3d else ['--episode-goal-actions']
selection_port=10000 if args.canonical_3d else 9000
PYTHON=learning_python()
out=ROOT/'lerobot_policy/outputs/existing398_20261021';manifest=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
plan=dict(experiments=names,training_episodes=398,steps=40000,batch_size=32,seed=42,
    reference='measured actual state[0], not reset_target; measured initial green target RGB-D',
    action='episode-fixed XY initial/goal anchors, Z relative initial measured TCP, learned alpha; 9 latent->8 physical',
    gate='MLP of reset-relative robot history, gripper, validity and action-history timer predicts transport-started; no runtime frame/stage input; masks both point and goal features',
    training_supervision='alpha clip((expert_frame-19)/7,0,1), phase expert_frame>=20; labels only',
    position_loss_weight=6.,phase_loss_weight=.05,
    selection='each10k checkpoint A8 validation seeds0,1,2; choose successes/lifted/finalXY; then test A4 and A2 at chosen checkpoint on same3seeds; choose same ranking',
    final_evaluation='chosen3validation trials reused, remaining seeds3..9 plus4training layouts x5 seeds; all selection failures retained',
    test_used=False,raw_new_data_used=False,
    code_sha256={n:sha256(ROOT/'lerobot_policy'/n) for n in ['pointcloud_diffusion.py','episode_goal_anchor.py','rgbd_object_features.py','train_delivery_diffusion.py','diffusion_overfit_common.py']})
if args.canonical_3d:
    plan['action']='episode fixed3D anchors: XY alpha; Z beta=clip((7alpha-4)/3); residual; learned9->8'
    plan['state']='learned gate switches robot origin from measured initial TCP to measured goal; classifier remains initial-relative'
    plan['goal_embedding']='relativeXYZ and valid only; observed extents/confidence excluded from learned features to avoid scene-identification shortcuts'
    plan['code_sha256']['episode_goal_3d.py']=sha256(ROOT/'lerobot_policy/episode_goal_3d.py')
(out/(prefix+'_anchor_plan.json')).write_text(json.dumps(plan,indent=2)+'\n')

def run(label,script,argv,gpu=None):
    print('START',label,flush=True);env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (out/f'{label}.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'lerobot_policy'/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def train(name,encoder,gpu):
    target=out/(name+'_train40k')
    run(name+'_training','train_delivery_diffusion.py',['--list',manifest/'train_4hz.jsonl','--output',target,'--steps',40000,'--save-every',10000,'--batch-size',32,
        '--prediction-type','sample','--timer-mode','bounded','--observation-frame','initial_tcp','--encoder',encoder,'--goal-memory',*action_flags,'--learned-goal-gate','--position-loss-weight',6],gpu)
    run(name+'_validation','monitor_delivery_validation.py',['--training',target,'--list',manifest/'validation_quick_4hz.jsonl','--output',out/(name+'_validation25'),'--gpu',gpu])
    return name,target

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(train,names[0],'proprioceptive',1),pool.submit(train,names[1],'pointnet',2)]
    models=dict(f.result() for f in jobs)

deadline=time.monotonic()+7200
while not (out/'goal_anchor_proprio_complete.json').is_file():
    if time.monotonic()>deadline:raise TimeoutError('Earlier comparison sweeps incomplete')
    time.sleep(5)

def score(row):
    trials=row['summary']['trials']
    return (sum(t['success'] for t in trials),sum(t.get('lifted',False) for t in trials),
            -sum(t.get('final_xy_error_m',10.) for t in trials)/len(trials),row['A'])

def select(name,training,port):
    choices=[]
    def trial(checkpoint,step,a):
        label=f'{name}_selection_s{step}_a{a}';target=out/label
        run(label,'run_delivery_sweep.py',['--list',manifest/'pilot_scene_checks_4hz.jsonl','--indices',4,'--seeds',0,1,2,'--mode','policy','--checkpoint',checkpoint,
            '--output',target,'--sim-gpus',0,1,'--server-gpu',2,'--port-base',port])
        row=dict(step=step,A=a,checkpoint=str(checkpoint),summary=json.loads((target/'summary.json').read_text()))
        if row['summary']['runtime_errors']:raise RuntimeError('Selection encountered runtime errors; cannot score as policy failure')
        choices.append(row);return row
    for step in [10000,20000,30000,40000]:trial(training/'checkpoints'/f'{step:06d}'/'ema',step,8)
    selected=max(choices,key=score);parent=Path(selected['checkpoint']);step=selected['step']
    for a in [4,2]:
        checkpoint=out/f'{name}_s{step}_a{a}'
        run(f'{name}_derive_a{a}','derive_action_steps_checkpoint.py',['--checkpoint',parent,'--output',checkpoint,'--action-steps',a])
        trial(checkpoint,step,a)
    selected=max(choices,key=score)
    (out/(name+'_closed_selection.json')).write_text(json.dumps(dict(rule=plan['selection'],choices=choices,selected=selected),indent=2)+'\n')
    return selected

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs={n:pool.submit(select,n,p,selection_port+i*100) for i,(n,p) in enumerate(models.items())}
    selected={n:f.result() for n,f in jobs.items()}

for name,choice in selected.items():
    checkpoint=Path(choice['checkpoint'])
    tasks=[dict(index=4,seed=s) for s in range(3,10)]+[dict(index=i,seed=s) for i in range(4) for s in range(5)]
    task_file=out/(name+'_remaining_tasks.json');task_file.write_text(json.dumps(tasks,indent=2)+'\n')
    target=out/(name+'_remaining_closed')
    run(name+'_remaining_closed','run_delivery_sweep.py',['--list',manifest/'pilot_scene_checks_4hz.jsonl','--tasks',task_file,'--mode','policy','--checkpoint',checkpoint,
        '--output',target,'--sim-gpus',0,0,1,1,'--server-gpu',2,'--port-base',selection_port+300])
    remainder=json.loads((target/'summary.json').read_text());trials=choice['summary']['trials']+remainder['trials'];trials.sort(key=lambda t:(t['index'],t['seed']))
    if len(trials)!=30 or len({(t['index'],t['seed']) for t in trials})!=30:raise RuntimeError('Final trial coverage mismatch')
    groups={str(i):dict(count=sum(t['index']==i for t in trials),successes=sum(t['index']==i and t['success'] for t in trials)) for i in range(5)}
    summary=dict(checkpoint=str(checkpoint),A=choice['A'],selected_step=choice['step'],requested=30,completed=30,successes=sum(t['success'] for t in trials),runtime_errors=sum('error' in t for t in trials),groups=groups,
        failure_categories=dict(Counter(c for t in trials for c in t.get('failure_reasons',[]))),trials=trials,validation_used_for_selection=True)
    final=out/(name+'_closed');final.mkdir();(final/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    run(name+'_offline_train20','evaluate_delivery_diffusion.py',['--checkpoint',checkpoint,'--list',manifest/'pilot20_4hz.jsonl','--output',out/(name+'_offline_train20'),'--seeds',0],2)
(out/(prefix+'_anchor_complete.json')).write_text(json.dumps({n:dict(checkpoint=c['checkpoint'],A=c['A'],step=c['step']) for n,c in selected.items()},indent=2)+'\n')
