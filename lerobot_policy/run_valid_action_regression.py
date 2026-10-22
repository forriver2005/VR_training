"""Compare valid-action projection and ensemble on known regression layouts only."""
from repro_paths import learning_python, isaac_python
import json,os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from delivery_policy_contract import ROOT,sha256
PYTHON=learning_python()
root=ROOT/'lerobot_policy/outputs/existing398_20261021';manifest=ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'
tasks=json.loads((manifest/'regression_tasks.json').read_text())
remaining=[t for t in tasks if (t['index'],t['seed'])!=(5,1)]
task_file=root/'valid_actions_remaining_tasks.json';task_file.write_text(json.dumps(remaining,indent=2)+'\n')
plan=dict(candidates=[1,4],action_samples='mean latent trajectories for N samples',A=8,binary_gripper=True,fixed_down_quaternion=True,anchor_snap_epsilon=.05,
    selection='maximize worst-layout success rate, then total successes, then lower worst final XY; tie favors fewer samples',
    role='known-layout regression; original test was consumed by debugging; fresh reserve layouts not used',
    probe_reuse='reuse valid index5seed1 for each variant; retain all other outcomes')
(root/'valid_actions_regression_plan.json').write_text(json.dumps(plan,indent=2)+'\n')

def run(n):
    cp=root/f'latched_valid_actions_n{n}';target=root/f'valid_actions_n{n}_remaining'
    print('START',n,flush=True)
    with (root/f'valid_actions_n{n}_remaining.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'lerobot_policy/run_delivery_sweep.py'),'--list',str(manifest/'regression_scenes.jsonl'),'--tasks',str(task_file),'--mode','policy','--checkpoint',str(cp),'--output',str(target),
            '--sim-gpus','0','0','1','1','--server-gpu','2','--port-base',str(12000 if n==1 else 12100)],cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    s=json.loads((target/'summary.json').read_text());probe=json.loads((root/f'valid_actions_probe_n{n}/summary.json').read_text());trials=s['trials']+probe['trials'];trials.sort(key=lambda t:(t['index'],t['seed']))
    if len(trials)!=50 or len({(t['index'],t['seed']) for t in trials})!=50:raise RuntimeError('Regression trial coverage mismatch')
    groups={str(i):dict(count=sum(t['index']==i for t in trials),successes=sum(t['index']==i and t['success'] for t in trials)) for i in range(6)}
    final=dict(checkpoint=str(cp),checkpoint_contract_sha256=sha256(cp/'run_config.json'),action_samples=n,completed=50,requested=50,successes=sum(t['success'] for t in trials),runtime_errors=sum('error' in t for t in trials),groups=groups,
        failure_categories=dict(Counter(c for t in trials for c in t.get('failure_reasons',[]))),trials=trials,role='regression, not fresh independent test')
    folder=root/f'valid_actions_n{n}_regression';folder.mkdir();(folder/'summary.json').write_text(json.dumps(final,indent=2)+'\n')
    if final['runtime_errors']:raise RuntimeError('Runtime failures must be resolved before choosing a policy')
    print('DONE',n,groups,flush=True);return final

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,n) for n in [1,4]];results=[f.result() for f in futures]
def score(r):
    return (min(v['successes']/v['count'] for v in r['groups'].values()),r['successes'],-max(t.get('final_xy_error_m',10.) for t in r['trials']),-r['action_samples'])
best=max(results,key=score)
(root/'valid_actions_selected.json').write_text(json.dumps(dict(plan=plan,selected=best,comparison=[{k:v for k,v in r.items() if k!='trials'} for r in results]),indent=2)+'\n')
