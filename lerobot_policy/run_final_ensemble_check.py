"""Recheck the ensemble with the completed controller; all cases are now regression."""
from repro_paths import learning_python, isaac_python
import hashlib,json,os,subprocess
from pathlib import Path
from delivery_policy_contract import ROOT
OUT=ROOT/'lerobot_policy/outputs/existing398_20261021';M=ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'
PYTHON=learning_python()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    base=[json.loads(x) for x in (M/'all_known_scenes.jsonl').read_text().splitlines()]
    local=[json.loads(x) for x in (M/'local_combinations.jsonl').read_text().splitlines()]
    rows=[dict(e,evaluation_only=True,evaluation_role='regression_after_all_feedback') for e in base+local]
    scene_file=M/'complete_regression_scenes.jsonl';scene_file.write_text(''.join(json.dumps(e)+'\n' for e in rows))
    tasks=json.loads((M/'all_known_tasks.json').read_text())+[dict(index=i+8,seed=s) for i in range(16) for s in [0,1,2]]
    remaining=[t for t in tasks if (t['index'],t['seed']) not in {(22,2),(23,2)}]
    task_file=M/'complete_ensemble_remaining_tasks.json';task_file.write_text(json.dumps(remaining,indent=2)+'\n')
    cp=OUT/'height_aware_policy_n4'
    plan=dict(checkpoint=str(cp),checkpoint_contract_sha256=sha(cp/'run_config.json'),total=138,reused_probes=2,remaining=len(remaining),
        role='All original and combination cases now are regression; no new independent-test claim',
        purpose='Test whether sampling averaging improves initial/terminal precision after the execution fixes; retain every valid failure')
    (OUT/'ensemble_final_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (OUT/'ensemble_full_remaining.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'lerobot_policy/run_delivery_sweep.py'),'--list',str(scene_file),'--tasks',str(task_file),'--mode','policy','--checkpoint',str(cp),
            '--output',str(OUT/'ensemble_full_remaining'),'--sim-gpus','0','0','0','1','1','1','--server-gpu','2','--port-base','14000'],cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    fresh=json.loads((OUT/'ensemble_full_remaining/summary.json').read_text());probe=json.loads((OUT/'ensemble_polish_probe/summary.json').read_text())
    reused=[dict(t,index=t['index']+8,original_probe_index=t['index']) for t in probe['trials']]
    trials=sorted(fresh['trials']+reused,key=lambda t:(t['index'],t['seed']))
    if len(trials)!=138 or len({(t['index'],t['seed']) for t in trials})!=138:raise ValueError('Trial coverage mismatch')
    result=dict(checkpoint=str(cp),completed=138,requested=138,successes=sum(t['success'] for t in trials),runtime_errors=sum('error' in t for t in trials),
        original_layout_successes=sum(t['index']<8 and t['success'] for t in trials),original_layout_count=90,
        combination_successes=sum(t['index']>=8 and t['success'] for t in trials),combination_count=48,
        groups={str(i):dict(count=sum(t['index']==i for t in trials),successes=sum(t['index']==i and t['success'] for t in trials)) for i in range(24)},
        trials=trials,role=plan['role'])
    target=OUT/'complete_ensemble_regression';target.mkdir(exist_ok=False);(target/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['trials','groups']}),flush=True)

if __name__=='__main__':main()
