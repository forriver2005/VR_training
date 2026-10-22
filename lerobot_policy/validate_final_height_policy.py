"""Freeze after known-layout regression, then test quarantined local combinations."""
from repro_paths import learning_python, isaac_python
import datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from delivery_policy_contract import ROOT
OUT=ROOT/'lerobot_policy/outputs/existing398_20261021';REPORT=ROOT/'reports/existing398_quality_20261021'
MANIFEST=ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'
PYTHON=learning_python()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    needed=[OUT/'height_guard_remaining/summary.json',OUT/'local_combination_expert_remaining/summary.json'];deadline=time.monotonic()+3600
    while not all(p.is_file() for p in needed):
        if time.monotonic()>deadline:raise TimeoutError('Prerequisite evaluations incomplete')
        time.sleep(5)
    remaining=json.loads(needed[0].read_text());probe=json.loads((OUT/'height_guard_probe/summary.json').read_text())
    reused=[dict(t,index=6,original_probe_index=t['index']) for t in probe['trials']]
    rows=sorted(remaining['trials']+reused,key=lambda r:(r['index'],r['seed']))
    if len(rows)!=90 or len({(r['index'],r['seed']) for r in rows})!=90:raise ValueError('Regression coverage mismatch')
    if not all(r['success'] and 'error' not in r for r in rows):raise RuntimeError('Known-layout regression gate did not pass; do not consume more evaluation cases')
    expert_new=json.loads(needed[1].read_text())
    if expert_new['successes']!=expert_new['requested'] or expert_new['runtime_errors']:raise RuntimeError('Local combination feasibility gate failed')
    old=ROOT/'lerobot_policy/outputs/layout_pilot_20261021/independent_replay'
    expert_old=[json.loads(p.read_text()) for p in old.glob('*/trial.json')]
    expert_rows=[r for r in expert_old if r['success'] and 'error' not in r]+expert_new['trials']
    if len(expert_rows)!=16 or len({r['index'] for r in expert_rows})!=16:raise ValueError('Local feasibility coverage mismatch')
    cp=OUT/'height_aware_policy';folder=OUT/'height_aware_regression';folder.mkdir(exist_ok=False)
    groups={str(i):dict(count=sum(r['index']==i for r in rows),successes=sum(r['index']==i and r['success'] for r in rows)) for i in range(8)}
    summary=dict(checkpoint=str(cp),requested=90,completed=90,successes=90,runtime_errors=0,groups=groups,trials=rows,
        role='all eight previously evaluated provided layouts; regression only')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    snapshot=REPORT/'frozen_height_aware_source';snapshot.mkdir(exist_ok=False)
    names=list(json.loads((OUT/'frozen_final_candidate.json').read_text())['source_sha256'])+['transport_clearance_guard.py','derive_transport_guard_checkpoint.py']
    for name in dict.fromkeys(names):shutil.copy2(ROOT/'lerobot_policy'/name,snapshot/name)
    frozen=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(cp),
        checkpoint_sha256={p.name:sha(p) for p in cp.iterdir() if p.is_file()},source_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        known_regression=dict(successes=90,count=90,groups=groups),
        composition_evaluation=dict(manifest=str(MANIFEST/'local_combinations.jsonl'),manifest_sha256=sha(MANIFEST/'local_combinations.jsonl'),
            tasks_sha256=sha(MANIFEST/'local_combination_policy_tasks.json'),layouts=16,seeds=[0,1,2],training_use=False,prior_policy_performance_selection_use=False,
            limits='same four backgrounds; input/action observability was audited earlier; not a fully untouched new-background benchmark'),
        raw_training_episodes=398,new_training_demonstrations_added=0)
    (OUT/'frozen_height_aware_candidate.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('FROZEN:90/90 known-layout regression;16local combinations next',flush=True)
    with (OUT/'local_combination_policy.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'lerobot_policy/run_delivery_sweep.py'),'--list',str(MANIFEST/'local_combinations.jsonl'),'--tasks',str(MANIFEST/'local_combination_policy_tasks.json'),
            '--mode','policy','--checkpoint',str(cp),'--output',str(OUT/'local_combination_policy'),'--sim-gpus','0','0','1','1','--server-gpu','2','--port-base','13600'],cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((OUT/'local_combination_policy/summary.json').read_text())
    for name,h in frozen['checkpoint_sha256'].items():
        if sha(cp/name)!=h:raise RuntimeError('Frozen checkpoint changed')
    for name,h in frozen['source_sha256'].items():
        if sha(ROOT/'lerobot_policy'/name)!=h:raise RuntimeError('Frozen source changed')
    (OUT/'height_policy_validation_complete.json').write_text(json.dumps(dict(checkpoint=str(cp),known_successes=90,known_count=90,
        combination_successes=result['successes'],combination_count=result['completed'],runtime_errors=result['runtime_errors'],frozen_verified=True),indent=2)+'\n')
    print('COMBINATIONS',result['successes'],'/',result['completed'],'runtime',result['runtime_errors'],flush=True)

if __name__=='__main__':main()
