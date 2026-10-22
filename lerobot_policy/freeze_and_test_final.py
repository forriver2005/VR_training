"""Freeze the selected known-layout candidate before fresh layout evaluation."""
from repro_paths import learning_python, isaac_python
import datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from delivery_policy_contract import ROOT
OUT=ROOT/'lerobot_policy/outputs/existing398_20261021';REPORT=ROOT/'reports/existing398_quality_20261021'
MANIFEST=ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'
PYTHON=learning_python()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    deadline=time.monotonic()+1800
    remaining=OUT/'smooth_release_remaining/summary.json'
    while not remaining.exists():
        if time.monotonic()>deadline:raise TimeoutError('Filtered regression did not complete')
        time.sleep(5)
    a=json.loads((OUT/'smooth_release_probe/summary.json').read_text());b=json.loads(remaining.read_text())
    rows=sorted(a['trials']+b['trials'],key=lambda r:(r['index'],r['seed']))
    if len(rows)!=50 or len({(r['index'],r['seed']) for r in rows})!=50:raise ValueError('Regression coverage mismatch')
    if not all(r['success'] and 'error' not in r and r['ik_failures']==0 and r['action_clipping_count']==0 for r in rows):raise RuntimeError('Do not promote a candidate that failed the regression gate')
    groups={str(i):dict(count=sum(r['index']==i for r in rows),successes=sum(r['index']==i and r['success'] for r in rows)) for i in range(6)}
    summary=dict(checkpoint=a['checkpoint'],completed=50,successes=50,runtime_errors=0,groups=groups,trials=rows,role='known-layout regression, not independent testing')
    folder=OUT/'smooth_release_regression';folder.mkdir(exist_ok=False);(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    previous=json.loads((OUT/'settled_release_regression/summary.json').read_text())
    mean=lambda rs:sum(r['final_xy_error_m'] for r in rs)/len(rs)
    selected=summary if mean(rows)<mean(previous['trials']) else previous
    cp=Path(selected['checkpoint']);snapshot=REPORT/'frozen_final_source';snapshot.mkdir(exist_ok=False)
    names=['delivery_policy_contract.py','diffusion_overfit_common.py','pointcloud_diffusion.py','spatial_pointcloud.py','relative_action.py','episode_goal_3d.py',
        'rgbd_object_features.py','rgbd_goal_geometry_v2.py','goal_observation_registry.py','release_guard.py','smooth_joint_targets.py',
        'diffusion_policy_server.py','run_delivery_episode.py','run_delivery_sweep.py','run_delivery_trials.py','socket_protocol.py',
        'train_delivery_diffusion.py','evaluate_delivery_diffusion.py','monitor_delivery_validation.py',
        'derive_goal_recipe_checkpoint.py','derive_monotonic_gate_checkpoint.py','derive_action_steps_checkpoint.py','derive_release_guard_checkpoint.py','derive_smooth_controller_checkpoint.py']
    for name in names:shutil.copy2(ROOT/'lerobot_policy'/name,snapshot/name)
    frozen=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(cp),
        checkpoint_sha256={p.name:sha(p) for p in cp.iterdir() if p.is_file()},source_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        regression=dict(successes=50,count=50,groups=groups,unfiltered_mean_xy_m=mean(previous['trials']),filtered_mean_xy_m=mean(rows)),
        first_test_history=dict(result='2/20 before debugging',path=str(OUT/'final_test/summary.json'),subsequent_role='known regression layout'),
        fresh_test=dict(manifest=str(MANIFEST/'fresh_scenes.jsonl'),manifest_sha256=sha(MANIFEST/'fresh_scenes.jsonl'),layouts=2,seeds=list(range(20)),
            policy_control_dt_s=.25,actions=40,source_control_dt_s=.125,scene_assets_and_reset_only=True,used_for_selection=False),
        quality_limits=dict(xy_m=.005,height_difference_m=.05,height_error_m=.004,drift_m=.001,speed_m_s=.05))
    (OUT/'frozen_final_candidate.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('FROZEN',cp,'regression50/50; fresh layouts not used for selection',flush=True)
    with (OUT/'fresh_holdout_policy.log').open('w') as log:
        subprocess.run([PYTHON,str(ROOT/'lerobot_policy/run_delivery_sweep.py'),'--list',str(MANIFEST/'fresh_scenes.jsonl'),'--indices','0','1','--seeds',*map(str,range(20)),
            '--mode','policy','--checkpoint',str(cp),'--output',str(OUT/'fresh_holdout_policy'),'--sim-gpus','0','0','1','1','--server-gpu','2','--port-base','13000'],
            cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((OUT/'fresh_holdout_policy/summary.json').read_text())
    for name,h in frozen['checkpoint_sha256'].items():
        if sha(cp/name)!=h:raise RuntimeError('Frozen checkpoint changed during test')
    for name,h in frozen['source_sha256'].items():
        if sha(ROOT/'lerobot_policy'/name)!=h:raise RuntimeError('Frozen code changed during test')
    print('FRESH RESULT',result['successes'],'/',result['completed'],'runtime',result['runtime_errors'],flush=True)
    (OUT/'fresh_test_complete.json').write_text(json.dumps(dict(checkpoint=str(cp),successes=result['successes'],count=result['completed'],runtime_errors=result['runtime_errors'],frozen_verified=True),indent=2)+'\n')

if __name__=='__main__':main()
