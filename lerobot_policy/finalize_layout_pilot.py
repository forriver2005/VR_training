"""Admit new physical layouts only after successful independent replay."""
import argparse,json
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,sha256,load_entries


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);args=p.parse_args()
    entries=load_entries(args.collection/'collected_candidates.jsonl');replay=json.loads((args.replay/'summary.json').read_text())
    trials={r['uid']:r for r in replay['trials']};accepted=[];excluded=[]
    for e in entries:
        trial=trials.get(e['uid']);reasons=[]
        if trial is None or not trial['success'] or 'error' in trial:reasons.append('independent_replay_failed')
        else:
            diag=json.loads(Path(trial['result']).read_text())
            if not diag['initial_state_audit']['passed'] or diag['ik_failures'] or diag['action_clipping_count']:reasons.append('runtime_contract')
            if diag['first_observation_position_error_m']>.001:reasons.append('initial_observation_alignment')
        with np.load(ROOT/e['npz']) as z:
            required=['depth','wrist_depth','wrist_camera_to_world','wrist_camera_intrinsics','camera_intrinsics','camera_to_world','base_to_world','point_cloud','state','action','joint_position','joint_target','reset_target','initial_ik_joint_position','is_first','history_valid']
            if any(k not in z for k in required):reasons.append('missing_field')
            if not np.allclose(np.diff(z['observation_time_s']),.25,rtol=0,atol=1e-6):reasons.append('timing')
            if np.any(z['depth_invalid_ratio']>=1):reasons.append('empty_depth')
            if not all(np.isfinite(z[k]).all() for k in ['state','action','point_cloud','depth','wrist_depth']):reasons.append('nonfinite')
        if reasons:excluded.append(dict(entry=e,reasons=reasons))
        else:accepted.append(dict(e,replay_report=str(Path(trial['result']).relative_to(ROOT))))
    if not accepted:raise ValueError('No usable new layouts')
    original=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    old=load_entries(original/'pilot20_4hz.jsonl');val=load_entries(original/'validation_quick_4hz.jsonl')
    # Freeze a composition holdout before fitting: A1/B1 is never trained.
    # A0/B-/B+ teaches distinct targets; A1/B- supplies the new grasp location.
    local_holdout=[e for e in accepted if e['episode_id'].endswith('_a1_b1')]
    new_train=[e for e in accepted if not e['episode_id'].endswith('_a1_b1')]
    forbidden={e['scene_id'] for e in val+load_entries(original/'test_4hz.jsonl')}
    if any(e['scene_id'] in forbidden or e.get('parent_scene_id') in forbidden for e in accepted):raise ValueError('Split leakage')
    target=args.collection/'manifests';target.mkdir(exist_ok=False)
    for name,rows in [('new_train',new_train),('local_holdout',local_holdout),('augmented_train',old+new_train),('closed_checks',load_entries(original/'pilot_scene_checks_4hz.jsonl')+accepted)]:
        (target/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tasks=[dict(index=i,seed=s) for i in range(4) for s in range(5)]+[dict(index=4,seed=s) for s in range(10)]
    tasks += [dict(index=i,seed=0) for i in range(5,5+len(accepted))]
    (target/'closed_tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (target/'baseline_new_tasks.json').write_text(json.dumps(tasks[30:],indent=2)+'\n')
    result=dict(collected_candidates=len(entries),accepted=len(accepted),excluded=excluded,train_total=len(old)+len(new_train),
        new_train=len(new_train),local_holdout=len(local_holdout),local_holdout_rule='a1_b1 in each original training background; not used for checkpoint selection',
        manifest_hashes={p.name:sha256(p) for p in target.iterdir()},test_used=False)
    (args.collection/'admission_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
