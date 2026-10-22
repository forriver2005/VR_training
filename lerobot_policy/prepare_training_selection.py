"""Exclude failed deliveries and freeze scene-disjoint training candidate lists.

Does not delete or rewrite source episodes. Lists are for a delivery-aware loader;
the legacy single-episode trainer does not consume them.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_BY_GROUP = {
    'v2_all_20261015_201918_02': 'train',
    'v2_all_20261016_172350_03': 'train',
    'v2_all_20261016_172350_06': 'test',
    'v2_all_20261018_193416_02.collection_retry1': 'train',
    'v2_all_20261018_193416_01': 'validation',
    'v2_all_20261018_193416_03.collection_retry1': 'train',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit-dir', type=Path, default=ROOT/'reports/data_delivery_audit_20261021')
    p.add_argument('--output', type=Path, default=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021')
    args = p.parse_args()
    details = read(args.audit_dir/'details.json')
    meta = read(args.audit_dir/'metadata.json')
    samples = read(args.audit_dir/'pointcloud_spotcheck.json')
    by_group = {g['group']: g for g in meta['groups']}
    fps_bad = {(s['group'], Path(s['episode']).stem) for s in samples if s['max_abs_error_m'] > 1e-6}
    lists = {k: [] for k in ['train_4hz', 'validation_4hz', 'test_4hz', 'reserve_8hz', 'excluded']}
    for g in details['groups']:
        group = g['group']; mg = by_group[group]
        if mg['errors'] or mg['source_hash_errors'] or g['missing_package_files']:
            raise ValueError(f'Incomplete audit/package: {group}')
        if any(g['checksums'][k] for k in ['missing', 'mismatched', 'invalid_lines']):
            raise ValueError(f'Checksum failures: {group}')
        diagnostics = {e['episode_id']: e for e in mg['episodes']}
        for episode in g['episodes'] + g['rejected']:
            eid = episode['episode_id']; d = diagnostics[eid]
            source = Path(episode['path'])
            if not source.is_absolute(): source = ROOT/source
            if not source.is_file(): raise FileNotFoundError(source)
            row = dict(uid=f'{group}/{eid}', group=group, episode_id=eid, scene_id=episode['scene_id'],
                npz=str(source.relative_to(ROOT)), source_npz_sha256=episode['sha256'],
                package_root=str(source.parent.parent.relative_to(ROOT)),
                diagnostics=str((source.parent.parent/'diagnostics'/f'{eid}.json').relative_to(ROOT)),
                replay_report=(d.get('replay') or {}).get('path'),
                frames=40, control_dt_s=episode['control_dt_s'],
                point_cloud_source=mg['config']['point_cloud_source'])
            reasons = []
            if not episode['success']: reasons.append('rejected_by_collector')
            if episode['errors'] or episode.get('semantic', {}).get('errors'): reasons.append('field_or_semantic_audit_failure')
            if not d['original_quality']['passed']: reasons.append('original_collection_quality_failure')
            if episode['success'] and not (d.get('replay') or {}).get('original_quality', {}).get('passed'):
                reasons.append('original_replay_quality_failure')
            if (group, eid) in fps_bad: reasons.append('fps_reconstruction_above_1e-6m')
            if reasons:
                row['reasons'] = reasons
                row['collection_failure_reasons'] = d.get('failure_reasons', [])
                row['replay_failure_reasons'] = (d.get('replay') or {}).get('original_quality', {}).get('reasons', [])
                lists['excluded'].append(row)
            elif episode['control_dt_s'] == .125:
                row['purpose'] = 'separate 8 Hz experiment; not mixed into the 4 Hz baseline'
                lists['reserve_8hz'].append(row)
            elif episode['control_dt_s'] == .25:
                lists[SPLIT_BY_GROUP[group]+'_4hz'].append(row)
            else:
                raise ValueError(f'Unexpected control period: {row}')
    seen = set(); scene_sets = []
    for key, entries in lists.items():
        for entry in entries:
            if entry['uid'] in seen: raise ValueError('Duplicate episode: '+entry['uid'])
            seen.add(entry['uid'])
        if key in ('train_4hz', 'validation_4hz', 'test_4hz'):
            scene_sets.append({e['scene_id'] for e in entries})
    if any(scene_sets[i]&scene_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Scene leakage across splits')
    lists['overfit_4hz'] = [next(e for e in lists['train_4hz']
        if e['group']=='v2_all_20261018_193416_03.collection_retry1' and e['episode_id']=='episode_000000')]
    args.output.mkdir(parents=True, exist_ok=True)
    for key, rows in lists.items():
        (args.output/f'{key}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    sources = [args.audit_dir/n for n in ['details.json', 'metadata.json', 'pointcloud_spotcheck.json']]
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        status='candidate lists prepared; new training/online adapters and local expert replay still required',
        paths_relative_to='DemoGrasp repository root; replay_report relative to package_root',
        audit_sources={str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        counts={k: len(v) for k,v in lists.items()},
        groups={k: dict(Counter(e['group'] for e in v)) for k,v in lists.items()},
        source_files_unchanged=True, source_npz_sha256='from full 2026-10-21 audit; loader must verify before use',
        pointcloud_scope='all-success geometric depth provenance; FPS equality sampled at 54 frames, not exhaustive',
        training_protocol=dict(H=2, K=16, A=8, control_dt_s=.25, physics_dt_s=1/120,
            physics_steps_per_frame=30, reset_settle_steps=240, frames=40,
            history_offsets=[-1,0], action_offsets=list(range(-1,15)), executed_prediction_slice=[1,9],
            point_cloud_source='wrist_depth', required_auxiliary_inputs=['history_valid','elapsed_same_target_s'],
            state_action_normalization_status='must be defined with new task bounds; legacy bounds invalid'),
        note='overfit_4hz is an intentional subset of train_4hz; reserve_8hz is valid data for a separate protocol, not a quality rejection')
    report['split_rationale'] = ('Freeze 4/1/1 scenes before fitting; include the elevated-start scene '
        '20261018_193416_03 in training so it is not the only height regime appearing exclusively in test. '
        'No policy test scores used to choose the split; validation/test each contain only one scene.')
    (args.output/'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['counts'], ensure_ascii=False, indent=2))


if __name__=='__main__': main()
