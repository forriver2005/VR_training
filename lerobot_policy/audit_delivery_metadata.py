"""Check package links and recompute original quality gates from supplied telemetry.

This audits archived evidence; it does not perform an independent simulation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def original_quality(d):
    failures = []
    if not d.get('lifted'): failures.append('not_lifted')
    if d.get('ik_failures') != 0: failures.append('ik_failures')
    positions = np.asarray(d.get('stability_positions_world', []))
    if positions.shape != (60, 2, 3) or not np.isfinite(positions).all():
        failures.append('missing_stability_samples')
        return dict(passed=False, reasons=failures)
    xy = np.linalg.norm(positions[:, 0, :2]-positions[:, 1, :2], axis=1)
    z = np.abs(positions[:, 0, 2]-positions[:, 1, 2]-.05)
    drift = np.linalg.norm(positions-positions[0], axis=2).max(0)
    speed = [np.linalg.norm(c['linear_velocity']) for c in d.get('final_cube_states', [])]
    if xy.max() > .005: failures.append('xy_over_5mm')
    if z.max() > .004: failures.append('height_error_over_4mm')
    if drift.max() >= .001: failures.append('drift_over_1mm')
    if len(speed) != 2 or max(speed) >= .05: failures.append('final_speed_over_50mm_s_or_missing')
    if min(d.get('final_finger_positions_m', [0])) < .035: failures.append('fingers_not_open')
    if not any(set(pair) == {'/World/cubeA', '/World/cubeB'} for pair in d.get('contact_pairs_observed', [])):
        failures.append('stack_contact_not_recorded')
    result = dict(passed=not failures, reasons=failures, max_xy_m=float(xy.max()),
                  max_height_error_m=float(z.max()), max_drift_m=float(drift.max()),
                  max_final_speed_m_s=float(max(speed)) if speed else None)
    velocities = np.asarray(d.get('stability_linear_velocities_world', []))
    if velocities.shape == (60, 2, 3):
        result['max_window_speed_m_s'] = float(np.linalg.norm(velocities, axis=2).max())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'evidence': 'supplied JSON diagnostics and replay reports, not a fresh local replay', 'groups': []}
    for root in sorted(args.root.glob('*/fr3v2')):
        def read(name): return json.loads((root/name).read_text(encoding='utf-8-sig'))
        config = read('meta/collection_config.json'); manifest = read('manifest.json')
        sources = read('meta/source_hashes.json'); splits = read('meta/splits.json')
        scenes = [json.loads(line) for line in (root/'meta/scenes.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        group = {'group': root.parent.name, 'errors': [], 'episodes': [], 'config': config,
                 'split_counts': {k: len(splits[k]) for k in ['train', 'validation', 'test']}}
        def check(name, passed):
            if not passed: group['errors'].append(name)
        source_errors = []
        for rel, expected in sources.items():
            p = root/rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                source_errors.append(rel)
        group['source_hash_errors'] = source_errors
        check('manifest_sources', manifest['source_hashes'] == sources)
        check('manifest_total', len(scenes) == manifest['total_trajectories'])
        check('manifest_successes', sum(s['success'] for s in scenes) == manifest['successful_trajectories'])
        check('manifest_rejected', sum(not s['success'] for s in scenes) == manifest['rejected_trajectories'])
        check('scene_id_unique_episode', len({s['episode_id'] for s in scenes}) == len(scenes))
        check('manifest_control', manifest['control_dt_s'] == config['control_dt_s'])
        calibration_hash = hashlib.sha256((root/'meta/camera_calibration.json').read_bytes()).hexdigest()
        failures = Counter()
        for scene in scenes:
            eid = scene['episode_id']; d = read(scene['diagnostics_path'])
            check(eid+':source_hashes', scene['source_hashes'] == sources)
            check(eid+':camera_hash', scene['camera_config_sha256'] == calibration_hash)
            for k in ['scene_id', 'episode_id', 'trajectory_id', 'success']:
                check(eid+':diagnostic:'+k, scene[k] == d[k])
            check(eid+':size', scene['cube_a_size_m'] == d['cube_a_size_m'] == .04 and scene['cube_b_size_m'] == d['cube_b_size_m'] == .06)
            check(eid+':reset', scene['reset_target'] == d['reset_target'])
            check(eid+':initial_ik', scene['initial_ik_joint_position'] == d['initial_ik_joint_position'])
            check(eid+':obstacles', bool(scene.get('obstacles')))
            if scene.get('episode_path'):
                check(eid+':npz_exists', (root/scene['episode_path']).is_file())
            failures.update(d.get('failure_reasons', []))
            row = {'episode_id': eid, 'scene_id': scene['scene_id'], 'success': scene['success'],
                   'original_quality': original_quality(d), 'failure_reasons': d.get('failure_reasons', []),
                   'quality_xy_pass_label': scene.get('quality_xy_pass'),
                   'reset_forbidden_contacts': d.get('reset_forbidden_contacts'),
                   'obstacle_collision_pairs': d.get('obstacle_collision_pairs'),
                   'initial_scene_audit': d.get('initial_scene_audit'),
                   'calibration_audit_passed': d.get('calibration_audit', {}).get('passed'),
                   'initial_rgb_projection': (d.get('rgb_projection_audits') or [None])[0]}
            evidence = d.get('independent_replay_result_path')
            if scene['success']:
                check(eid+':replay_evidence', bool(evidence) and (root/evidence).is_file())
                if evidence and (root/evidence).is_file():
                    v = read(evidence)
                    row['replay'] = {'path': evidence, 'success': v.get('success'),
                        'original_quality': original_quality(v), 'initial_state_audit': v.get('initial_state_audit'),
                        'max_observation_position_difference_m': v.get('max_observation_position_difference_m')}
                    check(eid+':replay_success', v.get('success') is True)
                check(eid+':visualization', any((root/'replays'/(eid+ext)).is_file() for ext in ['.gif', '.mp4', '.png']))
            group['episodes'].append(row)
        check('failure_categories', dict(failures) == manifest['failure_categories'])
        ids = [eid for k in ['train', 'validation', 'test'] for eid in splits[k]]
        check('split_duplicate', len(ids) == len(set(ids)))
        check('split_coverage', set(ids) == {s['episode_id'] for s in scenes if s['success']})
        scene_by_id = {s['episode_id']: s['scene_id'] for s in scenes}
        scene_sets = [{scene_by_id.get(eid) for eid in splits[k]} for k in ['train', 'validation', 'test']]
        check('split_scene_leakage', not any(scene_sets[i]&scene_sets[j] for i in range(3) for j in range(i+1, 3)))
        group['scene_ids'] = sorted(set(scene_by_id.values()))
        result['groups'].append(group)
        good = [e for e in group['episodes'] if e['success']]
        print(json.dumps(dict(group=group['group'], metadata_errors=group['errors'], source_hash_errors=source_errors,
            success=len(good), original_collection_pass=sum(e['original_quality']['passed'] for e in good),
            original_replay_pass=sum(e.get('replay', {}).get('original_quality', {}).get('passed', False) for e in good)), ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
