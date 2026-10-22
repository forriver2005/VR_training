"""Read-only audit of uploaded NPZ deliveries against the 4 cm / 6 cm contract.

Does not execute supplied collector code or certify physical replay success.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED = {
    'state': (40, 8), 'action': (40, 8), 'joint_position': (40, 7),
    'joint_target': (40, 7), 'camera_1_rgb': (40, 256, 256, 3),
    'camera_2_rgb': (40, 256, 256, 3), 'depth': (40, 2, 256, 256),
    'point_cloud': (40, 1024, 3), 'camera_intrinsics': (2, 3, 3),
    'camera_to_world': (2, 4, 4), 'camera_pose_usd_xyz_wxyz': (2, 7),
    'frame_index': (40,), 'is_first': (40,), 'history_valid': (40, 2),
    'observation_time_s': (40,), 'control_dt_s': (), 'success': (),
}
META = ['README.txt', 'manifest.json', 'meta/scenes.jsonl',
        'meta/collection_config.json', 'meta/camera_calibration.json',
        'meta/source_hashes.json', 'meta/splits.json', 'checksums.sha256']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def semantic_checks(data, path):
    """Cross-check measured arrays against their own package, without simulation."""
    root = path.parent.parent
    result = {'errors': []}
    def check(name, passed):
        if not passed:
            result['errors'].append(name)
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8-sig'))
    if not (root / 'meta/collection_config.json').is_file():
        return {'not_checked': 'package metadata missing'}
    config = read('meta/collection_config.json')
    calibration = read('meta/camera_calibration.json')
    diag = read('diagnostics/' + path.stem + '.json')
    for k in ['scene_id', 'episode_id', 'trajectory_id', 'seed', 'success', 'reset_settle_steps',
              'cube_a_size_m', 'cube_b_size_m']:
        check('diagnostic:' + k, k in data and k in diag and data[k].item() == diag[k])
    # The delivery contract permits physics step counts in metadata, not NPZ.
    if 'physics_steps_per_frame' in data:
        check('diagnostic:physics_steps_per_frame', data['physics_steps_per_frame'].item() == diag['physics_steps_per_frame'])
    else:
        result['physics_steps_per_frame_source'] = 'diagnostics and collection_config'
    check('physics_steps_metadata', config['physics_steps_per_frame'] == diag['physics_steps_per_frame'])
    check('control_period', np.isclose(float(data['control_dt_s']), config['control_dt_s'], atol=1e-8, rtol=0)
          and np.isclose(float(data['control_dt_s']), config['physics_dt'] * config['physics_steps_per_frame'], atol=1e-8, rtol=0)
          and np.isclose(float(data['control_dt_s']), diag['control_dt'], atol=1e-8, rtol=0))
    for k in ['camera_intrinsics', 'camera_to_world', 'camera_pose_usd_xyz_wxyz', 'wrist_camera_intrinsics', 'base_to_world']:
        check('calibration:' + k, k in data and k in calibration and np.allclose(data[k], calibration[k], atol=1e-7, rtol=0))
    reset = np.r_[diag['reset_target']['eef_xyz_xyzw'], diag['reset_target']['gripper_target_m']]
    check('reset_target', np.allclose(data['reset_target'], reset, atol=1e-7, rtol=0))
    check('initial_ik', np.allclose(data['initial_ik_joint_position'], diag['initial_ik_joint_position'], atol=1e-7, rtol=0))
    trace = diag.get('action_trace', [])
    check('trace_length', len(trace) == len(data['state']))
    if len(trace) == len(data['state']):
        target = np.asarray([t['target'] for t in trace])
        after = np.asarray([t['actual_after'] for t in trace])
        result['trace_action_max_abs_error'] = float(np.abs(target - data['action']).max())
        result['trace_next_state_max_abs_error'] = float(np.abs(after[:-1] - data['state'][1:]).max())
        check('trace_action', result['trace_action_max_abs_error'] < 1e-6)
        check('observe_before_action', result['trace_next_state_max_abs_error'] < 1e-6)
    a = data['action']; s = data['state']
    elapsed = 0.; active = None; expected = []
    for target in a:
        expected.append(elapsed)
        elapsed = elapsed + float(data['control_dt_s']) if active is not None and np.allclose(target, active, atol=1e-7, rtol=0) else float(data['control_dt_s'])
        active = target
    check('command_history_timer', 'elapsed_same_target_s' in data and np.allclose(expected, data['elapsed_same_target_s'], atol=1e-6, rtol=0))
    result['action_outside_old_bounds_frames'] = np.flatnonzero(np.any((a[:, :3] < [.15, -.45, 0]) | (a[:, :3] > [.95, .25, .95]), axis=1)).tolist()
    result['initial_state'] = s[0].tolist()
    result['initial_action'] = a[0].tolist()
    result['initial_reset_target'] = data['reset_target'].tolist()
    result['initial_ik'] = data['initial_ik_joint_position'].tolist()
    result['rgb_first_step_mae'] = [float(np.mean(np.abs(data[k][1].astype(float)-data[k][0]))) for k in ['camera_1_rgb', 'camera_2_rgb']]
    # Diagnostic candidates, not proof of perceptual aliasing: visual inputs may differ.
    candidates = []
    for t in range(1, len(s)-1):
        left = s[t-1:t+1]; right = s[t:t+2]
        pos = float(np.linalg.norm(left[:, :3]-right[:, :3], axis=1).max())
        grip = float(np.abs(left[:, 7]-right[:, 7]).max())
        q1 = left[:, 3:7] / np.linalg.norm(left[:, 3:7], axis=1, keepdims=True)
        q2 = right[:, 3:7] / np.linalg.norm(right[:, 3:7], axis=1, keepdims=True)
        angle = float(np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q1*q2, axis=1)), 0, 1))).max())
        delta = float(np.linalg.norm(a[t, :3]-a[t+1, :3]))
        gdelta = float(abs(a[t, 7]-a[t+1, 7]))
        if pos < .0001 and grip < .0001 and angle < .1 and (delta > .005 or gdelta > .002):
            candidates.append({'t': t, 'history_position_max_m': pos, 'action_delta_m': delta,
                'action_gripper_delta_m': gdelta, 'timer': data.get('elapsed_same_target_s', np.zeros(40))[t:t+2].tolist()})
    result['near_stationary_history_action_switches'] = candidates
    # Each saved point must map back to its wrist depth pixel at the same frame.
    # This tests depth provenance, not exact FPS ordering (checked separately).
    if config.get('point_cloud_source') == 'wrist_depth':
        points = data['point_cloud'].astype(np.float64)
        to_camera = np.linalg.inv(data['wrist_camera_to_world']) @ data['base_to_world']
        optical = np.einsum('tij,tnj->tni', to_camera[:, :3, :3], points) + to_camera[:, None, :3, 3]
        k = data['wrist_camera_intrinsics']; z = optical[:, :, 2]
        u = optical[:, :, 0] / np.maximum(z, 1e-12) * k[0, 0] + k[0, 2]
        v = optical[:, :, 1] / np.maximum(z, 1e-12) * k[1, 1] + k[1, 2]
        ui = np.floor(u).astype(int); vi = np.floor(v).astype(int)
        valid = (ui >= 0) & (ui < 256) & (vi >= 0) & (vi < 256) & (z > .01) & (z < 3)
        depth = data['wrist_depth'][np.arange(len(points))[:, None], vi.clip(0, 255), ui.clip(0, 255)]
        crop = np.asarray(config['crop_bounds_base'])
        in_crop = np.all((points >= crop[0]-1e-7) & (points <= crop[1]+1e-7), axis=-1)
        result['cloud_depth_max_error_m'] = float(np.abs(z-depth).max())
        result['cloud_pixel_center_max_error_px'] = float(max(np.abs(u-ui-.5).max(), np.abs(v-vi-.5).max()))
        result['cloud_invalid_projected_points'] = int((~valid).sum())
        result['cloud_points_outside_config_crop'] = int((~in_crop).sum())
        if bool(data['success']):
            check('cloud_depth_provenance', valid.all() and result['cloud_depth_max_error_m'] <= 1e-6 and result['cloud_pixel_center_max_error_px'] < .01)
            check('cloud_crop', in_crop.all())
    return result


def audit_episode(path):
    row = {'path': str(path), 'errors': [], 'warnings': []}
    before = path.stat()
    try:
        # Read every array (including images), exercising ZIP CRC validation.
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        row['schema'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in data.items()}
        row['missing_fields'] = sorted(set(REQUIRED) - data.keys())
        for k, shape in REQUIRED.items():
            if k not in data:
                row['errors'].append('missing:' + k)
            elif data[k].shape != shape:
                row['errors'].append('shape:' + k)
        for k, v in data.items():
            if np.issubdtype(v.dtype, np.number) and not np.isfinite(v).all():
                row['warnings' if k in ('depth', 'wrist_depth') else 'errors'].append('nonfinite:' + k)
        for k, dtype in {'camera_1_rgb': 'uint8', 'camera_2_rgb': 'uint8',
                         'depth': 'float32', 'point_cloud': 'float32',
                         'is_first': 'bool', 'success': 'bool'}.items():
            if k in data and str(data[k].dtype) != dtype:
                row['errors'].append('dtype:' + k)
        if row['errors']:
            return row
        checks = {
            'frame_index': np.array_equal(data['frame_index'], np.arange(40)),
            'is_first': np.array_equal(data['is_first'], np.arange(40) == 0),
            'history_valid': np.array_equal(data['history_valid'], np.r_[[[0, 1]], np.ones((39, 2))]),
            'time': np.allclose(np.diff(data['observation_time_s']), data['control_dt_s'], atol=1e-6, rtol=0)
                    and np.all(np.diff(data['observation_time_s']) > 0),
        }
        row['errors'] += [k for k, passed in checks.items() if not passed]
        for k in ['control_dt_s', 'success', 'scene_id', 'episode_id', 'trajectory_id', 'seed',
                  'cube_a_size_m', 'cube_b_size_m', 'physics_steps_per_frame', 'reset_settle_steps']:
            row[k] = data[k].item() if k in data else None
        row['extra_reset_fields_missing'] = [k for k in ['reset_target', 'initial_ik_joint_position',
            'base_to_world', 'reset_settle_steps', 'scene_id', 'episode_id', 'trajectory_id', 'seed',
            'cube_a_size_m', 'cube_b_size_m'] if k not in data]
        row['state_first_step_m'] = float(np.linalg.norm(data['state'][1, :3] - data['state'][0, :3]))
        row['action_first_step_m'] = float(np.linalg.norm(data['action'][1, :3] - data['action'][0, :3]))
        if 'reset_target' in data:
            row['reset_to_action0_m'] = float(np.linalg.norm(data['reset_target'][:3] - data['action'][0, :3]))
            row['reset_to_state0_m'] = float(np.linalg.norm(data['reset_target'][:3] - data['state'][0, :3]))
        row['quaternion_norm_max_error'] = {k: float(np.max(np.abs(np.linalg.norm(data[k][:, 3:7], axis=1)-1))) for k in ['state', 'action']}
        row['gripper_range'] = {k: [float(data[k][:, 7].min()), float(data[k][:, 7].max())] for k in ['state', 'action']}
        for k in ['depth', 'wrist_depth']:
            if k in data:
                v = data[k]; ratio = np.mean(~np.isfinite(v) | (v <= .01) | (v >= 3), axis=(-1, -2))
                row[k + '_invalid_ratio_range'] = [float(ratio.min()), float(ratio.max())]
        row['rgb_min_frame_std'] = {k: float(data[k].reshape(40, -1).std(1).min()) for k in ['camera_1_rgb', 'camera_2_rgb']}
        row['identical_camera_rgb_frames'] = int(np.all(data['camera_1_rgb'] == data['camera_2_rgb'], axis=(1, 2, 3)).sum())
        if 'point_cloud_valid_count' in data:
            counts = data['point_cloud_valid_count']
            row['point_count_range'] = [int(counts.min()), int(counts.max())]
            row['sparse_point_frames'] = int(np.sum(counts < 1024))
            row['empty_point_frames'] = int(np.sum(counts <= 0))
        points = data['point_cloud']
        row['point_bounds'] = [points.min((0, 1)).tolist(), points.max((0, 1)).tolist()]
        row['points_outside_old_crop'] = int(np.any((points < [.15, -.45, 0]) | (points > [.95, .25, .95]), axis=-1).sum())
        row['timer_present'] = 'elapsed_same_target_s' in data
        if row['timer_present']:
            row['timer_range'] = [float(data['elapsed_same_target_s'].min()), float(data['elapsed_same_target_s'].max())]
        row['diagnostic_present'] = (path.parent.parent / 'diagnostics' / (path.stem + '.json')).is_file()
        row['semantic'] = semantic_checks(data, path)
        row['sha256'] = sha256(path)
    except Exception as exc:
        row['errors'].append(type(exc).__name__ + ': ' + str(exc))
    after = path.stat()
    row['file_changed_during_audit'] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(args.root.resolve()),
              'scope': 'all locally complete NPZ arrays; no simulator replay; missing groups not substituted', 'groups': []}
    for root in sorted(args.root.glob('*/fr3v2')):
        paths = sorted((root / 'episodes').glob('*.npz'))
        group = {'group': root.parent.name, 'path': str(root), 'npz_count': len(paths),
                 'partial_files': [str(p.relative_to(root)) for p in sorted(root.rglob('*.codex-upload-part'))],
                 'missing_package_files': [p for p in META if not (root / p).is_file()]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            group['episodes'] = list(pool.map(audit_episode, paths))
            group['rejected'] = list(pool.map(audit_episode, sorted((root / 'rejected').glob('*.npz'))))
        group['field_errors_count'] = sum(bool(e['errors']) for e in group['episodes'])
        group['diagnostics'] = []
        for path in sorted((root / 'diagnostics').glob('episode_*.json')):
            try:
                d = json.loads(path.read_text(encoding='utf-8-sig'))
                keys = ['episode_id', 'scene_id', 'success', 'final_xy_error_m', 'final_z_error_m',
                        'stable_window_max_xy_error_m', 'stable_window_max_z_error_m', 'stable_window_drift_m',
                        'lifted', 'ik_failures', 'action_clamp_count', 'independent_replay_verified', 'control_dt']
                row = {k: d.get(k) for k in keys}
                row['path'] = str(path)
                row['missing_required_diagnostic_fields'] = [k for k in keys if k not in d]
                row['max_final_speed_m_s'] = max((float(np.linalg.norm(c['linear_velocity'])) for c in d.get('final_cube_states', [])), default=None)
                row['paired_npz_present'] = (root / 'episodes' / (path.stem + '.npz')).is_file()
                row['replay_evidence_present'] = bool(d.get('independent_replay_result_path')) and (root / d['independent_replay_result_path']).is_file()
                group['diagnostics'].append(row)
            except Exception as exc:
                group['diagnostics'].append({'path': str(path), 'error': str(exc)})
        if (root / 'manifest.json').is_file():
            group['manifest'] = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
        checksum = root / 'checksums.sha256'
        if checksum.is_file():
            checks = {'matched': [], 'missing': [], 'mismatched': [], 'invalid_lines': []}
            for line in checksum.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip(): continue
                pieces = line.split(maxsplit=1)
                if len(pieces) != 2:
                    checks['invalid_lines'].append(line); continue
                expected, name = pieces; name = name.lstrip('*').replace('\\', '/')
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()):
                    checks['invalid_lines'].append(line); continue
                checks['missing' if not path.is_file() else 'matched' if sha256(path) == expected else 'mismatched'].append(name)
            group['checksums'] = checks
        report['groups'].append(group)
        print(json.dumps({k: group[k] for k in ['group', 'npz_count', 'field_errors_count', 'missing_package_files']}, ensure_ascii=False), flush=True)
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
