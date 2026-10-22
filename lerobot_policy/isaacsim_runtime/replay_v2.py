"""Independent physics replay of archived joint commands and reset configuration."""
from pathlib import Path
import argparse,json,sys,traceback
for s in (sys.stdout,sys.stderr):
    if hasattr(s,'reconfigure'):s.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser()
parser.add_argument('dataset',type=Path)
parser.add_argument('--episode',default='episode_000000')
parser.add_argument('--headless',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--report-dir',type=Path,help='Diagnostic-only output; leave dataset reports unchanged')
args=parser.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless})
try:
    dataset=args.dataset.resolve()
    # Recovery tools live outside source/; use the archived robot description.
    import fr3_v2_core as core
    core.ROOT=dataset/'source'
    core.URDF=core.ROOT/'RobotDescriptions/franka_description-main/franka_description/fr3v2_franka_hand.urdf'
    from fr3_v2_sim import *
    from fr3_v2_delivery import update_checksums
    report_dir=args.report_dir.resolve() if args.report_dir else dataset/'replays'
    diagnostics_path=dataset/f'diagnostics/{args.episode}.json'
    diagnostics=json.loads(diagnostics_path.read_text(encoding='utf-8'))
    path=dataset/f'episodes/{args.episode}.npz'
    if not path.exists():path=dataset/f'rejected/{args.episode}.npz'
    with np.load(path,allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    runtime=Runtime(app,dataset/'source/scene.usda')
    rows=[json.loads(line) for line in (dataset/'meta/scenes.jsonl').read_text(encoding='utf-8').splitlines()]
    episode_row=next(row for row in rows if row['episode_id']==args.episode)
    obstacle_names={'/World/box':'box','/World/box1':'box (1)','/World/box2':'box (2)'}
    expected_obstacles=episode_row.get('obstacles',{})
    runtime.apply_obstacles([dict(name=obstacle_names[path],position=np.asarray(matrix)[:3,3],
                                 orientation_xyzw=Rotation.from_matrix(np.asarray(matrix)[:3,:3]).as_quat())
                             for path,matrix in expected_obstacles.items()])
    if not np.allclose(runtime.base_to_world,data['base_to_world'],atol=1e-6,rtol=0):
        raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: robot base transform')
    if any(not np.allclose(runtime.obstacle_poses().get(path),matrix,atol=1e-6,rtol=0)
           for path,matrix in expected_obstacles.items()):
        raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: obstacle transforms')
    from fr3_v2_sensors import Sensors
    sensors=Sensors(runtime)
    if not np.allclose(sensors.static_extrinsics,data['camera_to_world'],atol=1e-6,rtol=0):
        raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: camera extrinsics')
    if 'camera_pose_usd_xyz_wxyz' in data and not np.allclose(
            sensors.static_usd_poses,data['camera_pose_usd_xyz_wxyz'],atol=1e-6,rtol=0):
        raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: camera USD poses')
    q0=data['initial_ik_joint_position']
    initial_audit=runtime.restore_replay_initial(diagnostics['initial_cube_states'],q0,int(data['reset_settle_steps']))
    write_json(report_dir/f'{args.episode}_initial_state.json',initial_audit)
    if not initial_audit['passed']:
        raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: '+str(initial_audit['position_errors_m']))
    sensors.warmup_after_reset()
    initial=[runtime.cube_state(c) for c in runtime.cubes]
    runtime.contacts.clear()
    lifted=False;release_ok=False;frames=[];state_errors=[];rgb_stats=[];rgb_projection_audits=[];release_checks=[]
    def release_is_stable(cubes,frame):
        d=cubes[0]['position']-cubes[1]['position']
        speeds=[float(np.linalg.norm(c['linear_velocity'])) for c in cubes]
        fingers=runtime.robot.get_joint_positions()[7:].copy()
        passed=bool(np.linalg.norm(d[:2])<=RELEASE_XY_TOLERANCE_M and abs(d[2]-.05)<=.004 and
                    all(speed<RELEASE_SPEED_M_S for speed in speeds) and np.all(fingers>=.035))
        release_checks.append(dict(frame=frame,xy_error_m=float(np.linalg.norm(d[:2])),
                                   z_error_m=float(d[2]-.05),linear_speed_m_s=speeds,
                                   speed_limit_m_s=RELEASE_SPEED_M_S,finger_positions_m=fingers,passed=passed))
        return passed
    for i,(action,q) in enumerate(zip(data['action'],data['joint_target'])):
        state=runtime.observation_state()[0]
        state_errors.append(float(np.linalg.norm(state[:3]-data['state'][i,:3])))
        capture=sensors.capture()
        if i==0:
            initial_audit= replay_initial_audit(diagnostics['initial_cube_states'],[runtime.cube_state(c) for c in runtime.cubes])
            write_json(report_dir/f'{args.episode}_initial_state.json',initial_audit)
            if not initial_audit['passed']:
                raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: first rendered observation')
        rgb_projection_audits.append(sensors.audit_rgb_projection(capture))
        rgb_stats.append([float(capture[k].std()) for k in ['camera_1_rgb','camera_2_rgb']])
        # Equivalent visualization: paired views at all actual replay frames.
        from PIL import Image,ImageDraw
        image=Image.fromarray(np.hstack([capture['camera_1_rgb'],capture['camera_2_rgb']]))
        ImageDraw.Draw(image).text((5,5),f'Independent replay {i:02d}',fill='white')
        frames.append(image)
        runtime.command(q,action[7]);runtime.step(int(data.get('physics_steps_per_frame',STEPS_PER_ACTION)))
        if i>=33 and i<33+RELEASE_WINDOW_FRAMES:
            release_is_stable([runtime.cube_state(c) for c in runtime.cubes],i)
            if len(release_checks)>=2:
                release_ok=all(item['passed'] for item in release_checks[-2:])
        lifted=lifted or runtime.cubes[0].get_world_pose()[0][2]>initial[0]['position'][2]+.05
    result=runtime.evaluate(lifted,0,release_ok)
    result['initial_state_audit']=initial_audit
    result['release_checks']=release_checks
    result['rgb_frame_std']=rgb_stats
    result['rgb_projection_audits']=rgb_projection_audits
    if np.any(np.array(rgb_stats)<2):
        result['success']=False;result['failure_reasons'].append('replay_blank_rgb')
    if not any(sum(a[i]['matched'] for a in rgb_projection_audits)>=3 for i in range(2)):
        result['success']=False;result['failure_reasons'].append('replay_rgb_projection')
    result['reset_forbidden_contacts']=runtime.reset_forbidden_contacts
    result['max_observation_position_difference_m']=max(state_errors)
    result['command']=' '.join(sys.argv)
    write_json(report_dir/f'{args.episode}_verification.json',result)
    frames[0].save(report_dir/f'{args.episode}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
    if not args.report_dir:
        diagnostics['independent_replay_verified']=bool(result['success'])
        diagnostics['independent_replay_result_path']=f'replays/{args.episode}_verification.json'
        write_json(diagnostics_path,diagnostics)
        manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
        success_diags=[json.loads(p.read_text(encoding='utf-8')) for p in (dataset/'diagnostics').glob('*.json')]
        manifest['independent_replay_verified']=bool(manifest['successful_trajectories']) and all(d.get('independent_replay_verified',False) for d in success_diags if d['success'])
        write_json(dataset/'manifest.json',manifest)
        update_checksums(dataset)
    print('REPLAY_RESULT',result['success'],result['failure_reasons'],result['final_xy_error_m'],flush=True)
except BaseException:
    if 'sensors' in locals() and hasattr(sensors,'last_capture_failure'):
        sensors.last_capture_failure.update(episode_id=args.episode,frame=locals().get('i'))
        write_json(report_dir/f'{args.episode}_capture_failure.json',sensors.last_capture_failure)
    traceback.print_exc();raise
finally:
    app.close()

