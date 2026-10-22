"""Standalone entry: Isaac python.bat Tools/collect_v2.py --output <new directory>."""
from pathlib import Path
import sys, argparse, datetime, traceback, shutil, json, platform, subprocess, os
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path)
parser.add_argument('--input',type=Path,help='Version 2 Unity export; legacy files are rejected')
parser.add_argument('--limit',type=int,default=1)
parser.add_argument('--headless',action=argparse.BooleanOptionalAction,default=True)
args=parser.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless})
try:
    from fr3_v2_sim import *
    from fr3_v2_sensors import Sensors
    from fr3_v2_delivery import snapshot_sources, finalize_delivery
    output=(args.output or ROOT/'Outputs'/(VERSION+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be a new directory')
    for folder in ['episodes','rejected','diagnostics','meta','replays','source']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    runtime=Runtime(app)
    source_input=None
    if args.input:
        source_input=json.loads(args.input.read_text(encoding='utf-8-sig'))
        if source_input.get('schemaVersion')!=2:raise ValueError('Requires Unity schemaVersion=2')
        if source_input.get('cubeASizeM')!=.04 or source_input.get('cubeBSizeM')!=.06:raise ValueError('Requires 40/60mm cubes')
        if not np.allclose(source_input['robotBaseWorld'],runtime.base_to_world[:3,3],atol=1e-6):raise ValueError('Unity/Isaac base calibration differs')
        runtime.apply_obstacles(source_input.get('obstacles',[]))
    sensors=Sensors(runtime)
    write_json(output/'meta/geometry_audit.json',runtime.geometry_audit)
    calibration=sensors.calibration()
    write_json(output/'meta/camera_calibration.json',calibration)
    isaac_root=Path(os.environ.get('DEMOGRASP_ISAAC_ROOT',r'C:\Users\Wadmin\Desktop\isaac-sim'))
    config=dict(data_version=VERSION,field_version=2,trajectory_length=40,isaac_sim_version=(isaac_root/'VERSION').read_text().strip(),
                python=platform.python_version(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
                physics_dt=PHYSICS_DT,physics_steps_per_frame=STEPS_PER_ACTION,control_dt_s=CONTROL_DT,reset_settle_steps=240,
                phases=PHASE_COUNTS,position_unit='metres',quaternion='xyzw',state_source='PhysX actual TCP and finger joint1',
                action_semantics='FR3 base absolute TCP xyz + xyzw + single-finger joint target in metres',
                base_to_world=runtime.base_to_world,tcp_hand_offset=TCP_OFFSET,
                crop_bounds_base=CROP,fps='raster candidates <=4096 by linspace; farthest point sampling starts index0; cyclic padding',
                point_cloud_quality_policy=POINT_CLOUD_POLICY,
                release_policy=RELEASE_POLICY,release_settle_steps=RELEASE_SETTLE_STEPS,
                release_stable_window_steps=RELEASE_STABLE_STEPS,
                point_cloud_source='wrist_depth',point_cloud_reconstruction_atol=1e-6,
                camera_visibility_policy='at_least_one_fixed_camera_v1',
                rendering='Replicator orchestrator.step delta_time=0, rt_subframes=4; render must not increment physical time',
                canonical_scene='IsaacSim/FR3/FR3v2_cube40_cube60_v2.usda; composed geometry with obsolete render products removed',
                canonical_scene_sha256=sha256(ROOT/'IsaacSim/FR3/FR3v2_cube40_cube60_v2.usda'),
                camera_exposure='OmniRtxCameraExposureAPI ISO100, fStop1, time1; tonemap cm2Factor=0.0001 maps camera responsivity',
                anti_aliasing='disabled (/rtx/post/aa/op=0); temporal DLSS disabled to avoid stale multi-camera RGB',motion_blur=False,
                arm_drive_kp=20000.,arm_drive_kd=400.,finger_drive_kp=2000.,finger_drive_kd=40.,finger_force_limit_n=100.,
                finger_coupling='two force-limited drives receive same single-finger target; inherited mimic removed',
                friction_static=1.,friction_dynamic=1.,cube_density_kg_m3=1000.,
                 success_criteria=dict(xy_m=RELEASE_XY_TOLERANCE_M,z_delta_m=.050,z_error_m=.004,drift_m=.001,linear_speed_m_s=RELEASE_SPEED_M_S,
                                      extra_physics_steps=120,stable_window_steps=60,no_ik_failures=True,lift_min_m=.05,stack_contact_required=True),
                random_seed_rule='seed is recorded from Unity export; no hidden random sampling',
                online_condition_features=['elapsed_same_target_s; duration of the most recently sent target before each observation, derived only from command history; no expert phase labels'],
                collection_command=' '.join(sys.argv))
    write_json(output/'meta/collection_config.json',config)
    tasks=[dict(scene_id='original_scene',trajectoryId=i,seed=i) for i in range(args.limit)]
    if args.input:
        tasks=source_input['trajectories'][:args.limit]
        shutil.copy2(args.input,output/'source/unity_input.json')
    sources=snapshot_sources(output,runtime)
    rows=[]
    for episode_index,task in enumerate(tasks):
        episode_id=f'episode_{episode_index:06d}'
        print('COLLECT',episode_id,flush=True)
        a=task.get('cube_a_world'); b=task.get('cube_b_world')
        path=task.get('transport_path_base')
        if source_input:
            a=source_input['cubeAWorld'];b=source_input['cubeBWorld']
        scene_spec=dict(cube_a_world=a,cube_b_world=b,obstacles=source_input.get('obstacles',[]) if source_input else 'original')
        scene_key=hashlib.sha256(json.dumps(scene_spec,sort_keys=True).encode()).hexdigest()[:16]
        common=dict(episode_id=episode_id,scene_id=scene_key,trajectory_id=task['trajectoryId'],seed=task.get('seed',0),
                    cube_a_size_m=.04,cube_b_size_m=.06,base_to_world=runtime.base_to_world,
                    physics_dt=PHYSICS_DT,physics_steps_per_frame=STEPS_PER_ACTION,control_dt=CONTROL_DT,
                    reset_settle_steps=240,source_hashes=sources,obstacles=runtime.obstacle_poses())
        try:
            reset,q0,actions,phases,initial=runtime.reset(a,b,transport_path=path)
        except ValueError as e:
            diag=dict(common,success=False,failure_reasons=['reset_ik_failure'],error=str(e),
                      requested_cube_a_world=a,requested_cube_b_world=b)
            write_json(output/f'diagnostics/{episode_id}.json',diag)
            rows.append(dict(common,success=False,quality_xy_pass=False,episode_path=None,diagnostics_path=f'diagnostics/{episode_id}.json'))
            continue
        reset_target=dict(eef_xyz_xyzw=reset[:7],gripper_target_m=reset[7],gripper_definition='single finger joint1 displacement',
                          initial_ik_joint_position=q0,base_to_world=runtime.base_to_world,physics_dt=PHYSICS_DT,
                          control_dt_s=CONTROL_DT,reset_settle_steps=240,scene_settle_steps=240,scene_id=scene_key,
                          trajectory_id=task['trajectoryId'],seed=task.get('seed',0),usd_sha256=sources['source/scene.usda'],urdf_sha256=sha256(URDF))
        joint_targets=[]
        # Solve before recording; invalid IK is an explicit rejected diagnostic.
        try:
            for action in actions: joint_targets.append(runtime.kin.ik(action))
        except ValueError as e:
            diag=dict(common,success=False,failure_reasons=['ik_failure'],error=str(e),reset_target=reset_target,initial_ik_joint_position=q0,initial_cube_states=initial)
            write_json(output/f'diagnostics/{episode_id}.json',diag)
            rows.append(dict(common,reset_target=reset_target,initial_ik_joint_position=q0,quality_xy_pass=False,success=False,episode_path=None,diagnostics_path=f'diagnostics/{episode_id}.json'))
            continue
        runtime.command(q0,.04)
        # Reset/scene transforms are now committed. Refresh render products before
        # recording the first observation so it cannot reuse a pre-reset texture.
        sensors.warmup_after_reset()
        settled_initial_scene=np.asarray([state['position'] for state in initial],dtype=float)
        requested_initial_scene=(np.asarray([a,b],dtype=float) if a is not None and b is not None
                                 else settled_initial_scene.copy())
        initial_scene_position_errors=np.linalg.norm(settled_initial_scene-requested_initial_scene,axis=1)
        initial_scene_audit=dict(policy='requested_vs_settled_v1',requested_positions_world=requested_initial_scene,
                                 settled_positions_world=settled_initial_scene,
                                 position_errors_m=initial_scene_position_errors,
                                 passed=bool(np.all(initial_scene_position_errors<=.005)))
        buffers={}; trace=[]; lifted=False; release_ok=False; release_pose=None; release_window_checks=[]; rgb_projection_audits=[]
        calibration_candidates=[None,None]; calibration_visible_frames=[None,None]
        # This timer is available before the current action and is derived
        # only from commands already sent to the controller.  It therefore
        # carries no future expert-action information into the observation.
        active_target=None
        elapsed=0.
        for frame,(action,q,phase) in enumerate(zip(actions,joint_targets,phases)):
            # Observation precedes its matching action. Camera refresh has zero physics steps.
            state,measured_q=runtime.observation_state()
            capture=sensors.capture()
            rgb_projection_audits.append(sensors.audit_rgb_projection(capture))
            # Use the first visible depth sample for each camera.  The initial
            # frame can be occluded by a scene obstacle.
            calibration_frame=sensors.audit_calibration(capture)
            for camera_index,row in enumerate(calibration_frame['cameras']):
                if calibration_candidates[camera_index] is None or (
                        row['visible_cube_surface_samples']>0 and
                        calibration_candidates[camera_index]['visible_cube_surface_samples']==0):
                    calibration_candidates[camera_index]=row
                    if row['visible_cube_surface_samples']>0:
                        calibration_visible_frames[camera_index]=frame
            observation=dict(state=state,joint_position=measured_q,observation_time_s=runtime.sim.current_time,
                             elapsed_same_target_s=elapsed,**capture)
            for key,value in observation.items(): buffers.setdefault(key,[]).append(value)
            before=[runtime.cube_state(c) for c in runtime.cubes]
            if phase=='release' and frame==33:
                # Record the place pose before opening.  Release is evaluated
                # after the open command so the cube may naturally settle.
                release_pose=action[:3].copy()
            if phase in ['release','retreat']:
                # Always open during release/retreat.  Previously a failed
                # release forced .016 m (closed), which carried Cube A upward.
                action[7]=.04
            if phase=='retreat' and not release_ok and release_pose is not None:
                action[:3]=release_pose
                q=runtime.kin.ik(action)
                # Persist the held pose in rejected-episode diagnostics and
                # replay data as well; otherwise replay would reintroduce the
                # original upward retreat command.
                joint_targets[frame]=q.copy()
            runtime.command(q,action[7]); runtime.step(STEPS_PER_ACTION)
            if active_target is not None and np.allclose(action,active_target,atol=1e-7,rtol=0):
                elapsed += CONTROL_DT
            else:
                elapsed = CONTROL_DT
            active_target=action.copy()
            after=[runtime.cube_state(c) for c in runtime.cubes]
            if phase=='release':
                relative=after[0]['position']-after[1]['position']
                speeds=[float(np.linalg.norm(c['linear_velocity'])) for c in after]
                fingers=runtime.robot.get_joint_positions()[7:].copy()
                release_window_checks.append(dict(frame=frame,xy_error_m=float(np.linalg.norm(relative[:2])),
                    z_error_m=float(relative[2]-.05),linear_speed_m_s=speeds,
                    finger_positions_m=fingers,passed=bool(np.linalg.norm(relative[:2])<=RELEASE_XY_TOLERANCE_M and
                        abs(relative[2]-.05)<=.004 and all(v<RELEASE_SPEED_M_S for v in speeds) and
                        np.all(fingers>=.035))))
                if len(release_window_checks)>=2:
                    release_ok=all(row['passed'] for row in release_window_checks[-2:])
            lifted=lifted or after[0]['position'][2]>initial[0]['position'][2]+.05
            trace.append(dict(frame=frame,target=action,actual_after=runtime.observation_state()[0],cube_states=after,
                              measured_gripper_both_m=runtime.robot.get_joint_positions()[7:].copy(),contacts=sorted(runtime.contacts)))
            print('FRAME',frame,phase,'cloud',capture['point_cloud_valid_count'],flush=True)
        calibration_audit=dict(
             passed=any(row is not None and row['max_projection_difference_pixels']<.001 and
                        row['visible_cube_surface_samples']>0 for row in calibration_candidates),
            cameras=calibration_candidates,
            visible_frame_by_camera=calibration_visible_frames,
             visibility_policy='first visible depth sample per camera; at least one camera must be visible',
             required_visible_cameras=1)
        diag=runtime.evaluate(lifted,0,release_ok)
        if runtime.reset_forbidden_contacts:
            diag['success']=False;diag['failure_reasons'].append('reset_contact')
        # Sensor failures cannot enter the training set even if physics succeeded.
        cloud_audit=cloud_quality(buffers['point_cloud_valid_count'])
        if not cloud_audit['passed']:
            diag['success']=False;diag['failure_reasons'].append('insufficient_wrist_point_cloud')
        if not calibration_audit['passed']:
            diag['success']=False;diag['failure_reasons'].append('camera_calibration_audit')
        if not any(sum(a[i]['matched'] for a in rgb_projection_audits)>=3 for i in range(2)):
            diag['success']=False;diag['failure_reasons'].append('rgb_pose_projection_audit')
        if not initial_scene_audit['passed']:
            diag['success']=False;diag['failure_reasons'].append('initial_scene_mismatch')
        for name in ['camera_1_rgb','camera_2_rgb']:
            if any(image.std()<2 for image in buffers[name]):
                diag['success']=False;diag['failure_reasons'].append('blank_rgb_'+name)
        diag.update(episode_id=episode_id,scene_id=reset_target['scene_id'],trajectory_id=task['trajectoryId'],seed=reset_target['seed'],
                    reset_target=reset_target,initial_ik_joint_position=q0,base_to_world=runtime.base_to_world,reset_settle_steps=240,
                    physics_dt=PHYSICS_DT,physics_steps_per_frame=STEPS_PER_ACTION,control_dt=CONTROL_DT,
                    cube_a_size_m=.04,cube_b_size_m=.06,cube_mass_kg=[.064,.216],initial_cube_states=initial,
                    scene_settle_steps=240,reset_contact_pairs=runtime.reset_contacts,reset_forbidden_contacts=runtime.reset_forbidden_contacts,
                    action_clamp_count=0,depth_invalid_ratio=np.array(buffers['depth_invalid_ratio']),
                    point_cloud_valid_count=buffers['point_cloud_valid_count'],point_cloud_empty_frames=sum(n==0 for n in buffers['point_cloud_valid_count']),
                    point_cloud_insufficient_frames=len(cloud_audit['sparse_frames']),action_trace=trace,
                    point_cloud_quality_audit=cloud_audit,release_window_checks=release_window_checks,
                    calibration_audit=calibration_audit,initial_scene_audit=initial_scene_audit,
                    rgb_projection_audits=rgb_projection_audits,independent_replay_verified=False)
        data={key:np.array(values) for key,values in buffers.items()}
        data.update(action=actions.astype(np.float32),joint_target=np.array(joint_targets,dtype=np.float32),
                    camera_intrinsics=sensors.intrinsics[:2],camera_to_world=sensors.static_extrinsics,
                    camera_pose_usd_xyz_wxyz=sensors.static_usd_poses,wrist_camera_intrinsics=sensors.intrinsics[2],
                    base_to_world=runtime.base_to_world,frame_index=np.arange(40,dtype=np.int32),is_first=np.arange(40)==0,
                    history_valid=np.vstack([[0,1],np.ones((39,2),dtype=np.uint8)]).astype(np.uint8),
                    control_dt_s=np.array(CONTROL_DT),physics_steps_per_frame=np.array(STEPS_PER_ACTION),
                    success=np.array(diag['success'],dtype=np.bool_),
                    point_cloud_quality_policy=np.array(POINT_CLOUD_POLICY),
                    scene_id=np.array(diag['scene_id']),episode_id=np.array(episode_id),trajectory_id=np.array(task['trajectoryId']),
                    seed=np.array(diag['seed']),reset_target=reset.astype(np.float32),initial_ik_joint_position=q0,
                    reset_settle_steps=np.array(240),cube_a_size_m=np.array(.04),cube_b_size_m=np.array(.06))
        folder='episodes' if diag['success'] else 'rejected'
        np.savez_compressed(output/f'{folder}/{episode_id}.npz',**data)
        write_json(output/f'diagnostics/{episode_id}.json',diag)
        rows.append(dict(episode_id=episode_id,scene_id=diag['scene_id'],trajectory_id=task['trajectoryId'],success=diag['success'],
                         episode_path=f'{folder}/{episode_id}.npz',diagnostics_path=f'diagnostics/{episode_id}.json',
                         cube_a_size_m=.04,cube_b_size_m=.06,reset_target=reset_target,initial_ik_joint_position=q0,
                    quality_xy_pass=bool(diag['stable_window_max_xy_error_m']<=RELEASE_XY_TOLERANCE_M),source_hashes=sources,
                    obstacles=runtime.obstacle_poses(),
                         camera_config_sha256=sha256(output/'meta/camera_calibration.json')))
        print('EPISODE_RESULT',diag['success'],diag['failure_reasons'],flush=True)
    finalize_delivery(output,rows,config,sources)
    print('DATASET',output,flush=True)
except BaseException:
    if 'sensors' in locals() and hasattr(sensors,'last_capture_failure'):
        sensors.last_capture_failure.update(episode_id=locals().get('episode_id'),frame=locals().get('frame'))
        write_json(output/'meta/capture_failure.json',sensors.last_capture_failure)
    traceback.print_exc()
    raise
finally:
    app.close()

