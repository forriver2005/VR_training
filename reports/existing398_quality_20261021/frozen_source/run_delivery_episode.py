"""Independent new-task expert or policy rollout; never edits the delivery package."""
import argparse
import json
import os
import socket
import sys
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT, HoldTimer, wrist_point_cloud, fused_point_cloud, fused_view_quotas, POINT_SOURCES, sha256, load_entries,load_episode
from socket_protocol import send_arrays, receive_arrays, command_message, read_command

RUNTIME_PACKAGE=ROOT/'trajectory_replays/v2_all_20261018_193416_03.collection_retry1/fr3v2'

def verify_runtime_contract(package):
    """One reviewed runtime, package-specific assets; reject different dynamics."""
    config=json.loads((package/'meta/collection_config.json').read_text())
    expected=dict(control_dt_s=.25,physics_steps_per_frame=30,reset_settle_steps=240,
                  arm_drive_kp=20000.,arm_drive_kd=400.,finger_drive_kp=2000.,finger_drive_kd=40.,
                  finger_force_limit_n=100.,friction_static=1.,friction_dynamic=1.)
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError(f'Runtime contract mismatch {k}: {config.get(k)} != {v}')
    package_sources=json.loads((package/'meta/source_hashes.json').read_text())
    for name in ['source/scene.usda','source/RobotDescriptions/franka_description-main/franka_description/fr3v2_franka_hand.urdf']:
        if sha256(package/name)!=package_sources[name]:raise ValueError('Changed asset '+name)
    runtime_sources=json.loads((RUNTIME_PACKAGE/'meta/source_hashes.json').read_text())
    checked={}
    for name in ['fr3_v2_core.py','fr3_v2_sim.py','fr3_v2_sensors.py']:
        rel='source/Tools/'+name;actual=sha256(RUNTIME_PACKAGE/rel)
        if actual!=runtime_sources[rel]:raise ValueError('Changed reference runtime '+rel)
        checked[rel]=actual
    return checked

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--mode',choices=['expert_joint','expert_ik','policy'],default='expert_ik')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=6200)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--contract',type=Path)
    p.add_argument('--server-token',help='launcher-generated identity token for the reserved policy listener')
    p.add_argument('--pointcloud-source',choices=list(POINT_SOURCES))
    p.add_argument('--layout-spec',type=Path,help='Physically recollect a new layout from a training source; expert_ik only')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    entry=load_entries(args.list)[args.index];package=ROOT/entry['package_root']
    if sha256(ROOT/entry['npz'])!=entry['source_npz_sha256']:raise ValueError('Source NPZ changed')
    with np.load(ROOT/entry['npz']) as z:
        data={k:z[k] for k in ['state','action','joint_target','initial_ik_joint_position','base_to_world','reset_settle_steps','camera_to_world','camera_intrinsics','point_cloud','reset_target']}
    diag=json.loads((ROOT/entry['diagnostics']).read_text())
    scene=(json.loads((ROOT/entry['scene_metadata']).read_text()) if 'scene_metadata' in entry else
           next(json.loads(l) for l in (package/'meta/scenes.jsonl').read_text().splitlines() if json.loads(l)['episode_id']==entry['episode_id']))
    layout=None
    if args.layout_spec:
        if args.mode!='expert_ik':raise ValueError('New layouts must first be collected with the expert')
        from layout_variants import shifted_episode
        layout=json.loads(args.layout_spec.read_text())
        data,diag=shifted_episode(entry,data,diag,layout)
    if args.mode=='policy':
        if args.contract is None:raise ValueError('Policy requires frozen checkpoint contract')
        contract=json.loads(args.contract.read_text())
        if contract['protocol_version']!='delivery_4hz_v1' or contract['control_dt_s']!=.25:raise ValueError('Policy contract mismatch')
    else:contract=None
    point_source=contract['pointcloud_contract']['source'] if contract else (args.pointcloud_source or 'wrist_depth')
    if point_source not in POINT_SOURCES:raise ValueError('Unsupported point-cloud source')
    if contract and args.pointcloud_source and args.pointcloud_source!=point_source:raise ValueError('Point-cloud source conflicts with checkpoint')
    if point_source=='fused_depth':
        data['point_cloud']=load_episode(entry,contract['timer'] if contract else None,point_source)['point_cloud']
    runtime_hashes=verify_runtime_contract(package)
    # Some early packages have no restore_replay_initial. Use the validated
    # reference implementation for every scene, retaining each scene's assets.
    sys.path.insert(0,str(RUNTIME_PACKAGE/'source/Tools'))
    from isaacsim import SimulationApp
    # Vulkan uses the physical device index; CUDA is remapped by visibility.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    if not visible.isdigit():raise ValueError('Use a numeric CUDA_VISIBLE_DEVICES for this runner')
    app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':int(visible),'physics_gpu':0})
    connection=None;completed=False
    try:
        import fr3_v2_core as core
        core.ROOT=package/'source';core.URDF=core.ROOT/'RobotDescriptions/franka_description-main/franka_description/fr3v2_franka_hand.urdf'
        # Restore the original quality gates, never the relaxed delivery gates.
        core.RELEASE_XY_TOLERANCE_M=.005;core.RELEASE_SPEED_M_S=.05
        from fr3_v2_sim import Runtime,Rotation
        import fr3_v2_sensors as sensor_module
        if point_source=='fused_depth':
            # The capture helper's wrist-only cloud is replaced below. An empty
            # wrist is allowed only when other measured views supply the cloud.
            def optional_wrist_cloud(*values):
                try:return wrist_point_cloud(*values)
                except ValueError as exc:
                    if 'Empty cropped' not in str(exc):raise
                    return np.empty((0,3),dtype=np.float32),0
            sensor_module.depth_to_cloud=optional_wrist_cloud
        else:sensor_module.depth_to_cloud=wrist_point_cloud
        runtime=Runtime(app,package/'source/scene.usda')
        names={'/World/box':'box','/World/box1':'box (1)','/World/box2':'box (2)'}
        runtime.apply_obstacles([dict(name=names[k],position=np.asarray(v)[:3,3],
            orientation_xyzw=Rotation.from_matrix(np.asarray(v)[:3,:3]).as_quat()) for k,v in scene['obstacles'].items()])
        sensors=sensor_module.Sensors(runtime)
        if not np.allclose(sensors.static_extrinsics,data['camera_to_world'],atol=1e-6,rtol=0):raise ValueError('Camera extrinsics mismatch')
        if not np.allclose(sensors.intrinsics[:2],data['camera_intrinsics'],atol=1e-6,rtol=0):raise ValueError('Camera intrinsics mismatch')
        if not np.allclose(runtime.base_to_world,data['base_to_world'],atol=1e-6,rtol=0):raise ValueError('Base transform mismatch')
        if layout:
            # IK validation does not execute or fabricate any observations.
            runtime.kin.seed=data['initial_ik_joint_position'].copy()
            data['initial_ik_joint_position']=runtime.kin.ik(data['reset_target'])
            for action in data['action']:runtime.kin.ik(action)
        initial_audit=runtime.restore_replay_initial(diag['initial_cube_states'],data['initial_ik_joint_position'],int(data['reset_settle_steps']))
        if not initial_audit['passed']:raise RuntimeError('Initial state restoration failed: '+str(initial_audit))
        sensors.warmup_after_reset();runtime.kin.seed=data['initial_ik_joint_position'].copy()
        initial=[runtime.cube_state(c) for c in runtime.cubes];runtime.contacts.clear()
        timer=HoldTimer(contract['timer'] if contract else None)
        goal_memory=None
        if contract and contract.get('goal_memory'):
            from goal_observation_registry import goal_estimator
            GoalMemory,_,goal_code=goal_estimator(contract['goal_memory']['recipe'])
            if sha256(goal_code)!=contract['goal_memory']['code_sha256']:raise ValueError('Goal extractor code changed')
            goal_memory=GoalMemory()
        if args.mode=='policy':
            connection=socket.create_connection(('127.0.0.1',args.port),timeout=120)
            send_arrays(connection,command_message('ping'));identity=receive_arrays(connection)
            if read_command(identity)!='pong':raise ValueError('Unexpected policy endpoint')
            if args.server_token is not None and str(identity.get('ready_token',''))!=args.server_token:raise ValueError('Policy server identity mismatch')
            if 'checkpoint_contract_sha256' in identity and str(identity['checkpoint_contract_sha256'])!=sha256(args.contract):raise ValueError('Connected policy checkpoint differs from requested contract')
            send_arrays(connection,{**command_message('reset'),'episode_id':np.asarray(0),'seed':np.asarray(args.seed)})
            if read_command(receive_arrays(connection))!='ok':raise ValueError('Policy reset failed')
        buffers={};frames=[];ik_failures=0;clipping=0;fallbacks=0;calls=0;lifted=False;ik_errors=[];grip_projections=orientation_projections=anchor_projections=0
        for t in range(40):
            state,measured_q=runtime.observation_state();capture=sensors.capture()
            if goal_memory is not None:
                capture['goal_features']=goal_memory.observe([capture['camera_1_rgb'],capture['camera_2_rgb']],capture['depth'],sensors.intrinsics[:2],sensors.static_extrinsics,runtime.base_to_world)
            if point_source=='fused_depth':
                cloud,view_counts=fused_point_cloud(capture['depth'],sensors.intrinsics[:2],sensors.static_extrinsics,
                    capture['wrist_depth'],sensors.intrinsics[2],capture['wrist_camera_to_world'],runtime.base_to_world)
                capture['point_cloud']=cloud;capture['point_cloud_view_counts']=view_counts
                capture['point_cloud_view_quotas']=fused_view_quotas(view_counts)
                capture['point_cloud_valid_count']=int(view_counts.sum())
            if t==0:
                first_cloud_error=None if layout else float(np.max(np.abs(capture['point_cloud']-data['point_cloud'][0])))
                first_state_error=None if layout else float(np.linalg.norm(state[:3]-data['state'][0,:3]))
            observation=dict(state=state,joint_position=measured_q,elapsed_same_target_s=timer.elapsed,
                observation_time_s=runtime.sim.current_time,**capture)
            for k,v in observation.items():buffers.setdefault(k,[]).append(v)
            if connection:
                send_arrays(connection,{**command_message('predict'),'episode_id':np.asarray(0),'request_id':np.asarray(t),
                    'observation.state':state[None],'observation.point_cloud':capture['point_cloud'][None],
                    'observation.elapsed_same_target_s':np.asarray([timer.elapsed],dtype=np.float32),
                    **({'observation.goal_features':capture['goal_features'][None]} if goal_memory is not None else {}),
                    'observation.camera_1.rgb':capture['camera_1_rgb'][None],'observation.camera_2.rgb':capture['camera_2_rgb'][None]})
                reply=receive_arrays(connection)
                if read_command(reply)!='action' or int(reply['request_id'])!=t or int(reply['episode_id'])!=0:raise ValueError('Policy response mismatch')
                action=reply['action'][0]
                clipping+=int(reply['action_clipped']);fallbacks+=int(reply['quaternion_fallback']);calls=int(reply['inference_calls'])
                grip_projections+=int(reply.get('gripper_projected',False));orientation_projections+=int(reply.get('orientation_projected',False))
                anchor_projections=int(reply.get('anchor_projection_count',0))
            else:action=data['action'][t].copy()
            try:
                q=data['joint_target'][t] if args.mode=='expert_joint' else runtime.kin.ik(action)
            except ValueError as exc:
                ik_failures+=1;ik_errors.append(dict(frame=t,error=str(exc)))
                q=measured_q.copy()
            runtime.command(q,float(action[7]));runtime.step(30);timer.advance(action,.25)
            for k,v in [('action',action),('joint_target',q)]:buffers.setdefault(k,[]).append(v)
            current=[runtime.cube_state(c) for c in runtime.cubes]
            buffers.setdefault('cube_position_world',[]).append([c['position'] for c in current])
            lifted=lifted or current[0]['position'][2]>initial[0]['position'][2]+.05
            from PIL import Image,ImageDraw
            image=Image.fromarray(np.hstack([capture['camera_1_rgb'],capture['camera_2_rgb']]))
            ImageDraw.Draw(image).text((5,5),f'{args.mode} seed={args.seed} frame={t:02d}',fill='white');frames.append(image)
        if not np.allclose(np.diff(buffers['observation_time_s']),.25,atol=1e-6,rtol=0):raise RuntimeError('Incorrect observation/control timing')
        result=runtime.evaluate(lifted,ik_failures,False)
        result.update(mode=args.mode,seed=args.seed,uid=entry['uid'],initial_state_audit=initial_audit,
            pointcloud_source=point_source,pointcloud_recipe=POINT_SOURCES[point_source],
            scene_id=layout['scene_id'] if layout else entry['scene_id'],runtime_package=str(RUNTIME_PACKAGE),runtime_source_sha256=runtime_hashes,
            initial_cube_states=initial,initial_ik_joint_position=data['initial_ik_joint_position'],reset_target=data['reset_target'],
            reset_settle_steps=int(data['reset_settle_steps']),base_to_world=runtime.base_to_world,
            reset_forbidden_contacts=runtime.reset_forbidden_contacts,action_clipping_count=clipping,quaternion_fallback_count=fallbacks,
            gripper_projection_count=grip_projections,orientation_projection_count=orientation_projections,
            anchor_projection_count=anchor_projections,action_samples=contract.get('action_samples',1) if contract else 1,
            inference_calls=calls,ik_errors=ik_errors,first_observation_position_error_m=first_state_error,
            first_cloud_array_difference_m=first_cloud_error,control_dt_s=.25,physics_steps_per_frame=30,
            max_observation_position_difference_m=None if layout else float(np.linalg.norm(np.asarray(buffers['state'])[:,:3]-data['state'][:,:3],axis=1).max()),
            quality_limits=dict(xy_m=.005,height_difference_m=.05,height_error_m=.004,drift_m=.001,speed_m_s=.05))
        if runtime.reset_forbidden_contacts:
            result['success']=False;result['failure_reasons'].append('reset_contact')
        if goal_memory is not None:
            # Post-rollout diagnostic only; never consulted by prediction/control.
            goals=np.asarray(buffers['goal_features']);valid=np.flatnonzero(goals[:,6]>0)
            truth=(np.linalg.inv(runtime.base_to_world)@np.r_[initial[1]['position'],1.])[:3]
            result['goal_memory_audit']=dict(first_valid_frame=int(valid[0]) if len(valid) else None,
                measured_initial_goal_base=goals[valid[0],:3] if len(valid) else None,
                initial_position_error_m=float(np.linalg.norm(goals[valid[0],:3]-truth)) if len(valid) else None,
                recipe=contract['goal_memory']['recipe'],
                surface_methods=(getattr(goal_memory,'last_measurement',None) or {}).get('methods'),
                source='fixed RGB-D; comparison to simulator state is a post-rollout diagnostic only')
        if layout:
            result['layout_spec']=layout
            valid=np.asarray(buffers['depth_invalid_ratio'])<1
            if not valid.all():
                result['success']=False;result['failure_reasons'].append('empty_depth_view')
            for key in ['camera_1_rgb','camera_2_rgb']:
                if any(np.asarray(v).std()<2 for v in buffers[key]):
                    result['success']=False;result['failure_reasons'].append('blank_'+key)
        core.write_json(args.output/'result.json',result)
        saved={k:np.asarray(v) for k,v in buffers.items()}
        saved.update(success=np.asarray(result['success']),control_dt_s=np.asarray(.25),physics_steps_per_frame=np.asarray(30),
            reset_settle_steps=np.asarray(240),initial_ik_joint_position=data['initial_ik_joint_position'],reset_target=data['reset_target'],
            base_to_world=runtime.base_to_world,camera_intrinsics=sensors.intrinsics[:2],camera_to_world=sensors.static_extrinsics,
            camera_pose_usd_xyz_wxyz=sensors.static_usd_poses,wrist_camera_intrinsics=sensors.intrinsics[2],
            frame_index=np.arange(40,dtype=np.int32),is_first=np.arange(40)==0,
            history_valid=np.vstack([[0,1],np.ones((39,2),dtype=np.uint8)]).astype(np.uint8),
            cube_a_size_m=np.asarray(.04),cube_b_size_m=np.asarray(.06),scene_id=np.asarray(result['scene_id']),
            episode_id=np.asarray(layout['layout_id'] if layout else entry['episode_id']),seed=np.asarray(args.seed))
        np.savez_compressed(args.output/'rollout.npz',**saved)
        if layout:core.write_json(args.output/'scene.json',dict(scene,scene_id=layout['scene_id'],layout_spec=layout))
        frames[0].save(args.output/'rollout.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
        core.write_json(args.output/'completed.json',dict(frames=40,artifacts=['result.json','rollout.npz','rollout.gif']))
        print('RESULT',json.dumps({k:result[k] for k in ['success','failure_reasons','final_xy_error_m','final_z_error_m','ik_failures','action_clipping_count']},default=lambda x:x.item()),flush=True)
        completed=True
    except Exception as exc:
        (args.output/'error.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        raise
    finally:
        if connection:connection.close()
        # RGB/depth readbacks and our NPZ/GIF/JSON writes are synchronous and
        # complete above. No Replicator writer is attached to wait for here.
        app.close(wait_for_replicator=False,skip_cleanup=completed)

if __name__=='__main__':main()
