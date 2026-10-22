"""No SimulationApp: strict delivery integrity and depth-cloud reproducibility."""
from fr3_v2_core import *
import argparse, os

def audit_physics(record, label, errors):
    positions=np.asarray(record.get('stability_positions_world',[]),dtype=float)
    if positions.shape!=(60,2,3) or not np.isfinite(positions).all():
        errors.append(label+' stability samples');return
    xy=np.linalg.norm(positions[:,0,:2]-positions[:,1,:2],axis=1)
    z=positions[:,0,2]-positions[:,1,2]-.05
    drift=np.max(np.linalg.norm(positions-positions[0],axis=2),axis=0)
    if np.any(xy>RELEASE_XY_TOLERANCE_M) or np.any(np.abs(z)>.004) or np.any(drift>=.001):errors.append(label+' physical stability')
    for key,value in [('final_xy_error_m',xy[-1]),('final_z_error_m',z[-1]),
                      ('stable_window_max_xy_error_m',xy.max()),('stable_window_max_z_error_m',np.abs(z).max()),
                      ('stable_window_drift_m',drift)]:
        if key not in record or not np.allclose(record[key],value,atol=1e-9,rtol=0):errors.append(label+' inconsistent '+key)
    states=record.get('final_cube_states',[])
    if len(states)!=2:errors.append(label+' final states');return
    for state in states:
        if np.linalg.norm(state['linear_velocity'])>=RELEASE_SPEED_M_S:errors.append(label+' final speed')
        if not np.isfinite(state['angular_velocity']).all():errors.append(label+' angular velocity')
    if not record.get('lifted') or record.get('ik_failures')!=0 or not record.get('release_condition_passed'):
        errors.append(label+' lift/IK/release')
    if record.get('failure_reasons') or not record.get('success'):errors.append(label+' success evidence')
    if ['/World/cubeA','/World/cubeB'] not in record.get('contact_pairs_observed',[]):errors.append(label+' stack contact')
    if record.get('reset_forbidden_contacts'):errors.append(label+' reset collision')
    if record.get('release_policy')==RELEASE_POLICY:
        audit=release_window_audit(positions,record.get('stability_linear_velocities_world',[]),
                                  record.get('final_finger_positions_m',[]),
                                  ['/World/cubeA','/World/cubeB'] in record.get('contact_pairs_observed',[]))
        if record.get('release_window_audit')!=audit or not audit['passed']:
            errors.append(label+' post-release stability window')
    if 'allowed_support_contacts' in record:
        allowed=object_support_contacts(record.get('reset_contact_pairs',[]))
        if record['allowed_support_contacts']!=[list(p) for p in allowed]:errors.append(label+' support contact evidence')
        if forbidden_obstacle_contacts(record.get('contact_pairs_observed',[]),allowed):errors.append(label+' obstacle collision')
    if 'initial_state_audit' in record:
        audit=record['initial_state_audit']
        if not replay_initial_audit(audit.get('expected',[]),audit.get('actual',[]))['passed']:
            errors.append(label+' replay initial scene mismatch')
    if min(record.get('final_finger_positions_m',[0,0]))<.035:errors.append(label+' fingers not released')
    audits=record.get('rgb_projection_audits',[])
    if len(audits)!=40 or not any(sum(a[i]['matched'] for a in audits)>=3 for i in range(2)):errors.append(label+' RGB pose audit')

def validate(dataset):
    dataset=filesystem_path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    calibration=json.loads((dataset/'meta/camera_calibration.json').read_text(encoding='utf-8'))
    config=json.loads((dataset/'meta/collection_config.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (dataset/'meta/scenes.jsonl').read_text(encoding='utf-8').splitlines()]
    errors=[];checked=0;cloud_error=0.
    if config.get('release_policy')==RELEASE_POLICY and (
            config.get('release_settle_steps')!=RELEASE_SETTLE_STEPS or
            config.get('release_stable_window_steps')!=RELEASE_STABLE_STEPS):
        errors.append('release window contract')
    geometry=json.loads((dataset/'meta/geometry_audit.json').read_text(encoding='utf-8'))
    for item,size in zip(geometry,[.04,.06]):
        if not np.allclose(item['local_geometry_size_m'],size,atol=1e-5) or not item['collision_prims']:errors.append('visual/collision geometry')
        expected_mass=1000*size**3
        if not np.isclose(item['mass_kg'],expected_mass,rtol=1e-6) or not np.allclose(item['diagonal_inertia_kg_m2'],expected_mass*size**2/6,atol=1e-10):errors.append('mass/inertia')
    checksum_names=set()
    for line in (dataset/'checksums.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        checksum_names.add(name)
        if not (dataset/name).is_file() or sha256(dataset/name)!=digest:errors.append('checksum '+name)
    actual_names={p.relative_to(dataset).as_posix() for p in delivery_files(dataset)
                  if p.name!='checksums.sha256'}
    if checksum_names!=actual_names:errors.append('checksum inventory incomplete')
    shapes=dict(state=(40,8),action=(40,8),joint_position=(40,7),joint_target=(40,7),camera_1_rgb=(40,256,256,3),
                camera_2_rgb=(40,256,256,3),depth=(40,2,256,256),point_cloud=(40,1024,3),camera_intrinsics=(2,3,3),
                camera_to_world=(2,4,4),camera_pose_usd_xyz_wxyz=(2,7),frame_index=(40,),is_first=(40,),history_valid=(40,2),
                observation_time_s=(40,),control_dt_s=(),success=(),wrist_depth=(40,256,256),wrist_camera_to_world=(40,4,4),
                wrist_camera_intrinsics=(3,3),base_to_world=(4,4),elapsed_same_target_s=(40,),reset_target=(8,),
                initial_ik_joint_position=(7,),reset_settle_steps=(),physics_steps_per_frame=(),cube_a_size_m=(),cube_b_size_m=(),
                scene_id=(),episode_id=(),trajectory_id=(),seed=())
    kin=Kinematics()
    for row in rows:
        diag=json.loads((dataset/row['diagnostics_path']).read_text(encoding='utf-8'))
        if not row['episode_path']:continue
        with np.load(dataset/row['episode_path'],allow_pickle=False) as d:
            for name,shape in shapes.items():
                if name not in d:errors.append(row['episode_id']+' missing '+name)
                elif d[name].shape!=shape:errors.append(row['episode_id']+' shape '+name)
            if any(name not in d for name in shapes):continue
            for name in ['camera_1_rgb','camera_2_rgb']:
                if d[name].dtype!=np.uint8 or np.any(d[name].std(axis=(1,2,3))<2):errors.append('RGB dtype/blank '+name)
            for name in ['depth','point_cloud','state','action','wrist_depth','joint_position','joint_target']:
                if d[name].dtype!=np.float32 or not np.all(np.isfinite(d[name])):errors.append('float dtype/nonfinite '+name)
            for name in ['camera_intrinsics','camera_to_world','wrist_camera_intrinsics','wrist_camera_to_world','base_to_world','observation_time_s']:
                if not np.issubdtype(d[name].dtype,np.floating) or not np.isfinite(d[name]).all():errors.append('calibration/time dtype '+name)
            if d['success'].dtype!=np.bool_:errors.append('success dtype')
            if d['is_first'].dtype!=np.bool_ or not np.issubdtype(d['frame_index'].dtype,np.integer) or d['history_valid'].dtype!=np.uint8:errors.append('index/history dtype')
            if not np.array_equal(d['frame_index'],np.arange(40)):errors.append('frame_index')
            if not np.array_equal(d['is_first'],np.arange(40)==0):errors.append('is_first')
            if not np.array_equal(d['history_valid'],np.vstack([[0,1],np.ones((39,2))])):errors.append('history_valid')
            if not np.allclose(np.diff(d['observation_time_s']),float(d['control_dt_s']),atol=2e-6,rtol=0):errors.append('timestamps')
            expected_elapsed=np.zeros(40,dtype=float)
            expected_elapsed[1]=float(d['control_dt_s'])
            for i in range(2,40):
                if np.allclose(d['action'][i-1],d['action'][i-2],atol=1e-7,rtol=0):
                    expected_elapsed[i]=expected_elapsed[i-1]+float(d['control_dt_s'])
                else:
                    expected_elapsed[i]=float(d['control_dt_s'])
            if not np.allclose(d['elapsed_same_target_s'],expected_elapsed,atol=2e-6,rtol=0):errors.append('online target timer history')
            if float(d['control_dt_s'])!=CONTROL_DT:errors.append('control contract')
            if int(d['physics_steps_per_frame'])!=STEPS_PER_ACTION:errors.append('physics step contract')
            if not np.isclose(d['cube_a_size_m'],.04) or not np.isclose(d['cube_b_size_m'],.06):errors.append('cube dimensions')
            if str(d['episode_id'])!=row['episode_id'] or str(d['scene_id'])!=row['scene_id']:errors.append('episode identity')
            for name in ['state','action']:
                if not np.allclose(np.linalg.norm(d[name][:,3:7],axis=1),1,atol=1e-5):errors.append('quaternion '+name)
                if np.any(d[name][:,7]<-.001) or np.any(d[name][:,7]>.041):errors.append('gripper metres '+name)
            if np.allclose(d['reset_target'],d['action'][0],atol=1e-6):errors.append('action0 duplicates reset')
            reset=d['reset_target'].astype(float)
            if not np.all(np.isfinite(reset)) or not np.isclose(reset[7],.04,atol=1e-6):errors.append('reset target contract')
            fk_reset=kin.fk(d['initial_ik_joint_position'])
            fk_quat=Rotation.from_matrix(fk_reset[:3,:3]).as_quat()
            if np.linalg.norm(fk_reset[:3,3]-reset[:3])>1e-4 or abs(float(np.dot(fk_quat,reset[3:7])))<1-1e-4:
                errors.append('reset target/initial IK disagreement')
            row_reset=row.get('reset_target',{})
            diag_reset=diag.get('reset_target',{})
            for saved in [row_reset,diag_reset]:
                if not saved or not np.allclose(saved.get('eef_xyz_xyzw',[]),reset[:7],atol=1e-6) or not np.isclose(saved.get('gripper_target_m',-1),reset[7],atol=1e-6):
                    errors.append('reset target metadata disagreement')
            for q,state in zip(d['joint_position'],d['state']):
                if np.linalg.norm(kin.fk(q)[:3,3]-state[:3])>1e-4:errors.append('measured joint/TCP disagreement');break
            for q,action in zip(d['joint_target'],d['action']):
                if np.linalg.norm(kin.fk(q)[:3,3]-action[:3])>.001:errors.append('target IK disagreement');break
            for name in ['camera_intrinsics','camera_to_world','camera_pose_usd_xyz_wxyz','wrist_camera_intrinsics','base_to_world']:
                if not np.allclose(d[name],calibration[name],atol=1e-7):errors.append('calibration '+name)
            for i in range(40):
                points,count=depth_to_cloud(d['wrist_depth'][i],d['wrist_camera_intrinsics'],d['wrist_camera_to_world'][i],d['base_to_world'])
                error=float(np.max(np.abs(points-d['point_cloud'][i])))
                cloud_error=max(cloud_error,error)
                if error>1e-6:errors.append(f'point cloud reconstruction frame {i}')
                partial_cloud_policy=str(d['point_cloud_quality_policy']) if 'point_cloud_quality_policy' in d else None
                if partial_cloud_policy and (partial_cloud_policy!=POINT_CLOUD_POLICY or config.get('point_cloud_quality_policy')!=POINT_CLOUD_POLICY):
                    errors.append('unsupported point cloud quality policy')
                minimum=1 if partial_cloud_policy==POINT_CLOUD_POLICY else 1024
                if row['success'] and count<minimum:errors.append('accepted insufficient cloud')
                if 'point_cloud_valid_count' not in d or int(d['point_cloud_valid_count'][i])!=count:
                    errors.append(f'point cloud count evidence frame {i}')
            if 'point_cloud_quality_policy' in d:
                if diag.get('point_cloud_quality_audit')!=cloud_quality(d['point_cloud_valid_count']):
                    errors.append('point cloud quality audit disagreement')
            if bool(d['success'])!=bool(diag['success']) or bool(d['success'])!=row['success']:errors.append('success inconsistent')
            if row['success']:
                if config.get('release_policy')==RELEASE_POLICY and diag.get('release_policy')!=RELEASE_POLICY:
                    errors.append(row['episode_id']+' release policy missing')
                audit_physics(diag,row['episode_id']+' collection',errors)
                if not diag.get('calibration_audit',{}).get('passed'):errors.append('calibration audit missing')
                if 'initial_scene_audit' in diag and not diag['initial_scene_audit'].get('passed',False):errors.append('initial scene audit failed')
                label=row['episode_id']
                replay_path=diag.get('independent_replay_result_path')
                replay=json.loads((dataset/replay_path).read_text(encoding='utf-8')) if replay_path and (dataset/replay_path).is_file() else {}
                if not replay:errors.append(label+' independent replay missing')
                elif not replay.get('success'):errors.append(label+' independent replay failed: '+','.join(replay.get('failure_reasons',[])))
                else:audit_physics(replay,row['episode_id']+' replay',errors)
                if replay and bool(diag.get('independent_replay_verified'))!=bool(replay.get('success')):
                    errors.append(label+' independent replay status inconsistent')
                replay_rgb=np.asarray(replay.get('rgb_frame_std',[]))
                if replay and (replay_rgb.shape!=(40,2) or not np.isfinite(replay_rgb).all() or np.any(replay_rgb<2)):
                    errors.append(label+' replay RGB missing/invalid/blank')
                if not (dataset/f"replays/{row['episode_id']}.gif").is_file():errors.append(label+' replay visualization missing')
            checked+=1
    splits=json.loads((dataset/'meta/splits.json').read_text(encoding='utf-8'))
    scene_split={}
    partitioned=[]
    for split in ['train','validation','test']:
        for episode in splits[split]:
            partitioned.append(episode)
            row=next(r for r in rows if r['episode_id']==episode)
            if row['scene_id'] in scene_split and scene_split[row['scene_id']]!=split:errors.append('scene leakage')
            scene_split[row['scene_id']]=split
    if len(partitioned)!=len(set(partitioned)) or set(partitioned)!={r['episode_id'] for r in rows if r['success']}:errors.append('split membership')
    accepted=sum(bool(r['success']) for r in rows)
    if manifest['total_trajectories']!=len(rows) or manifest['successful_trajectories']!=accepted or manifest['rejected_trajectories']!=len(rows)-accepted:errors.append('manifest counts')
    for name,digest in json.loads((dataset/'meta/source_hashes.json').read_text(encoding='utf-8')).items():
        source_path=dataset/name
        if not source_path.is_file() or sha256(source_path)!=digest:errors.append('source hash '+name)
    if not manifest['successful_trajectories']:errors.append('no accepted successful episode')
    result=dict(passed=not errors,episodes_checked=checked,max_cloud_reconstruction_error_m=cloud_error,errors=errors)
    write_json(dataset/'meta/integrity_validation.json',result)
    manifest['delivery_verified']=not errors
    manifest['status']='validated' if not errors else 'validation_failed'
    write_json(dataset/'manifest.json',manifest)
    from fr3_v2_delivery import update_checksums
    update_checksums(dataset)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return not errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path)
    raise SystemExit(0 if validate(p.parse_args().dataset) else 1)
