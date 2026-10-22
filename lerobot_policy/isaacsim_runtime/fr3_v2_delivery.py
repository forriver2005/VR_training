"""Self-contained source snapshot, grouped metadata and checksums."""
from fr3_v2_core import *
import datetime, shutil, collections, os

def snapshot_sources(output,runtime):
    from pxr import UsdUtils,Sdf
    source=output/'source'
    layer=runtime.stage.Flatten()
    # Render products and annotator graphs are transient GPU runtime resources.
    # Edit the detached layer without opening a second live stage. Opening one
    # with the same camera paths can disturb Kit's active exposure/render state.
    graph_paths=[str(p.GetPath()) for p in runtime.stage.Traverse() if p.GetTypeName()=='OmniGraph' and not str(p.GetPath()).startswith('/Graphs/Position_Controller')]
    edit=Sdf.BatchNamespaceEdit()
    for path in ['/Render','/Replicator']+graph_paths:
        if layer.GetPrimAtPath(path):edit.Add(path,Sdf.Path.emptyPath)
    if not layer.Apply(edit):raise RuntimeError('Could not strip transient render products from snapshot')
    missing=[]
    copied={}
    def remap(asset):
        if not asset: return asset
        original=Path(asset)
        # MDL core modules are a documented dependency of the Isaac runtime.
        if asset in ['OmniPBR.mdl','OmniGlass.mdl']:
            copied[asset]='Isaac Sim built-in kit/mdl/core/Base/'+asset
            return asset
        if not filesystem_path(original).is_file():
            missing.append(asset); return asset
        digest=sha256(original)
        relative=f'assets/{digest[:16]}_{original.name}'
        target=source/relative
        filesystem_path(target.parent).mkdir(parents=True,exist_ok=True)
        if not filesystem_path(target).exists(): shutil.copy2(filesystem_path(original),filesystem_path(target))
        copied[asset]=relative
        return relative
    UsdUtils.ModifyAssetPaths(layer,remap)
    if missing: raise RuntimeError('Unresolved snapshot assets: '+str(missing))
    layer.Export(str(source/'scene.usda'))
    # Preserve the path expected by the copied simulator-independent IK code.
    urdf_dest=source/URDF.relative_to(ROOT)
    urdf_dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(URDF,urdf_dest)
    # Keep only meshes referenced by the FR3v2 URDF. Optional accessory
    # directories are not part of this arm and can be removed by Isaac's
    # asset resolver while the simulator is shutting down.
    mesh_root=URDF.parent/'meshes'
    for rel in ['robots/fr3v2','robot_ee/franka_hand_white']:
        shutil.copytree(filesystem_path(mesh_root/rel),filesystem_path(urdf_dest.parent/'meshes'/rel),dirs_exist_ok=True)
    for name in ['fr3_v2_core.py','fr3_v2_sim.py','fr3_v2_sensors.py','fr3_v2_delivery.py','collect_v2.py','replay_v2.py','run_replay_v2.py','validate_delivery_v2.py']:
        target=source/'Tools'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'Tools'/name,target)
    for name in ['Unity/Assets/New Scripts/TrajectoryGeneratorFromMaxPoint.cs','Unity/Assets/Editor/DemoGraspV2Validation.cs',
                 'Unity/Assets/XR/XRGeneralSettingsPerBuildTarget.asset','Environment/requirements_isaac.txt',
                 'Environment/requirements_unity.txt','Docs/IMPLEMENTATION_PLAN_20261010.md',
                 'Unity/Assets/New Scripts/RandomPlaceCubes.cs','Unity/Assets/panda.unity',
                 'Unity/Assets/Scripts/DemoGraspPicoStatus.cs',
                 'Unity/Assets/Editor/DemoGraspPicoBindingValidation.cs',
                 'Unity/Assets/Samples/XR Interaction Toolkit/2.6.4/Starter Assets/XRI Default Input Actions.inputactions',
                 'Docs/START_HERE_V2.md','Tools/prepare_pico_bindings.py','Docs/generate.txt','Docs/交付要求.txt',
                 'Tools/run_v2_pipeline.ps1','Tools/collect_latest_unity_v2.bat']:
        target=filesystem_path(source/name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(filesystem_path(ROOT/name),target)
    write_json(source/'asset_map.json',copied)
    hashes={p.relative_to(output).as_posix():sha256(p) for p in delivery_files(source)}
    write_json(output/'meta/source_hashes.json',hashes)
    return hashes

def update_checksums(output):
    paths=sorted(p for p in delivery_files(output) if p.name!='checksums.sha256')
    (output/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(output).as_posix()}\n' for p in paths),encoding='utf-8')

def finalize_delivery(output,rows,config,sources):
    with (output/'meta/scenes.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,(np.ndarray,np.generic)) else str(x),ensure_ascii=False)+'\n')
    groups=sorted(set(str(row['scene_id']) for row in rows))
    partitions=dict(train=[],validation=[],test=[])
    # Whole scenes stay together; a single-scene pilot is train only.
    for scene in groups:
        bucket=int(hashlib.sha256(scene.encode()).hexdigest()[:8],16)%10
        split='test' if bucket==9 else 'validation' if bucket==8 else 'train'
        if len(groups)<3: split='train'
        partitions[split].extend(r['episode_id'] for r in rows if r['scene_id']==scene and r['success'])
    write_json(output/'meta/splits.json',dict(grouping='scene_id; all trajectories and all history windows stay together',**partitions))
    diagnostics=[json.loads((output/r['diagnostics_path']).read_text(encoding='utf-8')) for r in rows]
    counts=collections.Counter(reason for d in diagnostics for reason in d.get('failure_reasons',[]))
    accepted=sum(r['success'] for r in rows)
    manifest=dict(data_version=VERSION,field_version=2,total_trajectories=len(rows),successful_trajectories=accepted,
                  rejected_trajectories=len(rows)-accepted,success_rate=accepted/len(rows) if rows else 0.,failure_categories=dict(counts),
                  cube_a_size_m=.04,cube_b_size_m=.06,trajectory_length=40,control_dt_s=CONTROL_DT,phases=PHASE_COUNTS,
                  generated_at=datetime.datetime.now().astimezone().isoformat(),source_hashes=sources,
                  independent_replay_verified=bool(accepted) and all(d.get('independent_replay_verified',False) for d in diagnostics if d['success']),
                  delivery_verified=False,status='requires independent replay and integrity validation')
    write_json(output/'manifest.json',manifest)
    (output/'README.txt').write_text(f'''{VERSION}\nGenerated: {manifest['generated_at']}\nThis is a newly simulated 40 mm / 60 mm task. Do not mix with old 20 mm / 25 mm data.\nCommand: {config['collection_command']}\nRuntime: Isaac Sim {config['isaac_sim_version']}; Python {config['python']}; GPU {config['gpu']}\nObservations precede each action. 40 real control cycles, 30 physics steps at 120 Hz per action; 4 Hz control.\nState is measured TCP XYZ and xyzw in the FR3 base frame, followed by measured single-finger joint1 displacement in metres.\nAction is absolute TCP XYZ/xyzw and single-finger target in metres; joint_target is the actual arm controller target.\nDepth is distance_to_image_plane in metres; RGB is uint8, 256x256. Fixed two-camera matrices use optical +X right/+Y down/+Z forward.\nPixel centres: u+0.5,v+0.5; see meta/camera_calibration.json.\npoint_cloud is 1024 XYZ points in FR3 base coordinates from wrist_depth ONLY. Reconstruct with wrist_camera_intrinsics and wrist_camera_to_world[t].\nBase transform, wrist depth and extrinsics are archived in every NPZ. Deterministic crop/FPS and padding are specified in collection_config.json.\nFrame zero has history_valid [0,1]; subsequent windows [1,1]. elapsed_same_target_s is an online-computable timer, not a phase label.\nPhysics acceptance: lifted >50mm, no IK failures, released and stable, XY within the 40 mm-on-60 mm support footprint (12mm Euclidean tolerance), delta-Z 50mm +/-4mm, both drifts <1mm and final speeds <80mm/s.\n120 post-action physics steps, last 60 samples all satisfy XY and Z limits. Diagnostics contain physical contact evidence, angular velocities and tilt.\nRejects remain in rejected/ and diagnostics/. Splits keep entire scenes together. A one-scene pilot has no validation/test data.\nsource/scene.usda is flattened, asset dependencies copied and remapped; URDF and runnable code are included.\nIndependent replay command (from any directory):\n  C:\\isaacsim\\python.bat "{output / 'source/Tools/replay_v2.py'}" "{output}" --episode episode_000000 --no-headless\nIntegrity command:\n  C:\\isaacsim\\python.bat "{output / 'source/Tools/validate_delivery_v2.py'}" "{output}"\nKnown limits: This NPZ contract extends DemoGrasp RGB/state modalities and is not automatically accepted by its unmodified LeRobot loader.\nPICO hardware input is validated separately after connecting the headset; a simulated expert pilot does not certify worn-headset teaching.\nA delivery is accepted only when manifest.delivery_verified and independent_replay_verified are both true.\n''',encoding='utf-8')
    with (output/'README.txt').open('a',encoding='utf-8') as stream:
        stream.write('\nRelease acceptance: allow natural settling after opening. After all actions, simulate 120 extra physics steps; the final 60 consecutive samples (0.5 s at 120 Hz) must satisfy XY, Z, drift and speed limits with open fingers and stack contact. release_precheck_passed is diagnostic only. See release_window_audit and stability_linear_velocities_world in diagnostics/replay reports.\n')
    update_checksums(output)
