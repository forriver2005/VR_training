"""PhysX runtime. Import only after SimulationApp has started."""
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
import omni.physx
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from fr3_v2_core import *

class Runtime:
    def __init__(self, app, scene=None):
        self.app = app
        omni.usd.get_context().open_stage(str(scene or ROOT/'IsaacSim/FR3/FR3v2_cube40_cube60_v2.usda'))
        app.update()
        self.stage = omni.usd.get_context().get_stage()
        # Preserve the original eight-command OmniGraph interface.
        names = self.stage.GetPrimAtPath('/Graphs/Position_Controller/JointNameArray')
        names.CreateAttribute('inputs:input0',Sdf.ValueTypeNames.Token).Set('fr3v2_joint1')
        self.command_node = self.stage.GetPrimAtPath('/Graphs/Position_Controller/JointCommandArray')
        self.base_to_world = np.array(UsdGeom.Xformable(self.stage.GetPrimAtPath('/fr3v2')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        self.kin = Kinematics()
        self.contacts = set()
        from pxr import PhysicsSchemaTools
        def contact_callback(headers,data):
            for h in headers:
                pair=tuple(sorted([str(PhysicsSchemaTools.intToSdfPath(h.actor0)),str(PhysicsSchemaTools.intToSdfPath(h.actor1))]))
                self.contacts.add(pair)
        self.contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
        follower = self.stage.GetPrimAtPath('/fr3v2/joints/fr3v2_finger_joint2')
        follower.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,'rotY')
        # PhysX 5.1's inherited mimic did not close the passive finger reliably.
        # Both physical drives receive the same single-finger action in metres.
        follower.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,'rotX')
        drive=UsdPhysics.DriveAPI.Apply(follower,'linear')
        drive.CreateStiffnessAttr(2000.); drive.CreateDampingAttr(40.)
        drive.CreateMaxForceAttr(100.)
        UsdPhysics.PrismaticJoint(follower).GetLowerLimitAttr().Set(0.)
        UsdPhysics.PrismaticJoint(follower).GetUpperLimitAttr().Set(.04)
        for label,size in [('A',.04),('B',.06)]:
            prim = self.stage.GetPrimAtPath('/World/cube'+label)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            mass = 1000.*size**3
            UsdPhysics.MassAPI(prim).GetMassAttr().Set(mass)
            UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(np.ones(3)*mass*size**2/6)))
            UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Set(Gf.Vec3f(0))
            UsdPhysics.MassAPI(prim).GetPrincipalAxesAttr().Set(Gf.Quatf(1))
            UsdGeom.Xformable(prim).GetPrim().GetAttribute('xformOp:scale').Set(Gf.Vec3d(size))
        material = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim('/World/GraspMaterial','Material'))
        material.CreateStaticFrictionAttr(1.)
        material.CreateDynamicFrictionAttr(1.)
        material.CreateRestitutionAttr(0.)
        from pxr import UsdShade
        physics_material = UsdShade.Material(material.GetPrim())
        for path in ['/fr3v2/fr3v2_leftfinger','/fr3v2/fr3v2_rightfinger']:
            p=self.stage.GetPrimAtPath(path)
            if p.IsValid(): UsdShade.MaterialBindingAPI.Apply(p).Bind(physics_material,bindingStrength='strongerThanDescendants',materialPurpose='physics')
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/fr3v2/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if any(name in str(prim.GetPath()) for name in ['cubeA','cubeB','finger']):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_material, materialPurpose='physics')
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(.001)
                    collision.CreateRestOffsetAttr(0.)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(16)
                api.CreateSolverVelocityIterationCountAttr(4)
        self.sim = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.)
        self.sim.initialize_physics()
        self.sim.play()
        import omni.timeline as timeline_module
        self.timeline=timeline_module.get_timeline_interface()
        self.timeline.set_auto_update(False)
        self.robot = SingleArticulation('/fr3v2', reset_xform_properties=False)
        self.robot.initialize()
        self.hand = SingleRigidPrim('/fr3v2/fr3v2_hand', reset_xform_properties=False)
        self.hand.initialize()
        self.cubes = [SingleRigidPrim('/World/cube'+label,reset_xform_properties=False) for label in ['A','B']]
        for cube in self.cubes: cube.initialize()
        self.default_cube_positions=[np.array(c.get_world_pose()[0]).copy() for c in self.cubes]
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
        self.geometry_audit=[]
        for label,size in [('A',.04),('B',.06)]:
            prim=self.stage.GetPrimAtPath('/World/cube'+label)
            actual_size=np.array(bounds.ComputeUntransformedBound(prim).ComputeAlignedRange().GetSize())*size
            collisions=[str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
            if not np.allclose(actual_size,size,atol=1e-5) or not collisions:
                raise RuntimeError(f'Cube {label} visual/collision geometry invalid: {actual_size}, {collisions}')
            mass=UsdPhysics.MassAPI(prim)
            self.geometry_audit.append(dict(cube=label,local_geometry_size_m=actual_size,collision_prims=collisions,
                mass_kg=mass.GetMassAttr().Get(),diagonal_inertia_kg_m2=np.array(mass.GetDiagonalInertiaAttr().Get())))
        # Gains are explicit, force-limited drives; no runtime joint teleportation.
        self.robot.get_articulation_controller().set_gains(kps=np.array([20000.]*7+[2000.,2000.]), kds=np.array([400.]*7+[40.,40.]))
        self.q_target = None

    def command(self, q, grip):
        import omni.graph.core as og
        for i,value in enumerate(np.r_[q,grip]):
            attr = og.Controller.attribute(f'/Graphs/Position_Controller/JointCommandArray.inputs:input{i}')
            og.Controller.set(attr,float(value))
        self.robot.apply_action(ArticulationAction(joint_positions=np.r_[q,grip,grip],joint_indices=np.arange(9)))
        self.q_target = np.array(q)
        self.grip_target = float(grip)

    def apply_obstacles(self, obstacles):
        mapping={'box':'/World/box','box (1)':'/World/box1','box (2)':'/World/box2'}
        start_time=self.sim.current_time
        expected_poses={}
        for row in obstacles:
            path=mapping.get(row['name'])
            if path is None: raise ValueError('Unmapped Unity obstacle: '+row['name'])
            prim=self.stage.GetPrimAtPath(path)
            if not prim.IsValid():raise ValueError('Missing Isaac obstacle '+path)
            transform=UsdGeom.Xformable(prim)
            ops={op.GetOpName():op for op in transform.GetOrderedXformOps()}
            position=ops.get('xformOp:translate') or transform.AddTranslateOp()
            position.Set(Gf.Vec3d(*row['position']))
            quaternion=Rotation.from_quat(row['orientation_xyzw']).as_quat()
            orientation=ops.get('xformOp:orient') or transform.AddOrientOp()
            orientation.Set(Gf.Quatd(float(quaternion[3]),Gf.Vec3d(*quaternion[:3])))
            expected_poses[path]=(np.asarray(row['position'],dtype=float),
                                  Rotation.from_quat(row['orientation_xyzw']))
        # USD edits do not immediately move the already-created PhysX
        # colliders. Commit the stage and refresh transforms before the first
        # reset, otherwise episode 0 can use stale obstacle poses while later
        # episodes use the updated poses.
        self.app.update()
        omni.physx.get_physx_interface().update_transformations(False,True,True)
        if self.sim.current_time!=start_time:
            raise RuntimeError('Applying obstacle transforms advanced physics')
        for path,(expected_position,expected_rotation) in expected_poses.items():
            prim=self.stage.GetPrimAtPath(path)
            actual=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            position_error=float(np.linalg.norm(actual[:3,3]-expected_position))
            rotation=Rotation.from_matrix(actual[:3,:3])
            rotation_error=float((rotation.inv()*expected_rotation).magnitude())
            if position_error>.001 or rotation_error>np.deg2rad(.1):
                raise RuntimeError(f'OBSTACLE_RESET_MISMATCH {path}: position={position_error:.6f}m, '
                                   f'rotation={np.degrees(rotation_error):.4f}deg')

    def sync_rigid_transforms(self):
        """Commit rigid-body pose edits to PhysX before the next physics step."""
        start_time=self.sim.current_time
        self.app.update()
        omni.physx.get_physx_interface().update_transformations(False,True,True)
        if self.sim.current_time!=start_time:
            raise RuntimeError('Synchronizing rigid transforms advanced physics')

    def obstacle_poses(self):
        return {str(p.GetPath()):np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.tolist()
                for p in self.stage.Traverse() if str(p.GetPath()) in ['/World/box','/World/box1','/World/box2']}

    def step(self, n):
        start = self.sim.current_time
        if not self.sim.is_playing():
            raise RuntimeError(f'Cannot step physics: timeline is not playing at time {start}')
        for _ in range(n):
            self.command(self.q_target,self.grip_target)
            self.sim.step(render=False,update_fabric=True)
        if abs((self.sim.current_time-start)-n*PHYSICS_DT)>2e-6:
            raise RuntimeError(f'Unexpected physical time increment: start={start}, '
                               f'end={self.sim.current_time}, expected_delta={n*PHYSICS_DT}, '
                               f'steps={n}, physics_dt={self.sim.get_physics_dt()}, '
                               f'timeline_playing={self.sim.is_playing()}')

    def render(self, n=2, updates=0):
        import carb.settings
        import omni.replicator.core as rep
        start = self.sim.current_time
        was_playing=self.sim.is_playing()
        settings=carb.settings.get_settings()
        play_simulations=settings.get('/app/player/playSimulations')
        settings.set('/app/player/playSimulations',False)
        try:
            self.timeline.set_current_time(start)
            self.timeline.commit()
            # The physics-only step does not reliably refresh RTX transforms.
            # Match SimulationContext.render's force-update without an app step.
            if self.sim.physics_sim_view is not None:
                self.sim.physics_sim_view.update_articulations_kinematic()
            if omni.kit.app.get_app().get_extension_manager().is_extension_enabled('omni.physx.fabric'):
                from omni.physxfabric import get_physx_fabric_interface
                get_physx_fabric_interface().force_update(PHYSICS_DT,start)
            else:
                omni.physx.get_physx_interface().update_transformations(False,True,True)
            rep.orchestrator.step(delta_time=0.,rt_subframes=max(4,n),pause_timeline=False,wait_for_render=True)
            # Pump texture transfers and material updates at the same physics
            # pose. Never use wait_until_complete here: it stops generation.
            for _ in range(updates):
                self.app.update()
        finally:
            settings.set('/app/player/playSimulations',play_simulations)
        if self.sim.current_time != start: raise RuntimeError(f'Rendering advanced physics {start} -> {self.sim.current_time}')
        if self.sim.is_playing()!=was_playing:
            raise RuntimeError(f'Rendering changed timeline playing state {was_playing} -> {self.sim.is_playing()}')

    def cube_state(self, cube):
        p,wxyz = cube.get_world_pose()
        # Own every array: PhysX views may reuse their buffers on later steps.
        return dict(position=np.array(p,copy=True),orientation_xyzw=np.roll(wxyz,-1).copy(),
                    linear_velocity=np.array(cube.get_linear_velocity(),copy=True),
                    angular_velocity=np.array(cube.get_angular_velocity(),copy=True))

    def observation_state(self):
        p,wxyz = self.hand.get_world_pose()
        rotation = Rotation.from_quat(np.roll(wxyz,-1))
        tcp = p + rotation.apply(TCP_OFFSET)
        base_rotation = Rotation.from_matrix(self.base_to_world[:3,:3])
        xyz = base_rotation.inv().apply(tcp-self.base_to_world[:3,3])
        orientation = (base_rotation.inv()*rotation).as_quat()
        q = self.robot.get_joint_positions().copy()
        return np.r_[xyz,orientation,q[7]].astype(np.float32), q[:7]

    def reset(self, a_world=None, b_world=None, settle_steps=240, transport_path=None):
        # Snapshot geometry supplies support height; positions are never moved during actions.
        self.contacts.clear()
        self.kin.seed=np.array([0.,-.3,0.,-2.2,0.,2.,.7854])
        initial = [p.copy() for p in self.default_cube_positions]
        if a_world is not None: initial[0] = np.array(a_world)
        if b_world is not None: initial[1] = np.array(b_world)
        for cube,p in zip(self.cubes,initial):
            cube.set_world_pose(p,np.array([1.,0.,0.,0.]))
            cube.set_linear_velocity(np.zeros(3)); cube.set_angular_velocity(np.zeros(3))
        self.sync_rigid_transforms()
        a,b = [p-self.base_to_world[:3,3] for p in initial]
        reset,_,_ = compile_actions(a,b)
        q = self.kin.ik(reset)
        self.robot.set_joint_positions(np.r_[q,.04,.04])
        self.robot.set_joint_velocities(np.zeros(9))
        self.command(q,.04)
        # Check the PhysX state before the longer reset motion. Some invalid
        # rigid-body views accept USD edits without moving the simulated cube.
        for attempt in range(2):
            for cube,p in zip(self.cubes,initial):
                if not cube._prim_view.is_physics_handle_valid():
                    cube.initialize()
                if not cube._prim_view.is_physics_handle_valid():
                    raise RuntimeError('RESET_CUBE_PHYSICS_HANDLE_INVALID')
                cube.set_world_pose(p,np.array([1.,0.,0.,0.]))
                cube.set_linear_velocity(np.zeros(3));cube.set_angular_velocity(np.zeros(3))
            self.sync_rigid_transforms()
            self.step(1)
            measured_now=[self.cube_state(c) for c in self.cubes]
            errors=[float(np.linalg.norm(row['position']-target)) for row,target in zip(measured_now,initial)]
            if max(errors,default=0.)<=.005:
                break
            print(f'CUBE_RESET_RETRY {attempt+1}/2 position_errors_m={errors}',flush=True)
        else:
            raise RuntimeError(f'RESET_CUBE_POSE_MISMATCH position_errors_m={errors}')
        self.step(settle_steps)
        # Use actual settled cube height when calculating grasp/place targets.
        measured = [self.cube_state(c) for c in self.cubes]
        a,b = [c['position']-self.base_to_world[:3,3] for c in measured]
        reset,actions,phases = compile_actions(a,b,transport_path)
        q=self.kin.ik(reset)
        self.robot.set_joint_positions(np.r_[q,.04,.04])
        self.robot.set_joint_velocities(np.zeros(9))
        for cube in self.cubes:
            cube.set_linear_velocity(np.zeros(3));cube.set_angular_velocity(np.zeros(3))
        self.command(q,.04)
        self.sync_rigid_transforms()
        self.step(settle_steps)
        measured=[self.cube_state(c) for c in self.cubes]
        self.reset_contacts=sorted(self.contacts)
        self.allowed_support_contacts=object_support_contacts(self.reset_contacts)
        self.reset_forbidden_contacts=[pair for pair in self.reset_contacts
            if any('/fr3v2/' in p for p in pair)
            and any(p in ['/World/cubeA','/World/cubeB'] or '/World/box' in p for p in pair)]
        self.contacts.clear()
        return reset,q,actions,phases,measured

    def restore_replay_initial(self,states,q,settle_steps):
        # Restore the archived episode rather than recompile a new reset path.
        # An invalid PhysX handle otherwise falls back to editing only USD.
        self.robot.set_joint_positions(np.r_[q,.04,.04])
        self.robot.set_joint_velocities(np.zeros(9))
        self.command(q,.04)
        for cube,state in zip(self.cubes,states):
            cube.initialize()
            if not cube._prim_view.is_physics_handle_valid():
                raise RuntimeError('REPLAY_INITIAL_STATE_MISMATCH: invalid rigid-body physics handle')
            cube.set_world_pose(np.asarray(state['position']),np.roll(state['orientation_xyzw'],1))
            cube.set_linear_velocity(np.asarray(state['linear_velocity']))
            cube.set_angular_velocity(np.asarray(state['angular_velocity']))
        self.sync_rigid_transforms()
        self.contacts.clear()
        self.step(settle_steps)
        measured=[self.cube_state(c) for c in self.cubes]
        self.reset_contacts=sorted(self.contacts)
        self.allowed_support_contacts=object_support_contacts(self.reset_contacts)
        self.reset_forbidden_contacts=[pair for pair in self.reset_contacts
            if any('/fr3v2/' in p for p in pair)
            and any(p in ['/World/cubeA','/World/cubeB'] or '/World/box' in p for p in pair)]
        self.contacts.clear()
        return replay_initial_audit(states,measured)

    def evaluate(self, lifted, ik_failures, release_ok):
        samples=[]
        for i in range(RELEASE_SETTLE_STEPS):
            self.step(1)
            if i>=RELEASE_SETTLE_STEPS-RELEASE_STABLE_STEPS:
                samples.append([self.cube_state(c) for c in self.cubes])
        positions=np.array([[c['position'] for c in row] for row in samples])
        xy=np.linalg.norm(positions[:,0,:2]-positions[:,1,:2],axis=1)
        z=(positions[:,0,2]-positions[:,1,2])-.05
        drift=np.max(np.linalg.norm(positions-positions[0],axis=2),axis=0)
        final=samples[-1]
        speed=np.array([np.linalg.norm(c['linear_velocity']) for c in final])
        # The release-phase check only controls safe retreat. Acceptance uses
        # the final continuous window after the fingers have opened.
        release_precheck=release_ok
        velocities=np.array([[c['linear_velocity'] for c in row] for row in samples])
        release_audit=release_window_audit(positions,velocities,self.robot.get_joint_positions()[7:],
                                         ('/World/cubeA','/World/cubeB') in self.contacts)
        release_ok=release_audit['passed']
        failures=[]
        if not lifted: failures.append('not_lifted')
        if ik_failures: failures.append('ik_failure')
        if not release_ok: failures.append('release_not_stable')
        if np.any(xy>RELEASE_XY_TOLERANCE_M): failures.append('xy_error')
        if np.any(np.abs(z)>.004): failures.append('z_error')
        if np.any(drift>=.001): failures.append('position_drift')
        if np.any(speed>=RELEASE_SPEED_M_S): failures.append('linear_speed')
        if ('/World/cubeA','/World/cubeB') not in self.contacts: failures.append('no_stack_contact')
        if np.any(self.robot.get_joint_positions()[7:]<.035): failures.append('gripper_not_released')
        obstacle_contacts=forbidden_obstacle_contacts(self.contacts,getattr(self,'allowed_support_contacts',[]))
        if obstacle_contacts: failures.append('obstacle_collision')
        return dict(success=not failures,failure_reasons=failures,final_xy_error_m=xy[-1],
                    final_z_error_m=z[-1],stable_window_max_xy_error_m=xy.max(),
                    stable_window_max_z_error_m=np.abs(z).max(),stable_window_drift_m=drift,
                    final_cube_states=final,lifted=lifted,ik_failures=ik_failures,
                    release_condition_passed=release_ok,final_finger_positions_m=self.robot.get_joint_positions()[7:].copy(),
                    release_precheck_passed=release_precheck,release_policy=RELEASE_POLICY,
                    release_window_audit=release_audit,stability_linear_velocities_world=velocities,
                    reset_contact_pairs=getattr(self,'reset_contacts',[]),
                    allowed_support_contacts=getattr(self,'allowed_support_contacts',[]),obstacle_collision_pairs=obstacle_contacts,
                    stability_positions_world=positions,contact_pairs_observed=sorted(self.contacts),
                    final_tilt_deg=[float(np.degrees(np.arccos(np.clip(Rotation.from_quat(c['orientation_xyzw']).as_matrix()[2,2],-1,1)))) for c in final])


