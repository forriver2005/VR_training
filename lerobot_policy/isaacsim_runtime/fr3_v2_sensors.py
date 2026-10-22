"""Two fixed RGB-D views and one moving wrist depth view."""
from fr3_v2_sim import *
from isaacsim.sensors.camera import Camera
import carb.settings
import time

class Sensors:
    RGB_VALID_STD = 2.
    MAX_RENDER_READ_ATTEMPTS = 12
    RENDER_READ_TIMEOUT_S = 30.

    def __init__(self,runtime):
        self.runtime=runtime
        self.cameras=[]
        self._retired_render_products=[]
        self._render_product_recoveries=0
        # DLSS history at 256x256 can retain images from another render product.
        # Observations must depict the current sensor; disable temporal AA/blur.
        carb.settings.get_settings().set('/rtx/post/aa/op',0)
        carb.settings.get_settings().set('/rtx/post/motionblur/enabled',False)
        # Use one exposure scale, applied explicitly after camera initialization.
        # Global tonemap callbacks also modify per-camera exposure attributes.
        # Fixed D435 poses from DemoGrasp tasks/task/grasp.yaml, relative to FR3 base.
        fixed_extrinsics_base=np.array([
            [[-.95458387,-.22931432,.19022242,.25351984],
             [-.29675754,.78866419,-.53846426,.47084633],
             [-.02654405,-.57045923,-.82089687,.62032344],
             [0.,0.,0.,1.]],
            [[.6464361,.58084152,-.49471557,.89060788],
             [.76291282,-.4842839,.42829095,-.44560813],
             [.00918638,-.65428758,-.75619004,.49017396],
             [0.,0.,0.,1.]],
        ],dtype=np.float64)
        # Recreate sensor prims: Kit serializes tonemap into camera responsivity,
        # which would otherwise be multiplied again when loading a snapshot.
        for path in ['/World/CameraV2_1','/World/CameraV2_2','/fr3v2/fr3v2_hand/WristDepthV2']:
            runtime.stage.RemovePrim(path)
        settings=carb.settings.get_settings()
        start=runtime.sim.current_time
        settings.set('/app/player/playSimulations',False)
        runtime.app.update()  # Commit camera removal before reusing their prim paths.
        settings.set('/app/player/playSimulations',True)
        if runtime.sim.current_time!=start:raise RuntimeError('Camera initialization advanced physics')
        for i,extrinsics_base in enumerate(fixed_extrinsics_base):
            optical_to_world=runtime.base_to_world@extrinsics_base
            position=optical_to_world[:3,3]
            usd=optical_to_world[:3,:3]@np.diag([1.,-1.,-1.])
            camera=Camera(f'/World/CameraV2_{i+1}',name=f'camera_v2_{i+1}',frequency=-1,resolution=(256,256))
            quat=Rotation.from_matrix(usd).as_quat()
            camera.set_world_pose(position,np.roll(quat,1),camera_axes='usd')
            self.cameras.append(camera)
        wrist=Camera('/fr3v2/fr3v2_hand/WristDepthV2',name='wrist_depth_v2',frequency=-1,resolution=(256,256))
        # Mounted sideways off the hand, looking along the hand's +Z towards TCP.
        eye=np.array([.10,0.,.02]); target=np.array([0.,0.,.25])
        z=(target-eye)/np.linalg.norm(target-eye)
        x=np.cross(z,[0.,1.,0.]); x/=np.linalg.norm(x)
        y=np.cross(z,x)
        usd=np.column_stack([x,y,z])@np.diag([1.,-1.,-1.])
        wrist.set_local_pose(eye,np.roll(Rotation.from_matrix(usd).as_quat(),1),camera_axes='usd')
        self.cameras.append(wrist)
        for camera in self.cameras:
            # Physical pinhole model in USD tenths-of-unit; square pixels.
            camera.prim.GetAttribute('focalLength').Set(18.)
            camera.prim.GetAttribute('horizontalAperture').Set(24.)
            camera.prim.GetAttribute('verticalAperture').Set(24.)
            camera.prim.GetAttribute('clippingRange').Set(Gf.Vec2f(.01,3.))
            camera.prim.ApplyAPI('OmniRtxCameraExposureAPI_1')
            for name,value in {'exposure':0.,'exposure:fStop':1.,'exposure:iso':100.,'exposure:responsivity':1.,'exposure:time':1.}.items():
                camera.prim.CreateAttribute(name,Sdf.ValueTypeNames.Float).Set(value)
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
        # Trigger the setting callback even when a loaded scene/cache already
        # uses this scale; otherwise cached and newly-created camera states differ.
        settings.set('/rtx/post/tonemap/cm2Factor',1.)
        settings.set('/rtx/post/tonemap/cm2Factor',.0001)
        runtime.render(12)
        self.intrinsics=np.array([np.array(c.get_intrinsics_matrix()) for c in self.cameras],dtype=np.float64)
        self.static_extrinsics=np.array([self.pose(c)[0] for c in self.cameras[:2]])
        self.static_usd_poses=np.array([self.pose(c)[1] for c in self.cameras[:2]])

    def warmup_after_reset(self):
        """Refresh camera render products after a scene reset without advancing physics."""
        start=self.runtime.sim.current_time
        for camera,expected in zip(self.cameras[:2],self.static_usd_poses):
            actual=self.pose(camera)[1]
            if not np.allclose(actual,expected,atol=1e-6,rtol=0):
                camera.set_world_pose(expected[:3],expected[3:],camera_axes='usd')
                if not np.allclose(self.pose(camera)[1],expected,atol=1e-6,rtol=0):
                    raise RuntimeError(f'FIXED_CAMERA_POSE_MISMATCH {camera.prim_path}')
        self.runtime.render(16,updates=8)
        if self.runtime.sim.current_time!=start:
            raise RuntimeError('Camera warmup advanced physics')

    def pose(self,camera):
        # USD transform matrices are row-vector matrices; NPZ uses column vectors.
        usd=np.array(UsdGeom.Xformable(camera.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        optical=usd@np.diag([1.,-1.,-1.,1.])
        quat=Rotation.from_matrix(usd[:3,:3]).as_quat()
        return optical,np.r_[usd[:3,3],np.roll(quat,1)]

    def rebuild_render_products(self):
        import omni.replicator.core as rep
        start=self.runtime.sim.current_time
        poses=[self.pose(c)[0].copy() for c in self.cameras]
        self._render_product_recoveries+=1
        for index,camera in enumerate(self.cameras):
            old=camera._render_product
            old.hydra_texture.set_updates_enabled(False)
            # SDK 5.1's recursive detach can invalidate shared SDG nodes.
            # Retire the disabled product until shutdown instead of deleting
            # its graph. Recovery is bounded to two replacements per process.
            self._retired_render_products.append((old,camera._custom_annotators,camera._fabric_time_annotator))
            camera._custom_annotators={}
            camera._render_product=rep.create.render_product(camera.prim_path,(256,256),force_new=True,
                name=f'V2Recovery_{self._render_product_recoveries}_{index}')
            camera._render_product_path=camera._render_product.path
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
        self.runtime.render(16,updates=8)
        if self.runtime.sim.current_time!=start:
            raise RuntimeError('Render product recovery advanced physics')
        for camera,pose,intrinsics in zip(self.cameras,poses,self.intrinsics):
            if not np.allclose(self.pose(camera)[0],pose,atol=1e-9) or not np.allclose(camera.get_intrinsics_matrix(),intrinsics,atol=1e-9):
                raise RuntimeError('Render product recovery changed camera calibration')
        print('RGB_CAPTURE_REBUILD at unchanged physics time',start,flush=True)

    def capture(self):
        last_stats=None
        started=time.monotonic()
        history=[]
        for attempt in range(self.MAX_RENDER_READ_ATTEMPTS):
            # Runtime.render waits for this render's annotators without stopping
            # Replicator or advancing physics. Retry the entire RGB-D sample.
            if attempt:
                time.sleep(.1)
            self.runtime.render(3 if attempt==0 else 16,updates=0 if attempt==0 else 8)
            rgb=[]; depths=[]; invalid=[]
            for index,camera in enumerate(self.cameras):
                image=np.asarray(camera.get_rgb())
                depth=np.asarray(camera.get_depth())
                if image.shape!=(256,256,3) or depth.shape!=(256,256):
                    invalid.append(dict(camera=index+1,rgb_shape=image.shape,depth_shape=depth.shape))
                    continue
                if not np.isfinite(image).all():
                    invalid.append(dict(camera=index+1,reason='nonfinite_rgb'))
                    continue
                image=np.array(image,dtype=np.uint8,copy=True)
                depth=np.array(depth,dtype=np.float32,copy=True)
                depth[~np.isfinite(depth)]=0.
                rgb.append(image); depths.append(depth)
                if index<2 and float(image.std())<self.RGB_VALID_STD:
                    invalid.append(dict(camera=index+1,reason='blank_rgb',std=float(image.std()),
                                        mean=float(image.mean()),depth_valid_pixels=int(np.count_nonzero((depth>.01)&(depth<3.)))))
            if len(rgb)==len(depths)==len(self.cameras):
                stream_audit=camera_stream_duplicate_audit(rgb[0],rgb[1],depths[0],depths[1])
                if stream_audit.get('duplicate'):
                    invalid.append(dict(reason='duplicate_fixed_camera_stream',**stream_audit))
            if not invalid:
                break
            last_stats=invalid
            history.append(dict(attempt=attempt+1,elapsed_s=time.monotonic()-started,invalid=invalid))
            print(f'RGB_CAPTURE_RETRY {attempt+1}/{self.MAX_RENDER_READ_ATTEMPTS}: {invalid}',flush=True)
            if time.monotonic()-started>=self.RENDER_READ_TIMEOUT_S:
                break
            if attempt in (3,7) and self._render_product_recoveries<2:
                self.rebuild_render_products()
        if invalid:
            self.last_capture_failure=dict(physics_time_s=self.runtime.sim.current_time,retries=history,
                cameras=[dict(path=c.prim_path,render_product=c.get_render_product_path(),
                    exposure={name:c.prim.GetAttribute(name).Get() for name in
                              ['exposure','exposure:fStop','exposure:iso','exposure:responsivity','exposure:time']})
                         for c in self.cameras])
            print('RGB_CAPTURE_FAILED',self.last_capture_failure,flush=True)
            raise RuntimeError(f'RGB render did not become valid after {len(history)} attempts '
                               f'({time.monotonic()-started:.1f}s): {last_stats}')
        wrist_pose,wrist_usd=self.pose(self.cameras[2])
        cloud,count=depth_to_cloud(depths[2],self.intrinsics[2],wrist_pose,self.runtime.base_to_world)
        return dict(camera_1_rgb=rgb[0],camera_2_rgb=rgb[1],depth=np.stack(depths[:2]),
                    wrist_depth=depths[2],wrist_camera_to_world=wrist_pose,
                    wrist_camera_pose_usd_xyz_wxyz=wrist_usd,point_cloud=cloud,
                    point_cloud_valid_count=count,
                    depth_invalid_ratio=np.array([np.mean((d<=.01)|(d>=3.)) for d in depths]))

    def calibration(self):
        return dict(resolution=[256,256],camera_intrinsics=self.intrinsics[:2],
                    camera_to_world=self.static_extrinsics,camera_pose_usd_xyz_wxyz=self.static_usd_poses,
                    wrist_camera_intrinsics=self.intrinsics[2],wrist_extrinsics='per observation in NPZ',
                    depth_definition='distance_to_image_plane metres',distortion='none (pinhole)',
                    pixel_centres='(u+0.5,v+0.5); principal point (128,128)',
                    optical_axes='+X right, +Y down, +Z forward',transform_direction='optical to world',
                    base_to_world=self.runtime.base_to_world,point_cloud_source='wrist_depth')

    def audit_rgb_projection(self,capture):
        # Validation only: compare measured red cube pose with its RGB projection.
        # This geometry is never used to construct policy observations or clouds.
        p=self.runtime.cubes[0].get_world_pose()[0]
        rows=[]
        for i,camera in enumerate(self.cameras[:2]):
            uv=np.asarray(camera.get_image_coords_from_world_points(np.array([p])))[0]
            rgb=capture[f'camera_{i+1}_rgb'].astype(np.float32)
            red=(rgb[:,:,0]>60)&(rgb[:,:,0]>rgb[:,:,1]*1.5)&(rgb[:,:,0]>rgb[:,:,2]*1.5)
            v,u=np.where(red)
            distances=np.linalg.norm(np.stack([u+.5,v+.5],axis=1)-uv,axis=1) if len(u) else np.array([])
            closest=float(distances.min()) if len(distances) else None
            rows.append(dict(camera=i+1,actual_cube_world=p,projected_uv=uv,red_pixels=len(u),
                             nearest_red_pixel_distance=closest,matched=closest is not None and closest<12))
        return rows

    def audit_calibration(self,capture):
        result=[]
        for i,camera in enumerate(self.cameras[:2]):
            optical=self.static_extrinsics[i]
            inverse=np.linalg.inv(optical)
            points=[]
            for cube,size in zip(self.runtime.cubes,[.04,.06]):
                p,q=cube.get_world_pose()
                rotation=Rotation.from_quat(np.roll(q,-1))
                for axis in np.eye(3):
                    for sign in [-1,1]:points.append(p+rotation.apply(axis*sign*size/2))
            points=np.array(points)
            xyz=points@inverse[:3,:3].T+inverse[:3,3]
            uv=xyz[:,:2]/xyz[:,2,None]
            uv=uv*np.array([self.intrinsics[i,0,0],self.intrinsics[i,1,1]])+self.intrinsics[i,:2,2]
            sdk_uv=np.asarray(camera.get_image_coords_from_world_points(points))
            difference=float(np.max(np.abs(uv-sdk_uv)))
            visible_errors=[]
            for pixel,p in zip(uv,xyz):
                u,v=np.floor(pixel).astype(int)
                if 0<=u<256 and 0<=v<256 and p[2]>.01:
                    error=abs(float(capture['depth'][i,v,u])-p[2])
                    if error<.003:visible_errors.append(error)
            result.append(dict(camera=i+1,max_projection_difference_pixels=difference,
                               visible_cube_surface_samples=len(visible_errors),max_visible_surface_depth_error_m=max(visible_errors,default=None)))
        # Visibility is evaluated across collection frames. An obstacle may
        # occlude a cube in the initial frame without invalidating calibration.
        return dict(passed=all(r['max_projection_difference_pixels']<.001 and
                               r['visible_cube_surface_samples']>0 for r in result), cameras=result)
