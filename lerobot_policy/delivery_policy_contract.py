"""Shared, simulator-independent preprocessing for the 4 Hz delivery policy."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CROP = np.array([[.25, -.5, 0.], [.85, .5, .6]])
NORMALIZATION = dict(low=[.15, -.55, 0., -1., -1., -1., -1., 0.],
                     high=[.95, .55, .95, 1., 1., 1., 1., .04],
                     quaternion='xyzw, no variance scaling',
                     source='fixed simulation task bounds; IK/physics independently checked; not hardware safety limits')
TIMER = dict(position_tolerance_m=.001, angle_tolerance_rad=float(np.deg2rad(1.)),
             gripper_tolerance_m=.0005, normalization_seconds=2.,
             comparison='first target of current hold, not a drifting previous target')
BOUNDED_TIMER = dict(TIMER,comparison='bounded_local_continuity',position_tolerance_m=.0025,
                     anchor_position_tolerance_m=.005,anchor_gripper_tolerance_m=.001)
POINT_RECIPE = 'wrist_raster4096_fps_index0_squared_distance_round12_v2'
FUSED_POINT_RECIPE = 'fixed1_256_fixed2_256_wrist512_raster4096_fps_round12_available_views_v1'
POINT_SOURCES = {'wrist_depth': POINT_RECIPE, 'fused_depth': FUSED_POINT_RECIPE}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load_entries(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows or len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate episode list')
    if any(r['control_dt_s'] != .25 for r in rows): raise ValueError('Only 4 Hz delivery protocol supported')
    return rows

class HoldTimer:
    def __init__(self, config=None):
        self.config = dict(TIMER if config is None else config)
        self.anchor = None
        self.previous = None
        self.elapsed = 0.

    def advance(self, executed_action, dt):
        a = np.asarray(executed_action, dtype=np.float64)
        same = False
        if self.anchor is not None:
            qa, qb = a[3:7], self.anchor[3:7]
            dot = abs(np.dot(qa, qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)))
            reference=self.previous if self.config['comparison']=='bounded_local_continuity' else self.anchor
            same = (np.linalg.norm(a[:3]-reference[:3]) <= self.config['position_tolerance_m']
                    and 2*np.arccos(np.clip(dot, 0., 1.)) <= self.config['angle_tolerance_rad']
                    and abs(a[7]-reference[7]) <= self.config['gripper_tolerance_m'])
            if self.config['comparison']=='bounded_local_continuity':
                same=bool(same and np.linalg.norm(a[:3]-self.anchor[:3])<=self.config['anchor_position_tolerance_m']
                          and abs(a[7]-self.anchor[7])<=self.config['anchor_gripper_tolerance_m'])
        if same: self.elapsed += dt
        else: self.anchor, self.elapsed = a.copy(), float(dt)
        self.previous=a.copy()

def timer_sequence(actions, dt, config=None):
    timer = HoldTimer(config); values = []
    for action in actions:
        values.append(timer.elapsed)
        timer.advance(action, dt)
    return np.asarray(values, dtype=np.float32)

def normalize_points(points, crop=None):
    bounds = CROP if crop is None else np.asarray(crop)
    return (2*(np.asarray(points, dtype=np.float32)-bounds[0])/(bounds[1]-bounds[0])-1).astype(np.float32)

def _depth_points(depth, intrinsics, camera_to_world, base_to_world):
    depth = np.asarray(depth); v,u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > .01) & (depth < 3.)
    z = depth[valid].astype(np.float64); k = np.asarray(intrinsics)
    cam = np.stack([(u[valid]+.5-k[0,2])*z/k[0,0], (v[valid]+.5-k[1,2])*z/k[1,1], z], axis=1)
    transform = np.linalg.inv(base_to_world) @ camera_to_world
    points = cam @ transform[:3,:3].T + transform[:3,3]
    points = points[np.all((points >= CROP[0]) & (points <= CROP[1]), axis=1)]
    return points

def _sample_points(points, count):
    n = len(points)
    if not n: raise ValueError('Empty cropped camera point cloud')
    if n > 4096: points = points[np.linspace(0,n-1,4096,dtype=np.int64)]
    if count<=0:raise ValueError('Point count must be positive')
    indices = np.zeros(min(count,len(points)),dtype=np.int64)
    distances = np.full(len(points),np.inf); index = 0
    for i in range(len(indices)):
        indices[i] = index
        distances = np.minimum(distances,np.sum((points-points[index])**2,axis=1))
        # Equal squared distances at 1e-12 m^2 choose the first raster candidate.
        index = int(np.argmax(np.round(distances,12)))
    sampled = points[indices]
    return sampled[np.arange(count)%len(sampled)].astype(np.float32)

def wrist_point_cloud(depth, intrinsics, camera_to_world, base_to_world, count=1024):
    """Versioned FPS with explicit equal-distance handling for offline and online."""
    points=_depth_points(depth,intrinsics,camera_to_world,base_to_world)
    return _sample_points(points,count),len(points)

def fused_view_quotas(counts):
    available=np.asarray(counts)>0
    if not available.any():raise ValueError('All fused depth views are empty')
    quotas=np.zeros(3,dtype=np.int64)
    fixed=np.flatnonzero(available[:2])
    if not len(fixed):quotas[2]=1024
    else:
        total=512 if available[2] else 1024
        quotas[fixed]=total//len(fixed)
        if available[2]:quotas[2]=512
    return quotas

def fused_point_cloud(depth, intrinsics, camera_to_world, wrist_depth, wrist_intrinsics,
                      wrist_camera_to_world, base_to_world):
    """Fixed per-view quotas, independent of object identities or true poses."""
    if np.asarray(depth).shape!=(2,256,256):raise ValueError('Expected two fixed depth images')
    views=[_depth_points(d,k,t,base_to_world) for d,k,t in
           [(depth[0],intrinsics[0],camera_to_world[0]),(depth[1],intrinsics[1],camera_to_world[1]),
            (wrist_depth,wrist_intrinsics,wrist_camera_to_world)]]
    counts=np.asarray([len(v) for v in views],dtype=np.int64)
    quotas=fused_view_quotas(counts)
    return np.concatenate([_sample_points(v,int(n)) for v,n in zip(views,quotas) if n],axis=0),counts

def load_episode(entry, timer_config=None, pointcloud_source='wrist_depth'):
    if pointcloud_source not in POINT_SOURCES:raise ValueError('Unknown point cloud source '+pointcloud_source)
    recipe=POINT_SOURCES[pointcloud_source]
    path = ROOT/entry['npz']
    if sha256(path) != entry['source_npz_sha256']: raise ValueError(f'Data hash changed: {path}')
    cache_dir=ROOT/('data/derived/delivery_pointcloud_v2' if pointcloud_source=='wrist_depth' else 'data/derived/delivery_pointcloud_fused_v1')
    cache_dir.mkdir(parents=True,exist_ok=True)
    cache=cache_dir/(entry['source_npz_sha256']+'_'+sha256(__file__)[:12]+'.npz')
    with np.load(path, allow_pickle=False) as z:
        data = {k:z[k] for k in ['state','action','point_cloud','observation_time_s','control_dt_s','success']}
        if cache.is_file():
            with np.load(cache) as saved:
                if str(saved['source_sha256'].item())!=entry['source_npz_sha256'] or str(saved['recipe'].item())!=recipe:
                    raise ValueError('Invalid derived cloud provenance')
                data['point_cloud']=saved['point_cloud']
        else:
            depth=z['wrist_depth'];k=z['wrist_camera_intrinsics'];poses=z['wrist_camera_to_world'];base=z['base_to_world']
            if pointcloud_source=='wrist_depth':
                rebuilt=np.stack([wrist_point_cloud(d,k,t,base)[0] for d,t in zip(depth,poses)])
            else:
                fixed=z['depth'];fixed_k=z['camera_intrinsics'];fixed_poses=z['camera_to_world']
                rebuilt=np.stack([fused_point_cloud(fd,fixed_k,fixed_poses,d,k,t,base)[0] for fd,d,t in zip(fixed,depth,poses)])
            np.savez_compressed(cache,point_cloud=rebuilt,source_sha256=entry['source_npz_sha256'],recipe=recipe)
            data['point_cloud']=rebuilt
    data['derived_point_cloud_path']=str(cache)
    if not bool(data['success']) or data['state'].shape != (40,8) or data['action'].shape != (40,8):
        raise ValueError('Expected one successful 40-frame episode')
    if float(data['control_dt_s']) != .25 or not np.allclose(np.diff(data['observation_time_s']),.25,atol=1e-6,rtol=0):
        raise ValueError('Control time mismatch')
    if data['point_cloud'].shape != (40,1024,3): raise ValueError('Point cloud shape mismatch')
    for k in ['state','action','point_cloud']:
        if not np.isfinite(data[k]).all(): raise ValueError('Non-finite '+k)
    a = data['action']
    if np.any(a < np.asarray(NORMALIZATION['low'])-1e-6) or np.any(a > np.asarray(NORMALIZATION['high'])+1e-6):
        raise ValueError('Expert action outside fixed new task bounds')
    data['elapsed_same_target_s'] = timer_sequence(a,.25,timer_config)
    return data
