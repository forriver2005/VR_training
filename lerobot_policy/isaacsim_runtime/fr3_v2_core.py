"""Versioned, simulator-independent data contract and deterministic geometry."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from ikpy.chain import Chain

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'RobotDescriptions/franka_description-main/franka_description/fr3v2_franka_hand.urdf'
PHYSICS_DT = 1 / 120
# Hold each Cartesian target for 30 physics ticks (4 Hz control at 120 Hz).
# The previous 15-tick/8 Hz schedule left the cubes moving at release.
STEPS_PER_ACTION = 30
CONTROL_DT = PHYSICS_DT * STEPS_PER_ACTION
# Shared by collection, replay, and strict delivery validation.
RELEASE_SPEED_M_S = .08
CUBE_A_SIZE_M = .04
CUBE_B_SIZE_M = .06
# A 40 mm cube can overhang a 60 mm support by 10 mm in the XY plane.
# Keep a small numerical/settling margin while still requiring stack contact.
RELEASE_XY_TOLERANCE_M = (CUBE_B_SIZE_M-CUBE_A_SIZE_M)/2 + .002
POINT_CLOUD_POLICY = 'nonempty_cyclic_padding_v1'
VERSION = 'demograsp_fr3v2_cube40_cube60_v2'
BASE_WORLD = np.array([-.615, 0., 0.])
TCP_OFFSET = np.array([0., 0., .1034])
DOWN_XYZW = np.array([1., 0., 0., 0.])
CROP = np.array([[.25, -.5, 0.], [.85, .5, .6]])
PHASE_COUNTS = dict(approach=6, pregrasp_hold=1, descend=4, close=3,
                   grip_hold=1, lift=5, transport=7, place=4,
                   place_hold=2, release=3, retreat=4)
RELEASE_WINDOW_FRAMES = PHASE_COUNTS['release']
RELEASE_POLICY = 'post_release_stable_window_v1'
RELEASE_SETTLE_STEPS = 120
RELEASE_STABLE_STEPS = 60

def release_window_audit(positions,velocities,finger_positions,stack_contact):
    positions=np.asarray(positions,dtype=float)
    velocities=np.asarray(velocities,dtype=float)
    fingers=np.asarray(finger_positions,dtype=float)
    valid=(positions.shape==(RELEASE_STABLE_STEPS,2,3) and
           velocities.shape==positions.shape and fingers.shape==(2,) and
           np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(fingers).all())
    if not valid:return dict(policy=RELEASE_POLICY,passed=False,reason='invalid_window_samples')
    xy=np.linalg.norm(positions[:,0,:2]-positions[:,1,:2],axis=1)
    z=np.abs(positions[:,0,2]-positions[:,1,2]-.05)
    drift=np.max(np.linalg.norm(positions-positions[0],axis=2),axis=0)
    speeds=np.linalg.norm(velocities,axis=2)
    passed=bool(np.all(xy<=RELEASE_XY_TOLERANCE_M) and np.all(z<=.004) and np.all(drift<.001) and
                np.all(speeds<RELEASE_SPEED_M_S) and np.all(fingers>=.035) and stack_contact)
    return dict(policy=RELEASE_POLICY,passed=passed,samples=RELEASE_STABLE_STEPS,
                duration_s=RELEASE_STABLE_STEPS*PHYSICS_DT,max_xy_error_m=float(xy.max()),
                max_z_error_m=float(z.max()),max_drift_m=drift.tolist(),
                max_speed_m_s=speeds.max(axis=0).tolist(),fingers_open=bool(np.all(fingers>=.035)),
                stack_contact=bool(stack_contact))

def cloud_quality(counts):
    counts=np.asarray(counts)
    return dict(policy=POINT_CLOUD_POLICY,passed=bool(counts.size and np.all(counts>0)),
                empty_frames=np.flatnonzero(counts<=0).tolist(),
                sparse_frames=np.flatnonzero((counts>0)&(counts<1024)).tolist(),
                minimum_count=int(counts.min()) if counts.size else 0)

def object_support_contacts(contacts):
    return sorted(tuple(sorted(pair)) for pair in contacts
                  if '/World/cubeA' in pair and any(p.startswith('/World/box') for p in pair))

def forbidden_obstacle_contacts(contacts,allowed_support):
    allowed={tuple(sorted(pair)) for pair in allowed_support}
    return sorted(tuple(sorted(pair)) for pair in contacts
                  if any(p.startswith('/World/box') for p in pair)
                  and any(p.startswith('/fr3v2/') or p=='/World/cubeA' for p in pair)
                  and tuple(sorted(pair)) not in allowed)

def replay_initial_audit(expected,actual):
    position_errors=[float(np.linalg.norm(np.asarray(a['position'])-e['position'])) for e,a in zip(expected,actual)]
    angle_errors=[float(2*np.arccos(np.clip(abs(np.dot(e['orientation_xyzw'],a['orientation_xyzw'])),0,1)))
                  for e,a in zip(expected,actual)]
    passed=len(expected)==len(actual)==2 and max(position_errors,default=float('inf'))<=.001 and max(angle_errors,default=float('inf'))<=np.deg2rad(1.)
    return dict(passed=bool(passed),expected=expected,actual=actual,
                position_errors_m=position_errors,orientation_errors_rad=angle_errors)

def camera_stream_duplicate_audit(rgb_a,rgb_b,depth_a,depth_b):
    rgb_a=np.asarray(rgb_a)
    rgb_b=np.asarray(rgb_b)
    depth_a=np.asarray(depth_a)
    depth_b=np.asarray(depth_b)
    if rgb_a.shape!=rgb_b.shape or depth_a.shape!=depth_b.shape or rgb_a.shape[:2]!=depth_a.shape:
        return dict(duplicate=False,valid=False,reason='shape_mismatch')
    rgb_close_fraction=float(np.mean(np.max(np.abs(rgb_a.astype(np.int16)-rgb_b.astype(np.int16)),axis=-1)<=1))
    depth_equal_fraction=float(np.mean(np.isclose(depth_a,depth_b,atol=1e-7,rtol=0)))
    duplicate=rgb_close_fraction>=.90 and depth_equal_fraction>=.95
    return dict(duplicate=bool(duplicate),valid=True,rgb_close_fraction=rgb_close_fraction,
                depth_equal_fraction=depth_equal_fraction)

def filesystem_path(path):
    # Python's normal Windows file APIs are limited by MAX_PATH for some
    # installations.  Delivery snapshots contain nested Unity package paths
    # that can exceed that limit, so add the extended-length prefix before
    # opening absolute Windows paths.  UNC paths need the UNC form of the
    # prefix; drive-letter paths use the regular form.
    file_path = os.fspath(path)
    if os.name == 'nt':
        file_path = os.path.abspath(file_path)
        if not file_path.startswith('\\\\?\\'):
            if file_path.startswith('\\\\'):
                file_path = '\\\\?\\UNC\\' + file_path[2:]
            else:
                file_path = '\\\\?\\' + file_path
    return Path(file_path)

def delivery_files(output):
    """Walk using extended Windows paths, returning logical delivery paths."""
    logical=Path(output)
    physical=filesystem_path(logical)
    def fail(error):
        raise error
    for root,dirs,files in os.walk(physical,onerror=fail):
        dirs[:]=[d for d in dirs if d!='__pycache__']
        relative=Path(root).relative_to(physical)
        for name in files:
            yield logical/relative/name

def sha256(path):
    with open(filesystem_path(path), 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, data):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding='utf-8')

class Kinematics:
    def __init__(self):
        self.chain = Chain.from_urdf_file(str(URDF), base_elements=['base'])
        self.indices = [i for i,l in enumerate(self.chain.links) if l.joint_type == 'revolute']
        assert len(self.indices) == 7
        self.chain.active_links_mask = np.array([i in self.indices for i in range(len(self.chain.links))])
        self.seed = np.array([0., -.3, 0., -2.2, 0., 2., .7854])

    def fk(self, q):
        full = np.zeros(len(self.chain.links))
        full[self.indices] = q
        return self.chain.forward_kinematics(full)

    def ik(self, action):
        full = np.zeros(len(self.chain.links))
        full[self.indices] = self.seed
        target = np.eye(4)
        target[:3, :3] = Rotation.from_quat(action[3:7]).as_matrix()
        target[:3, 3] = action[:3]
        solution = self.chain.inverse_kinematics_frame(target, initial_position=full, orientation_mode='all')
        actual = self.chain.forward_kinematics(solution)
        error = np.linalg.norm(actual[:3,3] - target[:3,3])
        angle = Rotation.from_matrix(actual[:3,:3].T @ target[:3,:3]).magnitude()
        q = solution[self.indices]
        margin = min(min(q[j]-self.chain.links[i].bounds[0], self.chain.links[i].bounds[1]-q[j]) for j,i in enumerate(self.indices))
        if error > .001 or angle > .01 or margin < .02:
            raise ValueError(f'IK invalid: position={error}, angle={angle}, limit_margin={margin}')
        self.seed = q.copy()
        return q

def compile_actions(a, b, transport_path=None):
    """Compile control phases before simulation; never resample saved observations."""
    a, b = np.array(a), np.array(b)
    grasp = a.copy()
    place = b + [0.,0.,.050]
    pre = grasp + [0.,0.,.16]
    lifted = grasp + [0.,0.,.20]
    over = place + [0.,0.,.15]
    reset = pre + [-.025, .025, .02]
    rows, phases = [], []
    previous = reset.copy()
    def segment(name, end, grip, n):
        nonlocal previous
        for i in range(1,n+1):
            t = i / n
            # Smooth endpoints; each entry is a real absolute command.
            t = t*t*(3-2*t)
            rows.append(np.r_[previous + t*(end-previous), DOWN_XYZW, grip])
            phases.append(name)
        previous = np.array(end).copy()
    segment('approach', pre, .04, 6)
    segment('pregrasp_hold', pre, .04, 1)
    segment('descend', grasp, .04, 4)
    segment('close', grasp, .016, 3)
    segment('grip_hold', grasp, .016, 1)
    segment('lift', lifted, .016, 5)
    if transport_path is None:
        segment('transport', over, .016, 7)
    else:
        # Unity waypoints are base XYZ, interpreted as transport geometry only.
        path = np.vstack([lifted, np.asarray(transport_path).reshape(-1,3), over])
        path[:,2] = np.maximum(path[:,2], max(lifted[2],over[2]))
        # End at the actual placement approach height without an implicit jump.
        path[-1]=over
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        if distance[-1] <= 1e-6: raise ValueError('Degenerate transport path')
        for t in np.linspace(0,distance[-1],8)[1:]:
            p = [np.interp(t,distance,path[:,j]) for j in range(3)]
            rows.append(np.r_[p,DOWN_XYZW,.016]); phases.append('transport')
        previous = over.copy()
    segment('place', place, .016, 4)
    segment('place_hold', place, .016, 2)
    segment('release', place, .04, 3)
    segment('retreat', over, .04, 4)
    assert len(rows) == 40
    return np.r_[reset,DOWN_XYZW,.04], np.array(rows), phases

def depth_to_cloud(depth, intrinsics, optical_to_world, base_to_world, count=1024):
    """Optical +Z forward/+Y down, pixel centres (u+.5,v+.5), deterministic FPS."""
    depth = np.asarray(depth)
    v,u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > .01) & (depth < 3.)
    z = depth[valid].astype(np.float64)
    optical = np.stack([(u[valid]+.5-intrinsics[0,2])*z/intrinsics[0,0],
                        (v[valid]+.5-intrinsics[1,2])*z/intrinsics[1,1], z],axis=1)
    transform = np.linalg.inv(base_to_world) @ optical_to_world
    points = optical @ transform[:3,:3].T + transform[:3,3]
    points = points[np.all((points>=CROP[0]) & (points<=CROP[1]),axis=1)]
    n = len(points)
    # Fixed, documented raster-order candidate reduction before FPS.
    if n > 4096: points = points[np.linspace(0,n-1,4096,dtype=np.int64)]
    if not len(points): return np.zeros((count,3),np.float32), n
    chosen = np.zeros(min(count,len(points)),dtype=np.int64)
    distances = np.full(len(points),np.inf)
    index = 0
    for i in range(len(chosen)):
        chosen[i] = index
        distances = np.minimum(distances,np.sum((points-points[index])**2,axis=1))
        index = int(np.argmax(distances))
    result = points[chosen]
    if len(result)<count: result = result[np.arange(count)%len(result)]
    return result.astype(np.float32), n
