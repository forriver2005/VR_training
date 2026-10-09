"""Camera depth -> FR3-base XYZ; shared by collection and policy evaluation."""
import numpy as np

BASE_WORLD = np.array([-.615, 0., 0.], dtype=np.float32)
LOW = np.array([.15, -.45, 0.], dtype=np.float32)
HIGH = np.array([.95, .25, .95], dtype=np.float32)

def geometry_contract():
    return dict(depth="distance_to_image_plane; axial metres", pixel_center_offset=.5,
                optical_axes="x right, y down, z forward", transform="camera_to_world then subtract base_world",
                base_world=BASE_WORLD.tolist(), crop_low=LOW.tolist(), crop_high=HIGH.tolist(),
                sampling="deterministic FPS; farthest from centroid start; repeat if insufficient", points=1024)

def backproject(depth, intrinsics, camera_to_world):
    """Axial depth in metres; optical axes x right, y down, z forward."""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w)+.5, np.arange(h)+.5)
    valid = np.isfinite(depth) & (depth > .01) & (depth < 100.)
    z = depth[valid]
    k = np.asarray(intrinsics)
    cam = np.column_stack(((u[valid]-k[0,2])*z/k[0,0],
                           (v[valid]-k[1,2])*z/k[1,1], z))
    t = np.asarray(camera_to_world)
    world = cam @ t[:3,:3].T + t[:3,3]
    return (world - BASE_WORLD).astype(np.float32), float(1-valid.mean())

def sample_fps(points, count=1024):
    points = np.asarray(points, dtype=np.float32)
    if count <= 0: raise ValueError('count must be positive')
    if not len(points): raise ValueError('Empty workspace point cloud')
    # Deterministic start and tie-breaking; no scene/object labels are inputs.
    selected = np.empty(min(count,len(points)), dtype=np.int64)
    selected[0] = np.argmax(np.sum((points-points.mean(0))**2,axis=1))
    distance = np.full(len(points),np.inf,dtype=np.float32)
    for i in range(1,len(selected)):
        delta = points-points[selected[i-1]]
        distance = np.minimum(distance,np.einsum('ij,ij->i',delta,delta))
        distance[selected[:i]] = -1
        selected[i] = np.argmax(distance)
    if len(selected)<count: selected=np.resize(selected,count)
    return points[selected]

def point_cloud_from_depth(depths, intrinsics, camera_to_world, count=1024):
    if len(depths)!=len(intrinsics) or len(depths)!=len(camera_to_world):
        raise ValueError('Depth and calibration camera counts must match')
    clouds=[]; invalid=[]; retained=[]
    for d,k,t in zip(depths,intrinsics,camera_to_world):
        pts,bad=backproject(d,k,t)
        pts=pts[(pts>=LOW).all(1)&(pts<=HIGH).all(1)]
        clouds.append(pts); invalid.append(bad); retained.append(len(pts))
    cloud=np.concatenate(clouds,axis=0)
    return sample_fps(cloud,count),dict(invalid_fraction=invalid,retained_per_camera=retained,
                                       workspace_points=len(cloud),insufficient_points=len(cloud)<count)

def normalize_points(points):
    return 2*(np.asarray(points,dtype=np.float32)-LOW)/(HIGH-LOW)-1
