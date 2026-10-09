"""Audit RGB-D reprojection, frozen expert replay, and independent cube surfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rgbd_geometry import BASE_WORLD, backproject, point_cloud_from_depth


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    ep=np.load(args.source)
    ref=np.load(args.reference)
    manifest=json.loads((args.source.parent/'replay_manifest.json').read_text())
    row=manifest['episodes'][0]
    geometry=[]
    for i in range(2):
        d=ep['depth'][0,i]; k=ep['camera_intrinsics'][i]; t=ep['camera_to_world'][i]
        points,_=backproject(d,k,t)
        cam=(points+BASE_WORLD-t[:3,3])@t[:3,:3]
        u=cam[:,0]/cam[:,2]*k[0,0]+k[0,2]
        v=cam[:,1]/cam[:,2]*k[1,1]+k[1,2]
        valid=np.isfinite(d)&(d>.01)&(d<100)
        yy,xx=np.nonzero(valid)
        geometry.append(dict(camera=i+1,pixel_roundtrip_max=float(np.max(np.abs(np.stack([u-xx-.5,v-yy-.5])))),
                             depth_roundtrip_max_m=float(np.max(np.abs(cam[:,2]-d[valid])))))
    cubes=[]
    points=np.concatenate([backproject(ep['depth'][0,i],ep['camera_intrinsics'][i],ep['camera_to_world'][i])[0]
                           for i in range(2)])
    world=points+BASE_WORLD
    # Simulator truth is used only in this audit, never to create or crop policy inputs.
    for name,size in [('a',.02),('b',.025)]:
        center=np.array(row[f'cube_{name}_resting_xyz'])
        top=center[2]+size/2
        inside=(np.abs(world[:,:2]-center[:2])<size*.4).all(axis=1)
        surface=world[inside & (np.abs(world[:,2]-top)<.005)]
        if not len(surface): raise ValueError(f'No measured top-surface points for cube {name}')
        cubes.append(dict(cube=name,measured_top_points=len(surface),
                          top_error_m=float(np.median(np.abs(surface[:,2]-top)))))
    cloud_error=0.
    for frame in range(40):
        cloud,_=point_cloud_from_depth(ep['depth'][frame],ep['camera_intrinsics'],ep['camera_to_world'])
        cloud_error=max(cloud_error,float(np.abs(cloud-ep['point_cloud'][frame]).max()))
    report=dict(source=str(args.source.resolve()),sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                success=row['success'],control_dt_s=float(ep['control_dt_s']),
                observation_dt_min_s=float(np.diff(ep['observation_time_s']).min()),
                observation_dt_max_s=float(np.diff(ep['observation_time_s']).max()),
                original_state_max_difference=float(np.abs(ep['state']-ref['state']).max()),
                original_action_max_difference=float(np.abs(ep['action']-ref['action']).max()),
                geometry=geometry,cube_surface_audit=cubes,point_cloud_reconstruction_max_m=cloud_error,
                point_cloud_contract=manifest['point_cloud_contract'])
    if not row['success'] or cloud_error>1e-6 or max(c['top_error_m'] for c in cubes)>.002:
        raise ValueError(f'Failed RGB-D audit: {report}')
    (args.output/'geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    try:
        import matplotlib
    except ImportError:
        print('Numeric audit passed; install matplotlib to generate the optional figure.')
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i in range(2):
        axes[i,0].imshow(ep[f'camera_{i+1}_rgb'][0]); axes[i,0].set_title(f'Camera {i+1} RGB, frame 0')
        axes[i,1].imshow(ep['depth'][0,i],vmin=0,vmax=2); axes[i,1].set_title('Axial depth (m)')
        cloud=ep['point_cloud'][0]
        axes[i,2].scatter(cloud[:,0],cloud[:,i+1],c=cloud[:,2],s=2)
        axes[i,2].set_xlabel('Base x (m)'); axes[i,2].set_ylabel('Base '+('y' if i==0 else 'z')+' (m)')
        axes[i,2].set_aspect('equal')
    fig.tight_layout(); fig.savefig(args.output/'rgbd_geometry_audit.png',dpi=150); plt.close(fig)


if __name__=='__main__': main()
