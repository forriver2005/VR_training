"""Green target pose from measured surface normals and depth-supported edges.

No object transforms or simulator state are accepted by this module. Known task
priors: a green, initially static, axis-aligned 6cm cube. A missing geometric axis
is reported as unobserved, rather than treating an occlusion edge as a cube face.
"""
from itertools import product
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,CROP,sha256
GOAL_RECIPE='green_cube_normals_depth_silhouette_initial_memory_v2'
SIZE=.06;HALF=SIZE/2
NEIGHBORS=list(product([-1,0,1],repeat=3))


def view_geometry(rgb,depth,k,pose,base):
    d=np.asarray(depth,dtype=np.float64);rgb=np.asarray(rgb,dtype=np.float32)
    v,u=np.indices(d.shape);optical=np.stack([(u+.5-k[0,2])*d/k[0,0],(v+.5-k[1,2])*d/k[1,1],d],axis=-1)
    transform=np.linalg.inv(base)@pose
    points=optical@transform[:3,:3].T+transform[:3,3]
    good=np.isfinite(d)&(d>.01)&(d<3.)
    green=good&(rgb[:,:,1]>60)&(rgb[:,:,1]>1.5*rgb[:,:,0])&(rgb[:,:,1]>1.5*rgb[:,:,2])
    green &= np.all((points>=CROP[0])&(points<=CROP[1]),axis=-1)
    tu=np.roll(points,-1,axis=1)-np.roll(points,1,axis=1)
    tv=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    normals=np.cross(tu,tv);length=np.linalg.norm(normals,axis=-1)
    valid=green.copy()
    for axis in [0,1]:
        for shift in [-1,1]:valid &= np.roll(green,shift,axis=axis)
    valid[[0,-1],:]=False;valid[:,[0,-1]]=False
    valid &= (length>1e-10)&(np.linalg.norm(tu,axis=-1)<.02)&(np.linalg.norm(tv,axis=-1)<.02)
    normals/=np.maximum(length[:,:,None],1e-20)
    facing=np.sum(normals*(transform[:3,3]-points),axis=-1)
    normals[facing<0]*=-1;normals[~valid]=0.
    return dict(points=points,normals=normals,green=green,good=good,depth=d,k=k,transform=transform)


def groups(points):
    cells={}
    for i,key in enumerate(np.floor(points/.01).astype(np.int64)):cells.setdefault(tuple(key),[]).append(i)
    unseen=set(cells);out=[]
    while unseen:
        key=min(unseen);unseen.remove(key);queue=[key];indices=[]
        while queue:
            key=queue.pop();indices.extend(cells[key])
            for offset in NEIGHBORS:
                n=tuple(key[j]+offset[j] for j in range(3))
                if n in unseen:unseen.remove(n);queue.append(n)
        if len(indices)>=12:out.append(np.asarray(indices))
    return out


def modes(values,min_count=4,tolerance=.0015):
    if len(values)<min_count:return []
    bins,counts=np.unique(np.round(values/.001).astype(np.int64),return_counts=True)
    result=[]
    for idx in np.argsort(-counts,kind='stable'):
        centre=bins[idx]*.001;inside=np.abs(values-centre)<=tolerance
        if inside.sum()<min_count:continue
        value=float(np.median(values[inside]))
        if any(abs(value-r[0])<2*tolerance for r in result):continue
        result.append((value,int(inside.sum())))
        if len(result)>=4:break
    return result


def top_boundaries(views,lo,hi,top):
    points=[];directions=[]
    for view in views:
        p=view['points'];mask=view['green']&np.all((p>=lo-.003)&(p<=hi+.003),axis=-1)&(np.abs(p[:,:,2]-top)<.002)
        k=view['k'];tr=view['transform'];origin=tr[:3,3]
        vv,uu=np.indices(view['depth'].shape)
        for dv,du in [(0,1),(0,-1),(1,0),(-1,0)]:
            neighbor_d=np.roll(np.roll(view['depth'],-dv,axis=0),-du,axis=1)
            neighbor_green=np.roll(np.roll(view['green'],-dv,axis=0),-du,axis=1)
            neighbor_valid=np.roll(np.roll(view['good'],-dv,axis=0),-du,axis=1)
            # Compare with the TOP PLANE at the neighbor ray, not with the
            # foreground pixel depth: a slanted continuous plane has a depth
            # gradient but is not a silhouette (e.g. a colored reflection).
            ray_z=(uu+.5+du-k[0,2])/k[0,0]*tr[2,0]+(vv+.5+dv-k[1,2])/k[1,1]*tr[2,1]+tr[2,2]
            with np.errstate(divide='ignore',invalid='ignore'):
                expected=(top-origin[2])/ray_z
            edge=mask&~neighbor_green&neighbor_valid&np.isfinite(expected)&(expected>.01)&(neighbor_d-expected>.003)
            edge[[0,-1],:]=False;edge[:,[0,-1]]=False
            v,u=np.where(edge)
            if not len(u):continue
            def plane_points(offset):
                rays=np.stack([(u+.5+du*offset-k[0,2])/k[0,0],(v+.5+dv*offset-k[1,2])/k[1,1],np.ones(len(u))],axis=-1)@tr[:3,:3].T
                distance=(top-origin[2])/rays[:,2]
                return origin+distance[:,None]*rays
            boundary=plane_points(.5);outward=plane_points(1)-plane_points(0)
            points.append(boundary);directions.append(outward)
    if not points:return np.empty((0,3)),np.empty((0,3))
    return np.concatenate(points),np.concatenate(directions)


def measure_goal(rgb,depth,intrinsics,poses,base):
    views=[view_geometry(r,d,k,t,base) for r,d,k,t in zip(rgb,depth,intrinsics,poses)]
    points=np.concatenate([v['points'][v['green']] for v in views]);normals=np.concatenate([v['normals'][v['green']] for v in views])
    candidates=[]
    for index in groups(points):
        p,n=points[index],normals[index];lo,hi=np.quantile(p,[.01,.99],axis=0);extent=hi-lo
        if np.max(extent)>1.6*SIZE or np.min(extent[:2])<.2*SIZE:continue
        tops=[(z,count) for z,count in modes(p[n[:,2]>.97,2]-HALF) if z>=HALF]
        if not tops:continue
        centre_z,z_support=max(tops,key=lambda r:r[1]);top=centre_z+HALF
        top_points=p[np.abs(p[:,2]-top)<.002]
        if len(top_points)<6:continue
        boundary,directions=top_boundaries(views,lo,hi,top)
        axes=[];methods=[]
        for axis in [0,1]:
            select=np.abs(n[:,axis])>.97
            estimates=p[select,axis]-np.sign(n[select,axis])*HALF
            possible=[(c,num,'surface_normal') for c,num in modes(estimates)]
            # Silhouette fallback requires a line with substantial extent along
            # the other axis, not a single corner or an internal pixel row.
            if not possible and len(boundary):
                for edge,num in modes(boundary[:,axis],min_count=4,tolerance=.002):
                    line=np.abs(boundary[:,axis]-edge)<.002
                    if np.ptp(boundary[line,1-axis])<.3*SIZE:continue
                    signed=directions[line,axis]
                    signed=signed[np.abs(signed)>1e-5]
                    if not len(signed):continue
                    sign=np.sign(np.median(signed))
                    possible.append((edge-sign*HALF,num,'depth_silhouette'))
            valid=[]
            for c,num,method in possible:
                covered=float(np.mean(np.abs(top_points[:,axis]-c)<=HALF+.003))
                if covered>=.90:valid.append((c,num,method,covered))
            if not valid:break
            best=max(valid,key=lambda r:(r[3],r[1]));axes.append(best[0]);methods.append(best[2])
        if len(axes)!=2:continue
        centre=np.array([*axes,centre_z])
        q=np.abs(p-centre)-HALF
        distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
        inliers=float(np.mean(np.abs(distance)<.003))
        if inliers<.70:continue
        coverage=float(min(1.,np.min(extent[:2])/SIZE))
        candidates.append(dict(centre=centre,extent=extent,coverage=coverage,count=len(p),methods=methods+['surface_normal'],inlier_fraction=inliers,score=inliers*min(len(p),1000)))
    return max(candidates,key=lambda c:c['score']) if candidates else None


class GoalMemory:
    def __init__(self):self.value=None;self.last_measurement=None
    def observe(self,rgb,depth,intrinsics,poses,base):
        if self.value is None:
            m=measure_goal(rgb,depth,intrinsics,poses,base);self.last_measurement=m
            if m is not None:self.value=np.asarray([*m['centre'],*m['extent'],1.,m['coverage']],dtype=np.float32)
        return np.zeros(8,dtype=np.float32) if self.value is None else self.value.copy()


def load_goal_features(entry):
    path=ROOT/entry['npz']
    if sha256(path)!=entry['source_npz_sha256']:raise ValueError('Source changed')
    folder=ROOT/'data/derived/delivery_rgbd_goal_memory_v2';folder.mkdir(parents=True,exist_ok=True)
    cached=folder/(entry['source_npz_sha256']+'_'+sha256(__file__)[:12]+'.npz')
    if cached.is_file():
        with np.load(cached) as z:return z['goal_features']
    memory=GoalMemory();features=[]
    with np.load(path) as z:
        rgb1,rgb2=z['camera_1_rgb'],z['camera_2_rgb'];depth=z['depth'];ks=z['camera_intrinsics'];poses=z['camera_to_world'];base=z['base_to_world']
        for t in range(40):features.append(memory.observe([rgb1[t],rgb2[t]],depth[t],ks,poses,base))
    features=np.stack(features);np.savez_compressed(cached,goal_features=features,recipe=GOAL_RECIPE,source_sha256=entry['source_npz_sha256'])
    return features
