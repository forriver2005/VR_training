"""Task-specific red/green RGB-D measurements, without simulator object poses.

Known colors and cube sizes are explicit task priors. Geometry is reconstructed
only from fixed-camera RGB, depth and calibration. Missing observations are
represented by a causal last measurement and age, never by simulation truth.
"""
import hashlib,json
from pathlib import Path
from itertools import product
import numpy as np
from delivery_policy_contract import ROOT,CROP,sha256
RECIPE='dominant_rgb_fixed_depth_components_cube_faces_causal_v2'
SIZES=(.04,.06)
NEIGHBORS=list(product([-1,0,1],repeat=3))


def colored_points(rgb,depth,k,camera_to_world,base_to_world,channel):
    rgb=np.asarray(rgb,dtype=np.float32)
    mask=(rgb[:,:,channel]>60)&(depth>.01)&(depth<3)&np.isfinite(depth)
    for other in range(3):
        if other!=channel:mask &= rgb[:,:,channel]>1.5*rgb[:,:,other]
    v,u=np.where(mask);z=depth[v,u].astype(np.float64)
    camera=np.stack([(u+.5-k[0,2])*z/k[0,0],(v+.5-k[1,2])*z/k[1,1],z],axis=1)
    t=np.linalg.inv(base_to_world)@camera_to_world
    points=camera@t[:3,:3].T+t[:3,3]
    return points[np.all((points>=CROP[0])&(points<=CROP[1]),axis=1)]


def components(points):
    if len(points)<12:return []
    keys=np.floor(points/.01).astype(np.int64);cells={}
    for i,key in enumerate(keys):cells.setdefault(tuple(key),[]).append(i)
    unseen=set(cells);groups=[]
    while unseen:
        key=min(unseen);unseen.remove(key);todo=[key];indices=[]
        while todo:
            key=todo.pop();indices.extend(cells[key])
            for offset in NEIGHBORS:
                n=tuple(key[j]+offset[j] for j in range(3))
                if n in unseen:unseen.remove(n);todo.append(n)
        if len(indices)>=12:groups.append(points[indices])
    return groups


def measure_objects(rgb,depth,intrinsics,poses,base):
    result=[]
    for channel,size in enumerate(SIZES):
        points=np.concatenate([colored_points(r,d,k,t,base,channel) for r,d,k,t in zip(rgb,depth,intrinsics,poses)])
        candidates=[]
        for group in components(points):
            lo,hi=np.quantile(group,[.01,.99],axis=0);extent=hi-lo
            # Reject wide reflections/backgrounds and thin vertical colored reflections.
            if np.max(extent)>1.6*size or np.min(extent[:2])<.45*size:continue
            centre=(lo+hi)/2;centre[2]=hi[2]-.5*size
            # A dense axis-aligned side face anchors the center even when the
            # opposite edge is occluded. Otherwise retain the measured extent midpoint.
            for axis in [0,1]:
                bins=np.round(group[:,axis]/.001).astype(np.int64)
                keys,counts=np.unique(bins,return_counts=True);peak=keys[np.argmax(counts)]
                selected=np.abs(group[:,axis]-peak*.001)<=.001
                face=float(np.median(group[selected,axis]))
                if selected.sum()>=max(8,.2*len(group)) and extent[axis]>.01:
                    if abs(face-lo[axis])<.003:centre[axis]=face+size/2
                    elif abs(face-hi[axis])<.003:centre[axis]=face-size/2
            # A cube cannot be centered below its half-size above base z=0;
            # this removes red/green surface reflections on the ground plane.
            if centre[2]<size/2:continue
            coverage=float(min(1.,np.min(extent[:2])/size))
            score=coverage*min(len(group),1000)
            candidates.append(dict(centre=centre,extent=extent,coverage=coverage,count=len(group),score=score))
        result.append(max(candidates,key=lambda c:c['score']) if candidates else None)
    return result


class ObjectTracker:
    def __init__(self):self.last=[None,None];self.age=[0.,0.]
    def observe(self,rgb,depth,intrinsics,poses,base):
        measurements=measure_objects(rgb,depth,intrinsics,poses,base);values=[]
        for i,m in enumerate(measurements):
            # The static target's partial occlusion should not replace a complete measurement.
            update=m is not None and (i==0 or self.last[i] is None or m['coverage']>=.8)
            if update:self.last[i]=m;self.age[i]=0.
            else:self.age[i]=min(10.,self.age[i]+.25)
            old=self.last[i]
            if old is None:values.extend([0.]*6+[0.,1.])
            else:values.extend([*old['centre'],*old['extent'],1.,min(self.age[i]/2,1.)])
        return np.asarray(values,dtype=np.float32),measurements


def load_object_features(entry):
    path=ROOT/entry['npz']
    if sha256(path)!=entry['source_npz_sha256']:raise ValueError('Source changed')
    folder=ROOT/'data/derived/delivery_rgbd_object_features_v1';folder.mkdir(parents=True,exist_ok=True)
    cached=folder/(entry['source_npz_sha256']+'_'+sha256(__file__)[:12]+'.npz')
    if cached.is_file():
        with np.load(cached) as z:return z['object_features']
    tracker=ObjectTracker();features=[]
    with np.load(path) as z:
        rgb1,rgb2=z['camera_1_rgb'],z['camera_2_rgb'];depth=z['depth'];ks=z['camera_intrinsics'];poses=z['camera_to_world'];base=z['base_to_world']
        for t in range(40):features.append(tracker.observe([rgb1[t],rgb2[t]],depth[t],ks,poses,base)[0])
    features=np.stack(features);np.savez_compressed(cached,object_features=features,recipe=RECIPE,source_sha256=entry['source_npz_sha256'])
    return features

GOAL_RECIPE='initial_green_cube_rgbd_geometry_memory_v1'
class GoalMemory:
    """Initial task goal, measured after reset; no episode clock or true object state."""
    def __init__(self):self.value=None
    def observe(self,rgb,depth,intrinsics,poses,base):
        if self.value is None:
            m=measure_objects(rgb,depth,intrinsics,poses,base)[1]
            if m is not None:self.value=np.asarray([*m['centre'],*m['extent'],1.,m['coverage']],dtype=np.float32)
        return np.array([0.]*8,dtype=np.float32) if self.value is None else self.value.copy()


def load_goal_features(entry):
    path=ROOT/entry['npz']
    if sha256(path)!=entry['source_npz_sha256']:raise ValueError('Source changed')
    folder=ROOT/'data/derived/delivery_rgbd_goal_memory_v1';folder.mkdir(parents=True,exist_ok=True)
    cached=folder/(entry['source_npz_sha256']+'_'+sha256(__file__)[:12]+'.npz')
    if cached.is_file():
        with np.load(cached) as z:return z['goal_features']
    memory=GoalMemory();features=[]
    with np.load(path) as z:
        rgb1,rgb2=z['camera_1_rgb'],z['camera_2_rgb'];depth=z['depth'];ks=z['camera_intrinsics'];poses=z['camera_to_world'];base=z['base_to_world']
        for t in range(40):features.append(memory.observe([rgb1[t],rgb2[t]],depth[t],ks,poses,base))
    features=np.stack(features);np.savez_compressed(cached,goal_features=features,recipe=GOAL_RECIPE,source_sha256=entry['source_npz_sha256'])
    return features
