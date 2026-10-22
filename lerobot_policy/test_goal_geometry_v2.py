import unittest
import numpy as np
from rgbd_goal_geometry_v2 import measure_goal

def render(eye,occluder=False,flat_patch=False):
    centre=np.array([.5,0.,.1]);eye=np.asarray(eye);forward=centre-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    rotation=np.stack([right,down,forward],axis=1);pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=eye
    k=np.array([[180.,0,64.],[0,180.,64.],[0,0,1.]])
    v,u=np.indices((128,128));rays=np.stack([(u+.5-64)/180,(v+.5-64)/180,np.ones_like(u)],axis=-1)@rotation.T
    plane_z=.13 if flat_patch else .02
    with np.errstate(divide='ignore',invalid='ignore'):depth=(plane_z-eye[2])/rays[:,:,2]
    depth=np.where(depth>0,depth,2.5);rgb=np.full((128,128,3),80,dtype=np.uint8)
    def box(lo,hi):
        with np.errstate(divide='ignore',invalid='ignore'):
            t1=(np.array(lo)-eye)/rays;t2=(np.array(hi)-eye)/rays
        near=np.minimum(t1,t2).max(-1);far=np.maximum(t1,t2).min(-1)
        return near,(far>=near)&(near>0)
    if flat_patch:
        point=eye+depth[:,:,None]*rays;green=(np.abs(point[:,:,0]-.5)<.03)&(np.abs(point[:,:,1])<.03)
        rgb[green]=[10,180,10]
    else:
        near,hit=box(centre-.03,centre+.03);green=hit&(near<depth);depth[green]=near[green];rgb[green]=[10,180,10]
        if occluder:
            near,hit=box([.52,-.09,.06],[.55,-.06,.22]);front=hit&(near<depth);depth[front]=near[front];rgb[front]=80
    return rgb,depth.astype(np.float32),k,pose

class GoalGeometryTest(unittest.TestCase):
    def test_two_view_partially_occluded_cube(self):
        views=[render([.8,-.4,.4],True),render([.2,-.35,.35],True)]
        m=measure_goal([v[0] for v in views],np.stack([v[1] for v in views]),np.stack([v[2] for v in views]),np.stack([v[3] for v in views]),np.eye(4))
        self.assertIsNotNone(m);np.testing.assert_allclose(m['centre'],[.5,0,.1],atol=.0015)
    def test_colored_patch_on_continuous_plane_is_not_a_cube(self):
        views=[render([.8,-.4,.4],flat_patch=True),render([.2,-.35,.35],flat_patch=True)]
        m=measure_goal([v[0] for v in views],np.stack([v[1] for v in views]),np.stack([v[2] for v in views]),np.stack([v[3] for v in views]),np.eye(4))
        self.assertIsNone(m)

if __name__=='__main__':unittest.main()
