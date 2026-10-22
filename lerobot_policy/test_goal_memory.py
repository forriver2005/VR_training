import unittest
from unittest.mock import patch
import numpy as np
import torch
from rgbd_object_features import GoalMemory
from diffusion_overfit_common import make_config,vector_transform
from delivery_policy_contract import NORMALIZATION,normalize_points
from lerobot.configs.types import FeatureType,PolicyFeature
from pointcloud_diffusion import PointCloudPolicy

class GoalTests(unittest.TestCase):
    def test_causal_missing_then_fixed_measurement(self):
        g=GoalMemory();m=dict(centre=np.array([.4,.2,.045]),extent=np.full(3,.06),coverage=.9)
        with patch('rgbd_object_features.measure_objects',side_effect=[[None,None],[None,m]]) as detector:
            np.testing.assert_array_equal(g.observe(None,None,None,None,None),np.zeros(8))
            measured=g.observe(None,None,None,None,None)
            saved=g.observe(None,None,None,None,None)
            self.assertEqual(detector.call_count,2)
            np.testing.assert_array_equal(measured,saved)
            measured[0]=100
            self.assertEqual(float(g.observe(None,None,None,None,None)[0]),float(saved[0]))
        fresh=GoalMemory()
        self.assertIsNone(fresh.value)

    def test_goal_condition_translation_and_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(2)
        cfg=make_config('cpu');cfg.input_features={'observation.state':PolicyFeature(FeatureType.STATE,(8,)),'observation.goal_features':PolicyFeature(FeatureType.STATE,(8,))}
        m=PointCloudPolicy(cfg,observation_frame='current_tcp').eval()
        state=torch.tensor([.4,.1,.2,1,0,0,0,.04]).repeat(2,2,1)
        points=torch.rand(2,2,1024,3)*.2+.25
        goal=torch.tensor([.5,.2,.045,.06,.06,.06,1,.9]).repeat(2,2,1)
        batch={'observation.state':vector_transform(state,NORMALIZATION),'observation.point_cloud':torch.tensor(normalize_points(points.numpy())),'observation.goal_features':goal}
        a=m.diffusion._prepare_global_conditioning(batch)
        shifted={k:v.clone() for k,v in batch.items()};delta=torch.tensor([.03,-.02,.01]);state[:,:,:3]+=delta;goal[:,:,:3]+=delta
        shifted['observation.state']=vector_transform(state,NORMALIZATION);shifted['observation.point_cloud']=torch.tensor(normalize_points((points+delta).numpy()));shifted['observation.goal_features']=goal
        b=m.diffusion._prepare_global_conditioning(shifted)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=0)
        self.assertEqual(a.shape,(2,384));a.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.diffusion.goal_mlp.parameters()))

if __name__=='__main__':unittest.main()
