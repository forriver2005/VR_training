import unittest
import torch
from torch import nn
from lerobot.configs.types import FeatureType,PolicyFeature
from diffusion_overfit_common import make_config
from pointcloud_diffusion import PointCloudPolicy

class MonotonicGateTest(unittest.TestCase):
    def test_transport_event_cannot_revert_and_explicit_reset_clears_it(self):
        torch.set_num_threads(2);cfg=make_config('cpu')
        cfg.input_features={k:PolicyFeature(FeatureType.STATE,(n,)) for k,n in [('observation.state',8),('observation.reset_state',8),('observation.goal_features',8),('observation.history_valid',1),('observation.elapsed_same_target_s',1)]}
        p=PointCloudPolicy(cfg,encoder_type='proprioceptive',observation_frame='initial_tcp',learned_goal_gate=True,goal_state_switch=True).eval();d=p.diffusion
        class Prediction(nn.Module):
            def __init__(self):super().__init__();self.value=-8.
            def forward(self,x):return x.new_full((len(x),1),self.value)
        d.phase_mlp=Prediction();d.monotonic_goal_gate=True
        b={'observation.state':torch.zeros(1,2,8),'observation.reset_state':torch.zeros(1,2,8),'observation.goal_features':torch.tensor([.5,0.,.045,.06,.06,.06,1.,.9]).repeat(1,2,1),'observation.point_cloud':torch.zeros(1,2,1024,3),'observation.history_valid':torch.ones(1,2,1),'observation.elapsed_same_target_s':torch.zeros(1,2,1)}
        d._prepare_global_conditioning(b);self.assertEqual(float(d.last_effective_goal_gate[0,0]),0.)
        d.phase_mlp.value=8.;active=d._prepare_global_conditioning(b);self.assertEqual(float(d.last_effective_goal_gate[0,0]),1.)
        d.phase_mlp.value=-8.;continued=d._prepare_global_conditioning(b);self.assertEqual(float(d.last_effective_goal_gate[0,0]),1.)
        torch.testing.assert_close(active,continued,atol=0,rtol=0)
        d.inference_goal_gate_latch=None;d._prepare_global_conditioning(b);self.assertEqual(float(d.last_effective_goal_gate[0,0]),0.)

if __name__=='__main__':unittest.main()
