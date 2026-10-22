import unittest
import torch
from lerobot.configs.types import FeatureType,PolicyFeature
from diffusion_overfit_common import make_config,vector_transform
from pointcloud_diffusion import PointCloudPolicy
from delivery_policy_contract import NORMALIZATION
from episode_goal_3d import EPISODE_3D_NORMALIZATION,decode,blend

class CanonicalGoalTests(unittest.TestCase):
    def test_posttransport_condition_independent_of_scene_origin_and_coverage(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        cfg=make_config('cpu');cfg.input_features={k:PolicyFeature(FeatureType.STATE,(n,)) for k,n in [('observation.state',8),('observation.reset_state',8),('observation.goal_features',8),('observation.history_valid',1),('observation.elapsed_same_target_s',1)]}
        m=PointCloudPolicy(cfg,encoder_type='proprioceptive',observation_frame='initial_tcp',learned_goal_gate=True,goal_state_switch=True).eval()
        with torch.no_grad():
            for p in m.diffusion.phase_mlp.parameters():p.zero_()
            m.diffusion.phase_mlp[-1].bias.fill_(10.)
        def batch(initial,goal,extent,coverage):
            state=torch.tensor([*(torch.tensor(goal)+torch.tensor([.0,.0,.05])),1,0,0,0,.02]).repeat(1,2,1)
            reset=torch.tensor([*initial,1,0,0,0,.04]).repeat(1,2,1)
            return {'observation.state':vector_transform(state,NORMALIZATION),'observation.reset_state':vector_transform(reset,NORMALIZATION),
                'observation.goal_features':torch.tensor([*goal,*extent,1,coverage]).repeat(1,2,1),'observation.point_cloud':torch.rand(1,2,1024,3),
                'observation.history_valid':torch.ones(1,2,1),'observation.elapsed_same_target_s':torch.full((1,2,1),.25)}
        a=m.diffusion._prepare_global_conditioning(batch([.5,.1,.215],[.3,.2,.045],[.06,.06,.06],.95))
        b=m.diffusion._prepare_global_conditioning(batch([.6,-.1,.43],[.4,-.2,.045],[.045,.06,.055],.75))
        torch.testing.assert_close(a,b,atol=2e-6,rtol=0)

    def test_vertical_anchor_only_engages_late_in_transport(self):
        alpha=torch.tensor([[0.,4/7,5/7,1.]])
        weights=blend(alpha);torch.testing.assert_close(weights[0,:,2],torch.tensor([0.,0.,1/3,1.]),atol=2e-7,rtol=0)
        raw=torch.tensor([0.,0.,.05,1,0,0,0,.02,1.]).repeat(1,4,1);raw[:,:,8]=alpha
        encoded=vector_transform(raw,EPISODE_3D_NORMALIZATION)
        initial=torch.tensor([[.5,0,.43]]);goal=torch.tensor([[.3,.2,.045]])
        a=decode(encoded,initial,goal,torch.ones(1));b=decode(encoded,initial,goal+torch.tensor([0,0,.1]),torch.ones(1))
        torch.testing.assert_close(b[0,:,2]-a[0,:,2],torch.tensor([0.,0.,.1/3,.1]),atol=2e-7,rtol=0)

if __name__=='__main__':unittest.main()
