import unittest
import torch
from lerobot.configs.types import FeatureType,PolicyFeature
from diffusion_overfit_common import make_config,vector_transform
from pointcloud_diffusion import PointCloudPolicy
from goal_xy_anchor_action import NORMALIZATION_XY,targets,decode
from delivery_policy_contract import NORMALIZATION

class GoalGateTests(unittest.TestCase):
    def test_no_goal_or_phase_label_enters_closed_gate(self):
        torch.set_num_threads(2);torch.manual_seed(9)
        cfg=make_config('cpu');cfg.input_features={k:PolicyFeature(FeatureType.STATE,(n,)) for k,n in [('observation.state',8),('observation.reset_state',8),('observation.goal_features',8),('observation.history_valid',1),('observation.elapsed_same_target_s',1)]}
        m=PointCloudPolicy(cfg,encoder_type='proprioceptive',observation_frame='initial_tcp',learned_goal_gate=True).eval()
        with torch.no_grad():
            for p in m.diffusion.phase_mlp.parameters():p.zero_()
            m.diffusion.phase_mlp[-1].bias.fill_(-10.)
        s=torch.tensor([.5,.1,.2,1,0,0,0,.04]).repeat(1,2,1);s=vector_transform(s,NORMALIZATION)
        batch={'observation.state':s,'observation.reset_state':s.clone(),'observation.point_cloud':torch.rand(1,2,1024,3),'observation.goal_features':torch.tensor([.4,.2,.045,.06,.06,.06,1,.9]).repeat(1,2,1),'observation.history_valid':torch.tensor([[[0.],[1.]]]),'observation.elapsed_same_target_s':torch.zeros(1,2,1)}
        a=m.diffusion._prepare_global_conditioning(batch)
        changed=dict(batch);changed['observation.goal_features']=batch['observation.goal_features'].clone();changed['observation.goal_features'][:,:,:3]+=.2
        changed['transport_started_target']=torch.ones(1)
        b=m.diffusion._prepare_global_conditioning(changed)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_xy_goal_anchor_does_not_drag_vertical_motion(self):
        raw=torch.tensor([0.,0.,.02,1.,0.,0.,0.,.02,.5]).reshape(1,1,9)
        output=vector_transform(raw,NORMALIZATION_XY);tcp=torch.tensor([[.5,0.,.4]]);goal=torch.tensor([[.3,.2,.045]])
        a=decode(output,tcp,goal,torch.ones(1));b=decode(output,tcp,goal+torch.tensor([.04,0.,.2]),torch.ones(1))
        torch.testing.assert_close(b[0,0,0]-a[0,0,0],torch.tensor(.02),atol=1e-7,rtol=0)
        torch.testing.assert_close(a[:,:,2],torch.tensor([[.42]]),atol=1e-7,rtol=0)
        torch.testing.assert_close(a[:,:,2],b[:,:,2],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
