import unittest
import torch
from diffusion_overfit_common import vector_transform
from delivery_policy_contract import NORMALIZATION
from goal_anchor_action import GOAL_ANCHOR_NORMALIZATION,goal_anchor_targets,decode_goal_actions

class AnchorTests(unittest.TestCase):
    def test_roundtrip_and_transport_boundary(self):
        state=torch.tensor([.5,.1,.3,1.,0.,0.,0.,.02]).repeat(2,2,1)
        goals=torch.tensor([.3,.2,.045,.06,.06,.06,1,.9]).repeat(2,2,1)
        action=torch.tensor([.3,.2,.095,1.,0.,0.,0.,.02]).repeat(2,16,1)
        b={'observation.state':vector_transform(state,NORMALIZATION),'observation.goal_features':goals,'action':vector_transform(action,NORMALIZATION)}
        out=goal_anchor_targets(b,torch.tensor([16,28]))
        raw=vector_transform(out['action'],GOAL_ANCHOR_NORMALIZATION,inverse=True)
        torch.testing.assert_close(raw[0,:5,8],torch.zeros(5))
        torch.testing.assert_close(raw[0,5,8],torch.tensor(1/7),atol=1e-7,rtol=0)
        torch.testing.assert_close(raw[1,:,8],torch.ones(16))
        result=decode_goal_actions(out['action'],state[:,-1,:3],goals[:,-1,:3],goals[:,-1,6])
        torch.testing.assert_close(result,action,atol=2e-7,rtol=0)

    def test_measured_goal_motion_affects_only_predicted_anchor_share(self):
        raw=torch.tensor([0.,0.,.05,1,0,0,0,.04,0.]).repeat(1,3,1);raw[0,:,8]=torch.tensor([0.,.5,1.])
        encoded=vector_transform(raw,GOAL_ANCHOR_NORMALIZATION);tcp=torch.tensor([[.6,0.,.3]]);goal=torch.tensor([[.4,0.,.045]])
        first=decode_goal_actions(encoded,tcp,goal,torch.ones(1));second=decode_goal_actions(encoded,tcp,goal+torch.tensor([.03,0,0]),torch.ones(1))
        torch.testing.assert_close(second[0,:,0]-first[0,:,0],torch.tensor([0.,.015,.03]),atol=1e-7,rtol=0)
        missing=decode_goal_actions(encoded,tcp,goal,torch.zeros(1))
        torch.testing.assert_close(missing[0,:,0],torch.full((3,),.6))

if __name__=='__main__':unittest.main()
