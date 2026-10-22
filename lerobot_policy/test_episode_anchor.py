import unittest
import torch
from diffusion_overfit_common import vector_transform
from delivery_policy_contract import NORMALIZATION
from episode_goal_anchor import targets,decode,EPISODE_NORMALIZATION

class EpisodeAnchorTest(unittest.TestCase):
    def test_current_robot_error_does_not_move_the_target_reference(self):
        initial=torch.tensor([.5,.1,.2,1,0,0,0,.04]).repeat(1,2,1)
        state=initial.clone();state[:,:,:3]+=.02
        goals=torch.tensor([.3,.2,.045,.06,.06,.06,1,.9]).repeat(1,2,1)
        actions=torch.tensor([.3,.2,.095,1,0,0,0,.016]).repeat(1,16,1)
        batch={'observation.state':vector_transform(state,NORMALIZATION),'observation.reset_state':vector_transform(initial,NORMALIZATION),'observation.goal_features':goals,'action':vector_transform(actions,NORMALIZATION)}
        first=targets(batch,torch.tensor([24]))
        batch['observation.state']+=.1
        second=targets(batch,torch.tensor([24]))
        torch.testing.assert_close(first['action'],second['action'],atol=0,rtol=0)
        recovered=decode(first['action'],initial[:,-1,:3],goals[:,-1,:3],goals[:,-1,6])
        torch.testing.assert_close(recovered,actions,atol=2e-7,rtol=0)
        raw=vector_transform(first['action'],EPISODE_NORMALIZATION,inverse=True)
        torch.testing.assert_close(raw[:,:,2],torch.full((1,16),-.105),atol=2e-7,rtol=0)

if __name__=='__main__':unittest.main()
