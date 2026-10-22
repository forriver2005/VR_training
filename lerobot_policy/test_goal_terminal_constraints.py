import unittest
from types import SimpleNamespace
import numpy as np
import torch
from diffusion_overfit_common import Inference,vector_transform
from delivery_policy_contract import NORMALIZATION
from episode_goal_3d import EPISODE_3D_NORMALIZATION

class TerminalConstraintsTest(unittest.TestCase):
    def test_only_arrival_targets_are_projected_and_higher_targets_are_preserved(self):
        class Model:
            def _prepare_global_conditioning(self,batch):return torch.zeros(1,388)
            def conditional_sample(self,*args,**kwargs):
                a=torch.tensor([.03,-.02,-.18,1,0,0,0,.016,1.]).repeat(1,16,1)
                a[0,1,8]=.5;a[0,3,2]=.3
                return vector_transform(a,EPISODE_3D_NORMALIZATION)
        i=Inference.__new__(Inference);i.device='cpu';i.seed=0;i.modality='pointcloud';i.delivery=True
        i.history_validity=False;i.hold_timer=False;i.relative_actions=False;i.action_steps=8;i.goal_memory=True;i.reset_memory=True
        i.episode_goal_3d_actions=True;i.goal_terminal_constraints=True;i.stack_height_offset=.05
        i.norm=EPISODE_3D_NORMALIZATION;i.observation_norm=i.action_limits=NORMALIZATION;i.contract={};i.policy=SimpleNamespace(diffusion=Model());i.reset()
        state=np.array([[.5,0,.2,1,0,0,0,.04]],np.float32);goal=np.array([[.3,.2,.045,.06,.06,.06,1,.9]],np.float32)
        outputs=[]
        for t in range(3):
            a,d=i.predict(state,point_cloud=np.full((1,1024,3),.4,np.float32),goal_features=goal);outputs.append(a[0,:3])
        np.testing.assert_allclose(outputs,[[.43,.08,.02],[.3,.2,.095],[.3,.2,.345]],atol=2e-7)
        self.assertEqual(i.goal_terminal_projection_count,7)

if __name__=='__main__':unittest.main()
