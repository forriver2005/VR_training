import unittest
from types import SimpleNamespace
import numpy as np
import torch
from diffusion_overfit_common import Inference,vector_transform
from delivery_policy_contract import NORMALIZATION,BOUNDED_TIMER
from relative_action import RELATIVE_NORMALIZATION,relative_targets,decode_actions

class RelativeActionTest(unittest.TestCase):
    def test_training_roundtrip_and_translation_invariance(self):
        state=torch.tensor([[[.4,0.,.2,1.,0.,0.,0.,.04],[.42,0.,.2,1.,0.,0.,0.,.04]]])
        action=state[:,-1:].repeat(1,16,1);action[:,:,0]+=.03;action[:,:,2]-=.02
        batch={'observation.state':vector_transform(state,NORMALIZATION),'action':vector_transform(action,NORMALIZATION)}
        transformed=relative_targets(batch)
        anchor=vector_transform(batch['observation.state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
        recovered=decode_actions(transformed['action'],anchor,RELATIVE_NORMALIZATION)
        torch.testing.assert_close(recovered,action,atol=2e-7,rtol=0)
        self.assertIs(transformed['observation.state'],batch['observation.state'])
        shifted_s=state.clone();shifted_a=action.clone();shifted_s[:,:,:3]+=.05;shifted_a[:,:,:3]+=.05
        shifted=relative_targets({'observation.state':vector_transform(shifted_s,NORMALIZATION),'action':vector_transform(shifted_a,NORMALIZATION)})
        torch.testing.assert_close(shifted['action'],transformed['action'],atol=2e-7,rtol=0)

    def test_queued_world_actions_keep_the_original_anchor(self):
        class Model:
            def _prepare_global_conditioning(self,batch):return torch.zeros(1,260)
            def conditional_sample(self,*args,**kwargs):
                target=torch.zeros(1,16,8);target[:,:,0]=.01;target[:,:,3]=1.;target[:,:,7]=.02
                return vector_transform(target,RELATIVE_NORMALIZATION)
        inf=Inference.__new__(Inference);inf.device='cpu';inf.seed=0;inf.modality='pointcloud';inf.delivery=True
        inf.history_validity=True;inf.hold_timer=True;inf.relative_actions=True;inf.action_steps=8
        inf.norm=RELATIVE_NORMALIZATION;inf.observation_norm=NORMALIZATION;inf.action_limits=NORMALIZATION
        inf.contract={'timer':BOUNDED_TIMER};inf.policy=SimpleNamespace(diffusion=Model());inf.reset()
        outputs=[]
        for t in range(9):
            x=.4 if t==0 else .7
            action,_=inf.predict(np.array([[x,0,.2,1,0,0,0,.04]],np.float32),point_cloud=np.full((1,1024,3),.4,np.float32),elapsed_same_target_s=[0.])
            outputs.append(action[0,0])
        np.testing.assert_allclose(outputs,[.41]*8+[.71],atol=2e-7,rtol=0)
        self.assertEqual(inf.inference_calls,2)

if __name__=='__main__':unittest.main()
