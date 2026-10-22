import unittest
from types import SimpleNamespace
import numpy as np
import torch
from diffusion_overfit_common import Inference,vector_transform
from delivery_policy_contract import NORMALIZATION

class InferenceConstraintsTest(unittest.TestCase):
    def test_ensemble_and_declared_action_alphabet(self):
        class Model:
            def _prepare_global_conditioning(self,batch):return torch.zeros(1,256)
            def conditional_sample(self,batch_size,**kwargs):
                self.batch_size=batch_size
                a=torch.tensor([.4,0.,.2,.99,.1,0.,0.,.027]).repeat(batch_size,16,1)
                a[:,:,0]+=torch.arange(batch_size)[:,None]*.01
                return vector_transform(a,NORMALIZATION)
        model=Model();i=Inference.__new__(Inference);i.device='cpu';i.seed=0;i.modality='pointcloud';i.delivery=True
        i.history_validity=False;i.hold_timer=False;i.relative_actions=False;i.action_steps=8;i.action_samples=4
        i.norm=i.observation_norm=i.action_limits=NORMALIZATION;i.binary_gripper=True;i.fixed_down_quaternion=True
        i.policy=SimpleNamespace(diffusion=model);i.contract={};i.reset()
        a,d=i.predict(np.array([[.4,0.,.2,1,0,0,0,.04]],np.float32),point_cloud=np.full((1,1024,3),.4,np.float32))
        self.assertEqual(model.batch_size,4);np.testing.assert_allclose(a[0,:3],[.415,0,.2],atol=1e-7)
        np.testing.assert_array_equal(a[0,3:7],[1,0,0,0]);self.assertAlmostEqual(float(a[0,7]),.016,places=6)
        self.assertTrue(d['gripper_projected']);self.assertTrue(d['orientation_projected'])
        self.assertAlmostEqual(float(d['raw_action'][0,7]),.027,places=6)

if __name__=='__main__':unittest.main()
