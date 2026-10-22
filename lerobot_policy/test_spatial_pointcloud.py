import tempfile,unittest
import numpy as np
import torch
from lerobot.configs.types import FeatureType,PolicyFeature
from diffusion_overfit_common import make_config,vector_transform
from delivery_policy_contract import NORMALIZATION,normalize_points
from pointcloud_diffusion import PointCloudPolicy
from spatial_pointcloud import tcp_centered_inputs,SpatialPointEncoder

class SpatialTests(unittest.TestCase):
    def test_joint_translation_and_history_anchor(self):
        torch.manual_seed(7)
        p=torch.rand(2,2,1024,3)*.3+.2
        s=torch.tensor([.4,.1,.2,1,0,0,0,.04]).repeat(2,2,1);s[:,0,0]-=.01
        pn=torch.tensor(normalize_points(p.numpy()));sn=vector_transform(s,NORMALIZATION)
        pc,st=tcp_centered_inputs(pn,sn)
        delta=torch.tensor([.03,-.04,.02]);sp=s.clone();sp[:,:,:3]+=delta
        shifted=tcp_centered_inputs(torch.tensor(normalize_points((p+delta).numpy())),vector_transform(sp,NORMALIZATION))
        torch.testing.assert_close(shifted[0],pc,atol=3e-7,rtol=0)
        torch.testing.assert_close(shifted[1],st,atol=3e-7,rtol=0)
        torch.testing.assert_close(st[:,-1,:3],torch.zeros(2,3))
        torch.testing.assert_close(st[:,0,0],torch.full((2,),-.01/.8),atol=1e-7,rtol=0)
        torch.testing.assert_close(st[:,:,3:],sn[:,:,3:])

    def test_backward_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        cfg=make_config('cpu');cfg.input_features={'observation.state':PolicyFeature(FeatureType.STATE,(8,))}
        m=PointCloudPolicy(cfg,encoder_type='spatial_tokens',observation_frame='current_tcp')
        batch={'observation.state':torch.rand(2,2,8),'observation.point_cloud':torch.rand(2,2,1024,3)}
        out=m.diffusion._prepare_global_conditioning(batch);self.assertEqual(tuple(out.shape),(2,256))
        out.square().mean().backward()
        for param in m.diffusion.pointnet.parameters():
            self.assertIsNotNone(param.grad);self.assertTrue(torch.isfinite(param.grad).all())
        m.eval()
        with tempfile.TemporaryDirectory() as folder:
            m.save_pretrained(folder)
            other=PointCloudPolicy.from_pretrained(folder,config=cfg,encoder_type='spatial_tokens',observation_frame='current_tcp').eval()
            torch.testing.assert_close(m.diffusion._prepare_global_conditioning(batch),other.diffusion._prepare_global_conditioning(batch),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
