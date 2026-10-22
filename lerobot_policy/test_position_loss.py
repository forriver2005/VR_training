import unittest
import torch
from torch import nn
from lerobot.configs.types import FeatureType,PolicyFeature
from diffusion_overfit_common import make_config
from pointcloud_diffusion import PointCloudPolicy

class PositionLossTest(unittest.TestCase):
    def test_only_xyz_channels_receive_extra_weight(self):
        cfg=make_config('cpu');cfg.prediction_type='sample';cfg.output_features['action']=PolicyFeature(FeatureType.ACTION,(9,));cfg.input_features={'observation.state':PolicyFeature(FeatureType.STATE,(8,))}
        policy=PointCloudPolicy(cfg,encoder_type='proprioceptive',position_loss_weight=6.)
        class Constant(nn.Module):
            def __init__(self):super().__init__();self.value=nn.Parameter(torch.zeros(9))
            def forward(self,x,*args,**kwargs):return self.value.expand_as(x)
        policy.diffusion.unet=Constant()
        batch={'observation.state':torch.zeros(1,2,8),'observation.point_cloud':torch.zeros(1,2,1024,3),'action':torch.ones(1,16,9),'action_is_pad':torch.zeros(1,16,dtype=torch.bool)}
        loss,_=policy(batch);torch.testing.assert_close(loss,torch.tensor(24/9))
        loss.backward();g=policy.diffusion.unet.value.grad
        torch.testing.assert_close(g[:3],g[3:6]*6)

if __name__=='__main__':unittest.main()
