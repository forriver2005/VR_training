"""Grasp1-style XYZ PointNet + state MLP with the existing DDPM action model."""
import torch
from torch import nn
from diffusers import DDPMScheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, DiffusionModel, DiffusionConditionalUnet1d

class PointNetEncoderXYZ(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in [(3,64),(64,128),(128,256)]:
            layers.extend([nn.Linear(a,b),nn.LayerNorm(b),nn.ReLU()])
        self.mlp=nn.Sequential(*layers)
        self.projection=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64))

    def forward(self, points):
        return self.projection(self.mlp(points).amax(dim=1))

class PointCloudModel(DiffusionModel):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.config=config
        self.pointnet=PointNetEncoderXYZ()
        self.state_mlp=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64))
        self.history_validity='observation.history_valid' in config.input_features
        self.unet=DiffusionConditionalUnet1d(config,global_cond_dim=258 if self.history_validity else 256)
        self.noise_scheduler=DDPMScheduler(num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,prediction_type=config.prediction_type,
            clip_sample=True,clip_sample_range=1.)
        self.num_inference_steps=100

    def _prepare_global_conditioning(self,batch):
        p=batch['observation.point_cloud']
        b,h,n,d=p.shape
        if (h,n,d)!=(2,1024,3): raise ValueError(f'Unexpected point cloud shape {p.shape}')
        feature=self.pointnet(p.reshape(b*h,n,d)).reshape(b,h,64)
        state=self.state_mlp(batch['observation.state'])
        features=[feature,state]
        if self.history_validity:
            valid=batch['observation.history_valid']
            if valid.shape!=(b,h,1): raise ValueError('Expected history validity (B,2,1)')
            features.append(valid)
        return torch.cat(features,dim=-1).flatten(start_dim=1)

class PointCloudPolicy(DiffusionPolicy):
    def __init__(self,config,**kwargs):
        # The configuration remains serializable by LeRobot; this local loader
        # chooses PointCloudPolicy from the saved modality contract.
        PreTrainedPolicy.__init__(self,config)
        self.diffusion=PointCloudModel(config)
        self.reset()

    def forward(self,batch):
        # Shared DiffusionModel asserts a visual or environment key. The actual
        # conditioning uses only measured XYZ and proprioception above.
        batch=dict(batch)
        batch['observation.environment_state']=batch['observation.state']
        return self.diffusion.compute_loss(batch),None
