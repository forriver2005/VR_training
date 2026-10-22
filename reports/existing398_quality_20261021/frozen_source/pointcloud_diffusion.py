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

class NoPointCloudEncoder(nn.Module):
    """Explicit measured-goal/proprioception ablation; no background visual feature."""
    def forward(self,points):return points.new_zeros((points.shape[0],64))

class PointCloudModel(DiffusionModel):
    def __init__(self,config,encoder_type='pointnet',observation_frame='base',learned_goal_gate=False,goal_state_switch=False):
        nn.Module.__init__(self)
        self.config=config
        if encoder_type=='pointnet':self.pointnet=PointNetEncoderXYZ()
        elif encoder_type=='spatial_tokens':
            from spatial_pointcloud import SpatialPointEncoder
            self.pointnet=SpatialPointEncoder()
        elif encoder_type=='proprioceptive':self.pointnet=NoPointCloudEncoder()
        else:raise ValueError('Unknown point encoder '+encoder_type)
        if observation_frame not in ['base','current_tcp','initial_tcp']:raise ValueError('Unknown observation frame')
        self.observation_frame=observation_frame
        self.state_mlp=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64))
        self.history_validity='observation.history_valid' in config.input_features
        self.hold_timer='observation.elapsed_same_target_s' in config.input_features
        self.goal_memory='observation.goal_features' in config.input_features
        if self.goal_memory:self.goal_mlp=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64))
        self.learned_goal_gate=learned_goal_gate
        self.goal_state_switch=goal_state_switch
        if goal_state_switch and not learned_goal_gate:raise ValueError('Goal state switching requires a learned gate')
        if learned_goal_gate:
            if observation_frame!='initial_tcp' or not self.goal_memory:raise ValueError('Learned gate requires initial state and measured goal')
            self.phase_mlp=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.unet=DiffusionConditionalUnet1d(config,global_cond_dim=256+2*int(self.history_validity)+2*int(self.hold_timer)+128*int(self.goal_memory))
        self.noise_scheduler=DDPMScheduler(num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,prediction_type=config.prediction_type,
            clip_sample=True,clip_sample_range=1.)
        self.num_inference_steps=config.num_inference_steps or config.num_train_timesteps

    def _prepare_global_conditioning(self,batch):
        p=batch['observation.point_cloud']
        b,h,n,d=p.shape
        if (h,n,d)!=(2,1024,3): raise ValueError(f'Unexpected point cloud shape {p.shape}')
        state_input=batch['observation.state']
        if self.observation_frame=='current_tcp':
            from spatial_pointcloud import tcp_centered_inputs
            p,state_input=tcp_centered_inputs(p,state_input)
        elif self.observation_frame=='initial_tcp':
            from delivery_policy_contract import CROP,NORMALIZATION
            reset=batch['observation.reset_state'][:,-1:]
            low=torch.as_tensor(NORMALIZATION['low'][:3],device=p.device,dtype=p.dtype)
            span=torch.as_tensor(NORMALIZATION['high'][:3],device=p.device,dtype=p.dtype)-low
            anchor=(reset[:,:,:3]+1)*.5*span+low
            crop=torch.as_tensor(CROP,device=p.device,dtype=p.dtype)
            p=((p+1)*.5*(crop[1]-crop[0])+crop[0]-anchor[:,:,None])/(crop[1]-crop[0])
            state_input=state_input.clone();state_input[:,:,:3]=.5*(state_input[:,:,:3]-reset[:,:,:3])
        feature=self.pointnet(p.reshape(b*h,n,d)).reshape(b,h,64)
        state=self.state_mlp(state_input)
        features=[feature,state]
        if self.goal_memory:
            from delivery_policy_contract import CROP,NORMALIZATION
            goal=batch['observation.goal_features'].clone()
            if goal.shape!=(b,h,8):raise ValueError('Expected measured goal features (B,2,8)')
            low=torch.as_tensor(NORMALIZATION['low'][:3],device=goal.device,dtype=goal.dtype)
            span=torch.as_tensor(NORMALIZATION['high'][:3],device=goal.device,dtype=goal.dtype)-low
            anchor=(batch['observation.state'][:,-1:,:3]+1)*.5*span+low
            crop=torch.as_tensor(CROP,device=goal.device,dtype=goal.dtype)
            if self.observation_frame in ['current_tcp','initial_tcp']:goal[:,:,:3]=(goal[:,:,:3]-anchor)/(crop[1]-crop[0])
            else:goal[:,:,:3]=2*(goal[:,:,:3]-crop[0])/(crop[1]-crop[0])-1
            goal[:,:,:3]*=goal[:,:,6:7]
            goal[:,:,3:6]/=.06
            if self.goal_state_switch:
                # Bounding-box coverage changes with occlusion/background; it
                # must not serve as an implicit layout identity in this model.
                goal[:,:,3:6]=0.;goal[:,:,7]=0.
            goal_embedding=self.goal_mlp(goal)
            if self.learned_goal_gate:
                # This head sees robot history only, never goals, point clouds,
                # frame indices or the training label used by its auxiliary loss.
                phase_input=torch.cat([state_input,batch['observation.history_valid'],batch['observation.elapsed_same_target_s']],dim=-1).flatten(1)
                self.last_phase_logits=self.phase_mlp(phase_input)
                prob=self.last_phase_logits.sigmoid()
                gate=(prob>.5).to(prob.dtype)+prob-prob.detach()
                goal_embedding=goal_embedding*gate[:,None]
                features[0]=features[0]*gate[:,None]
                if self.goal_state_switch:
                    # Before transport use displacement from measured reset;
                    # after the predicted transition use displacement from the
                    # observed target. The classifier itself stays reset-relative.
                    initial=(batch['observation.reset_state'][:,-1:,:3]+1)*.5*span+low
                    actual_goal=batch['observation.goal_features'][:,:,:3]
                    valid_goal=batch['observation.goal_features'][:,:,6:7]
                    canonical=state_input.clone()
                    canonical[:,:,:3]-=gate[:,None]*valid_goal*(actual_goal-initial)/span
                    features[1]=self.state_mlp(canonical)
            features.append(goal_embedding)
        if self.history_validity:
            valid=batch['observation.history_valid']
            if valid.shape!=(b,h,1): raise ValueError('Expected history validity (B,2,1)')
            features.append(valid)
        if self.hold_timer:
            timer=batch['observation.elapsed_same_target_s']
            if timer.shape!=(b,h,1): raise ValueError('Expected normalized hold timer (B,2,1)')
            features.append(timer)
        return torch.cat(features,dim=-1).flatten(start_dim=1)

class PointCloudPolicy(DiffusionPolicy):
    def __init__(self,config,encoder_type='pointnet',observation_frame='base',learned_goal_gate=False,position_loss_weight=1.,goal_state_switch=False,**kwargs):
        # The configuration remains serializable by LeRobot; this local loader
        # chooses PointCloudPolicy from the saved modality contract.
        PreTrainedPolicy.__init__(self,config)
        self.diffusion=PointCloudModel(config,encoder_type,observation_frame,learned_goal_gate,goal_state_switch)
        self.position_loss_weight=float(position_loss_weight)
        self.reset()

    def forward(self,batch):
        # Shared DiffusionModel asserts a visual or environment key. The actual
        # conditioning uses only measured XYZ and proprioception above.
        batch=dict(batch)
        batch['observation.environment_state']=batch['observation.state']
        if self.position_loss_weight==1.:
            loss=self.diffusion.compute_loss(batch)
        else:
            # Same DDPM training objective and random draws as LeRobot; only
            # XYZ residual loss is weighted to prioritize millimeter accuracy.
            dm=self.diffusion;condition=dm._prepare_global_conditioning(batch)
            trajectory=batch['action'];noise=torch.randn(trajectory.shape,device=trajectory.device)
            if trajectory.shape[1]!=dm.config.horizon or batch['observation.state'].shape[1]!=dm.config.n_obs_steps:raise ValueError('Diffusion window mismatch')
            steps=torch.randint(0,dm.noise_scheduler.config.num_train_timesteps,(len(trajectory),),device=trajectory.device).long()
            noisy=dm.noise_scheduler.add_noise(trajectory,noise,steps)
            prediction=dm.unet(noisy,steps,global_cond=condition)
            target=noise if dm.config.prediction_type=='epsilon' else trajectory
            errors=(prediction-target).square()
            weights=torch.ones(errors.shape[-1],device=errors.device);weights[:3]=self.position_loss_weight
            errors=errors*weights
            if dm.config.do_mask_loss_for_padding:errors=errors*(~batch['action_is_pad']).unsqueeze(-1)
            loss=errors.mean()
        if self.diffusion.learned_goal_gate:
            phase=nn.functional.binary_cross_entropy_with_logits(self.diffusion.last_phase_logits[:,0],batch['transport_started_target'])
            return loss+.05*phase,dict(diffusion_loss=loss.detach(),phase_loss=phase.detach())
        return loss,None
