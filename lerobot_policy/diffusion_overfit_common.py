"""Fixed normalization and shared inference for the RGB overfit experiment."""
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

RAW = Path('data/isaacsim_trajectorysuccessful2_fr3v2_strict_friction12_20260927/episode_000000.npz')
CAMERAS = ['observation.camera_1.rgb', 'observation.camera_2.rgb']
LOW = [.15, -.45, 0., -1., -1., -1., -1., 0.]
HIGH = [.95, .25, .95, 1., 1., 1., 1., .04]

def make_config(device):
    return DiffusionConfig(
        device=device, n_obs_steps=2, horizon=16, n_action_steps=8,
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (8,)),
                        **{k: PolicyFeature(FeatureType.VISUAL, (3,256,256)) for k in CAMERAS}},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        down_dims=(64,128,256), diffusion_step_embed_dim=64,
        crop_shape=None, resize_shape=None, crop_is_random=False,
        pretrained_backbone_weights=None, num_train_timesteps=100, num_inference_steps=100,
        do_mask_loss_for_padding=False, drop_n_last_frames=0)

def make_normalization(episode):
    norm = {'low':LOW, 'high':HIGH, 'quaternion':'xyzw identity [-1,1]',
            'source':'tasks/task/grasp.yaml eeSafeWorkspace; stacking z lower bound 0; URDF finger [0,.04]', 'images':{}}
    for i,key in enumerate(CAMERAS,1):
        rgb = episode[f'camera_{i}_rgb'].astype(np.float32)/255
        norm['images'][key] = {'mean':rgb.mean((0,1,2)).tolist(),
                               'std':np.maximum(rgb.std((0,1,2)),1e-6).tolist()}
    return norm

def vector_transform(values, norm, inverse=False):
    low = torch.as_tensor(norm['low'],device=values.device,dtype=values.dtype)
    high = torch.as_tensor(norm['high'],device=values.device,dtype=values.dtype)
    return (values+1)*.5*(high-low)+low if inverse else 2*(values-low)/(high-low)-1

def observation_tensors(state, images, norm, device):
    out = {'observation.state':vector_transform(torch.as_tensor(np.ascontiguousarray(state),dtype=torch.float32,device=device),norm)}
    for key,image in zip(CAMERAS,images):
        rgb = torch.as_tensor(np.ascontiguousarray(image),device=device).float().permute(0,3,1,2)/255
        stats = norm['images'][key]
        mean = torch.tensor(stats['mean'],device=device).reshape(1,3,1,1)
        std = torch.tensor(stats['std'],device=device).reshape(1,3,1,1)
        out[key] = (rgb-mean)/std
    return out

class Inference:
    def __init__(self,checkpoint,device='cuda',seed=0):
        checkpoint = Path(checkpoint)
        # The generic loader selects the diffusion subclass from config.json.
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        self.modality=json.loads((checkpoint/'run_config.json').read_text()).get('modality','rgb')
        policy_class=DiffusionPolicy
        if self.modality=='pointcloud':
            from pointcloud_diffusion import PointCloudPolicy
            policy_class=PointCloudPolicy
        self.policy = policy_class.from_pretrained(checkpoint,config=config,strict=True).to(device).eval()
        self.norm = json.loads((checkpoint/'normalization.json').read_text())
        self.device,self.seed = device,seed
        self.reset()

    def reset(self,seed=None):
        if seed is not None: self.seed = int(seed)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.history,self.actions = deque(maxlen=2),deque()
        self.previous_quaternion = None
        self.inference_calls = self.clip_count = self.quaternion_fallbacks = 0

    @torch.inference_mode()
    def predict(self,state,images=None,point_cloud=None):
        if self.modality=='pointcloud':
            from rgbd_geometry import normalize_points
            points=np.asarray(point_cloud,dtype=np.float32)
            if points.shape!=(len(state),1024,3) or not np.isfinite(points).all():
                raise ValueError('Expected measured, finite point cloud (B,1024,3)')
            obs={'observation.state':vector_transform(torch.as_tensor(state,device=self.device,dtype=torch.float32),self.norm),
                 'observation.point_cloud':torch.as_tensor(normalize_points(points),device=self.device)}
        else:
            obs = observation_tensors(state,images,self.norm,self.device)
        self.history.append(obs)
        if len(self.history)==1: self.history.append(obs)
        if not self.actions:
            states = torch.stack([v['observation.state'] for v in self.history],dim=1)
            dm = self.policy.diffusion
            if self.modality=='pointcloud':
                pc=torch.stack([v['observation.point_cloud'] for v in self.history],dim=1)
                cond=dm._prepare_global_conditioning({'observation.state':states,'observation.point_cloud':pc})
            else:
                rgb = torch.stack([torch.stack([v[k] for k in CAMERAS],dim=1) for v in self.history],dim=1)
                cond = dm._prepare_global_conditioning({'observation.state':states,'observation.images':rgb})
            # Controls both initial Gaussian noise and every DDPM variance sample.
            full = dm.conditional_sample(states.shape[0],global_cond=cond,generator=self.rng)
            chunk = vector_transform(full[:,1:9],self.norm,inverse=True).cpu().numpy()
            self.actions.extend(chunk.transpose(1,0,2))
            self.inference_calls += 1
        raw = self.actions.popleft().copy()
        if not np.isfinite(raw).all(): raise ValueError('Non-finite diffusion output')
        safe = np.clip(raw,np.array(self.norm['low']),np.array(self.norm['high'])).astype(np.float32)
        clipped = bool(np.any(np.abs(safe-raw)>1e-7))
        self.clip_count += int(clipped)
        q = safe[:,3:7]
        length = np.linalg.norm(q,axis=-1,keepdims=True)
        bad = length[:,0]<1e-6
        q /= np.maximum(length,1e-6)
        if bad.any():
            q[bad] = [1,0,0,0] if self.previous_quaternion is None else self.previous_quaternion[bad]
            self.quaternion_fallbacks += int(bad.sum())
        if self.previous_quaternion is not None:
            q[np.sum(q*self.previous_quaternion,axis=-1)<0] *= -1
        self.previous_quaternion = q.copy()
        return safe, {'raw_action':raw,'action_clipped':clipped,'quaternion_fallback':bool(bad.any()),
                       'inference_calls':self.inference_calls}
