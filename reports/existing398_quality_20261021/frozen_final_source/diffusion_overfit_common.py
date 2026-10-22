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
        self.contract=json.loads((checkpoint/'run_config.json').read_text())
        self.modality=self.contract.get('modality','rgb')
        self.delivery=self.contract.get('protocol_version')=='delivery_4hz_v1'
        if self.delivery:
            from delivery_policy_contract import CROP,POINT_SOURCES
            geometry=self.contract['pointcloud_contract']
            if geometry.get('source') not in POINT_SOURCES or geometry.get('fps')!=POINT_SOURCES[geometry['source']]:
                raise ValueError('Checkpoint point-cloud recipe is unsupported')
            if not np.array_equal(np.asarray(geometry['crop']),CROP):
                raise ValueError('Checkpoint crop differs from online normalization')
        self.history_validity='observation.history_valid' in config.input_features
        self.hold_timer='observation.elapsed_same_target_s' in config.input_features
        self.goal_memory='observation.goal_features' in config.input_features
        self.reset_memory='observation.reset_state' in config.input_features
        if self.goal_memory:
            from goal_observation_registry import goal_estimator
            from delivery_policy_contract import sha256
            _,_,goal_code=goal_estimator(self.contract.get('goal_memory',{}).get('recipe'))
            expected_goal_hash=self.contract['goal_memory'].get('code_sha256')
            if expected_goal_hash and sha256(goal_code)!=expected_goal_hash:raise ValueError('Goal extractor differs from checkpoint contract')
        self.action_steps=config.n_action_steps
        self.action_samples=int(self.contract.get('action_samples',1))
        if self.action_samples<1 or self.action_samples>16:raise ValueError('Invalid action ensemble size')
        self.binary_gripper=bool(self.contract.get('binary_gripper',False))
        self.fixed_down_quaternion=bool(self.contract.get('fixed_down_quaternion',False))
        self.goal_terminal_constraints=bool(self.contract.get('goal_terminal_constraints',False))
        if self.goal_terminal_constraints:
            if not self.goal_memory or 'anchor' not in self.contract.get('position_action_mode',''):raise ValueError('Goal terminal constraints require measured-goal anchor actions')
            self.stack_height_offset=.5*(float(self.contract['cube_a_size_m'])+float(self.contract['cube_b_size_m']))
        self.release_guard_config=self.contract.get('release_guard')
        if self.release_guard_config:
            if not (self.goal_terminal_constraints and self.binary_gripper and self.fixed_down_quaternion):raise ValueError('Release guard requires verified terminal geometry and actuator constraints')
            from delivery_policy_contract import sha256
            if sha256(Path(__file__).with_name('release_guard.py'))!=self.release_guard_config.get('code_sha256'):raise ValueError('Release guard differs from checkpoint contract')
        self.anchor_snap_epsilon=float(self.contract.get('anchor_snap_epsilon',0.))
        if not 0<=self.anchor_snap_epsilon<=.05:raise ValueError('Invalid anchor endpoint tolerance')
        if self.anchor_snap_epsilon and 'anchor' not in self.contract.get('position_action_mode',''):raise ValueError('Anchor snapping requires an anchor action model')
        policy_class=DiffusionPolicy
        policy_kwargs={}
        if self.modality=='pointcloud':
            from pointcloud_diffusion import PointCloudPolicy
            policy_class=PointCloudPolicy
            policy_kwargs=dict(encoder_type=self.contract.get('encoder_type','pointnet'),
                observation_frame=self.contract.get('observation_frame','base'),learned_goal_gate=self.contract.get('learned_goal_gate',False),
                position_loss_weight=self.contract.get('position_loss_weight',1.),goal_state_switch=self.contract.get('goal_state_switch',False))
        self.policy = policy_class.from_pretrained(checkpoint,config=config,strict=True,**policy_kwargs).to(device).eval()
        self.monotonic_goal_gate=bool(self.contract.get('monotonic_goal_gate',False))
        if self.monotonic_goal_gate and not getattr(self.policy.diffusion,'learned_goal_gate',False):raise ValueError('Monotonic memory requires the learned transport classifier')
        self.norm = json.loads((checkpoint/'normalization.json').read_text())
        self.relative_actions=self.contract.get('position_action_mode','absolute_base')=='relative_to_current_tcp'
        self.goal_anchor_actions=self.contract.get('position_action_mode')=='goal_anchor_residual'
        self.goal_xy_anchor_actions=self.contract.get('position_action_mode')=='goal_xy_anchor_residual'
        self.episode_goal_actions=self.contract.get('position_action_mode')=='episode_goal_xy_anchor'
        self.episode_goal_3d_actions=self.contract.get('position_action_mode')=='episode_goal_3d_anchor'
        if self.episode_goal_3d_actions:
            from episode_goal_3d import EPISODE_3D_NORMALIZATION
            if not self.goal_memory or not self.reset_memory or self.norm!=EPISODE_3D_NORMALIZATION:raise ValueError('3D episode goal-anchor contract mismatch')
        if self.episode_goal_actions:
            from episode_goal_anchor import EPISODE_NORMALIZATION
            if not self.goal_memory or not self.reset_memory or self.norm!=EPISODE_NORMALIZATION:raise ValueError('Episode goal-anchor contract mismatch')
        if self.goal_xy_anchor_actions:
            from goal_xy_anchor_action import NORMALIZATION_XY
            if not self.goal_memory or self.norm!=NORMALIZATION_XY:raise ValueError('XY goal-anchor contract mismatch')
        if self.goal_anchor_actions:
            from goal_anchor_action import GOAL_ANCHOR_NORMALIZATION
            if not self.goal_memory or self.norm!=GOAL_ANCHOR_NORMALIZATION:raise ValueError('Goal-anchor contract mismatch')
        self.observation_norm=self.contract.get('observation_normalization',self.norm)
        self.action_limits=self.contract.get('action_limits',self.norm)
        if self.relative_actions:
            from relative_action import RELATIVE_NORMALIZATION
            if self.norm!=RELATIVE_NORMALIZATION:raise ValueError('Relative action normalization mismatch')
        self.device,self.seed = device,seed
        self.reset()

    def reset(self,seed=None):
        if seed is not None: self.seed = int(seed)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.history,self.actions = deque(maxlen=2),deque()
        self.planned_anchor_weights=deque()
        self.history_valid=deque(maxlen=2)
        self.previous_quaternion = None
        self.initial_observation_state=None
        self.inference_calls = self.clip_count = self.quaternion_fallbacks = 0
        self.gripper_projection_count=self.orientation_projection_count=0
        self.anchor_projection_count=0
        self.goal_terminal_projection_count=0
        self.last_goal_gate_probability=np.asarray([-1.],dtype=np.float32)
        self.last_effective_goal_gate=np.asarray([-1.],dtype=np.float32)
        self.policy.diffusion.monotonic_goal_gate=getattr(self,'monotonic_goal_gate',False)
        self.policy.diffusion.inference_goal_gate_latch=None
        self.release_guard=None
        if getattr(self,'release_guard_config',None):
            from release_guard import ReleaseGuard
            self.release_guard=ReleaseGuard(self.release_guard_config,self.stack_height_offset)

    @torch.inference_mode()
    def predict(self,state,images=None,point_cloud=None,elapsed_same_target_s=None,goal_features=None):
        if self.modality=='pointcloud':
            if self.delivery:
                from delivery_policy_contract import normalize_points
            else:
                from rgbd_geometry import normalize_points
            points=np.asarray(point_cloud,dtype=np.float32)
            if points.shape!=(len(state),1024,3) or not np.isfinite(points).all():
                raise ValueError('Expected measured, finite point cloud (B,1024,3)')
            obs={'observation.state':vector_transform(torch.as_tensor(state,device=self.device,dtype=torch.float32),self.observation_norm),
                 'observation.point_cloud':torch.as_tensor(normalize_points(points),device=self.device)}
            if getattr(self,'reset_memory',False):
                if self.initial_observation_state is None:self.initial_observation_state=obs['observation.state'].clone()
                obs['observation.reset_state']=self.initial_observation_state
            if getattr(self,'goal_memory',False):
                if goal_features is None:raise ValueError('Checkpoint requires measured RGB-D goal memory')
                goal=np.asarray(goal_features,dtype=np.float32)
                if goal.shape!=(len(state),8) or not np.isfinite(goal).all():raise ValueError('Invalid goal features')
                obs['observation.goal_features']=torch.as_tensor(goal,device=self.device)
            if self.hold_timer:
                if elapsed_same_target_s is None: raise ValueError('Checkpoint requires an observed command-history timer')
                timer=np.asarray(elapsed_same_target_s,dtype=np.float32).reshape(len(state),1)
                if not np.isfinite(timer).all() or (timer<0).any(): raise ValueError('Invalid hold timer')
                timer=np.clip(timer/self.contract['timer']['normalization_seconds'],0,1)
                obs['observation.elapsed_same_target_s']=torch.as_tensor(timer,device=self.device)
        else:
            obs = observation_tensors(state,images,self.norm,self.device)
        self.history.append(obs)
        self.history_valid.append(True)
        if len(self.history)==1:
            self.history.append(obs)
            self.history_valid.appendleft(False)
        if not self.actions:
            states = torch.stack([v['observation.state'] for v in self.history],dim=1)
            dm = self.policy.diffusion
            if self.modality=='pointcloud':
                pc=torch.stack([v['observation.point_cloud'] for v in self.history],dim=1)
                condition={'observation.state':states,'observation.point_cloud':pc}
                if getattr(self,'reset_memory',False):
                    condition['observation.reset_state']=torch.stack([v['observation.reset_state'] for v in self.history],dim=1)
                if getattr(self,'goal_memory',False):
                    condition['observation.goal_features']=torch.stack([v['observation.goal_features'] for v in self.history],dim=1)
                if self.history_validity:
                    condition['observation.history_valid']=torch.tensor(list(self.history_valid),device=self.device,dtype=states.dtype).reshape(1,2,1).expand(states.shape[0],-1,-1)
                if self.hold_timer:
                    condition['observation.elapsed_same_target_s']=torch.stack([v['observation.elapsed_same_target_s'] for v in self.history],dim=1)
                cond=dm._prepare_global_conditioning(condition)
                if getattr(dm,'learned_goal_gate',False):
                    self.last_goal_gate_probability=dm.last_phase_logits.sigmoid()[:,0].cpu().numpy()
                    self.last_effective_goal_gate=dm.last_effective_goal_gate[:,0].cpu().numpy()
            else:
                rgb = torch.stack([torch.stack([v[k] for k in CAMERAS],dim=1) for v in self.history],dim=1)
                cond = dm._prepare_global_conditioning({'observation.state':states,'observation.images':rgb})
            # Controls both initial Gaussian noise and every DDPM variance sample.
            samples=getattr(self,'action_samples',1)
            if samples==1:full = dm.conditional_sample(states.shape[0],global_cond=cond,generator=self.rng)
            else:
                full=dm.conditional_sample(states.shape[0]*samples,global_cond=cond.repeat_interleave(samples,dim=0),generator=self.rng)
                full=full.reshape(states.shape[0],samples,*full.shape[1:]).mean(dim=1)
            epsilon=getattr(self,'anchor_snap_epsilon',0.)
            if epsilon:
                alpha=full[:,:,8]
                projected=torch.where(alpha<=-1+2*epsilon,-torch.ones_like(alpha),torch.where(alpha>=1-2*epsilon,torch.ones_like(alpha),alpha))
                self.anchor_projection_count+=int((projected[:,1:1+self.action_steps]!=alpha[:,1:1+self.action_steps]).sum())
                full[:,:,8]=projected
            if getattr(self,'episode_goal_3d_actions',False):
                from episode_goal_3d import decode
                initial=vector_transform(condition['observation.reset_state'][:,-1],self.observation_norm,inverse=True)[:,:3]
                goals=condition['observation.goal_features'][:,-1]
                chunk=decode(full[:,1:1+self.action_steps],initial,goals[:,:3],goals[:,6]).cpu().numpy()
            elif getattr(self,'episode_goal_actions',False):
                from episode_goal_anchor import decode
                initial=vector_transform(condition['observation.reset_state'][:,-1],self.observation_norm,inverse=True)[:,:3]
                goals=condition['observation.goal_features'][:,-1]
                chunk=decode(full[:,1:1+self.action_steps],initial,goals[:,:3],goals[:,6]).cpu().numpy()
            elif getattr(self,'goal_xy_anchor_actions',False):
                from goal_xy_anchor_action import decode
                anchor=vector_transform(states[:,-1],self.observation_norm,inverse=True)[:,:3]
                goals=condition['observation.goal_features'][:,-1]
                chunk=decode(full[:,1:1+self.action_steps],anchor,goals[:,:3],goals[:,6]).cpu().numpy()
            elif getattr(self,'goal_anchor_actions',False):
                from goal_anchor_action import decode_goal_actions
                anchor=vector_transform(states[:,-1],self.observation_norm,inverse=True)[:,:3]
                goals=condition['observation.goal_features'][:,-1]
                chunk=decode_goal_actions(full[:,1:1+self.action_steps],anchor,goals[:,:3],goals[:,6]).cpu().numpy()
            elif self.relative_actions:
                from relative_action import decode_actions
                anchor=vector_transform(states[:,-1],self.observation_norm,inverse=True)[:,:3]
                chunk=decode_actions(full[:,1:1+self.action_steps],anchor,self.norm).cpu().numpy()
            else:
                chunk = vector_transform(full[:,1:1+self.action_steps],self.norm,inverse=True).cpu().numpy()
            if getattr(self,'goal_terminal_constraints',False):
                goal_values=condition['observation.goal_features'][:,-1].cpu().numpy()
                alpha_used=(((full[:,1:1+self.action_steps,8]+1)*.5).clamp(0,1)*condition['observation.goal_features'][:,-1,6:7]).cpu().numpy()
                terminal=alpha_used>=1-1e-6
                before=chunk[:,:,:3].copy()
                chunk[:,:,:2]=np.where(terminal[:,:,None],goal_values[:,None,:2],chunk[:,:,:2])
                minimum=goal_values[:,None,2]+np.float32(self.stack_height_offset)
                chunk[:,:,2]=np.where(terminal,np.maximum(chunk[:,:,2],minimum),chunk[:,:,2])
                self.goal_terminal_projection_count+=int(np.sum(terminal&np.any(np.abs(chunk[:,:,:3]-before)>1e-7,axis=-1)))
            self.actions.extend(chunk.transpose(1,0,2))
            if full.shape[-1]==9 and getattr(self,'goal_memory',False):
                executed_alpha=((full[:,1:1+self.action_steps,8]+1)*.5).clamp(0,1)*condition['observation.goal_features'][:,-1,6:7]
                self.planned_anchor_weights.extend(executed_alpha.transpose(0,1).cpu().numpy())
            else:self.planned_anchor_weights.extend([np.full(states.shape[0],-1.,dtype=np.float32) for _ in range(self.action_steps)])
            self.inference_calls += 1
        raw = self.actions.popleft().copy()
        anchor_weight=self.planned_anchor_weights.popleft()
        if getattr(self,'monotonic_goal_gate',False):
            # Only an action actually returned for execution can set this bit.
            # Unexecuted future predictions do not advance episode memory.
            triggered=torch.as_tensor(anchor_weight>1/14,device=self.device).reshape(-1,1)
            previous=self.policy.diffusion.inference_goal_gate_latch
            self.policy.diffusion.inference_goal_gate_latch=triggered if previous is None else (triggered|previous)
        if not np.isfinite(raw).all(): raise ValueError('Non-finite diffusion output')
        safe = np.clip(raw,np.array(self.action_limits['low']),np.array(self.action_limits['high'])).astype(np.float32)
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
        projected_gripper=projected_orientation=False
        if getattr(self,'fixed_down_quaternion',False):
            projected_orientation=bool(np.any(np.abs(q-np.array([1.,0.,0.,0.]))>1e-6))
            q[:]=[1.,0.,0.,0.]
            self.orientation_projection_count+=int(projected_orientation)
        if getattr(self,'binary_gripper',False):
            projected=np.where(safe[:,7]<.028,.016,.04).astype(np.float32)
            projected_gripper=bool(np.any(np.abs(safe[:,7]-projected)>1e-6))
            safe[:,7]=projected;self.gripper_projection_count+=int(projected_gripper)
        guard_diag={}
        if self.release_guard is not None:
            latch=self.policy.diffusion.inference_goal_gate_latch
            started=latch[:,0].cpu().numpy() if latch is not None else self.last_effective_goal_gate>.5
            safe,guard_diag=self.release_guard.apply(safe,state,goal_features,started,anchor_weight)
            bounded=np.clip(safe,np.asarray(self.action_limits['low']),np.asarray(self.action_limits['high'])).astype(np.float32)
            extra_clip=bool(np.any(np.abs(bounded-safe)>1e-7))
            if extra_clip and not clipped:self.clip_count+=1
            clipped=clipped or extra_clip;safe=bounded
        self.previous_quaternion = safe[:,3:7].copy()
        return safe, {'raw_action':raw,'action_clipped':clipped,'quaternion_fallback':bool(bad.any()),
                       'inference_calls':self.inference_calls,'gripper_projected':projected_gripper,'orientation_projected':projected_orientation,
                       'anchor_projection_count':self.anchor_projection_count,'action_samples':getattr(self,'action_samples',1),
                       'goal_terminal_projection_count':self.goal_terminal_projection_count,
                       'goal_gate_probability':self.last_goal_gate_probability,'effective_goal_gate':self.last_effective_goal_gate,
                       'executed_anchor_weight':anchor_weight,**guard_diag}
