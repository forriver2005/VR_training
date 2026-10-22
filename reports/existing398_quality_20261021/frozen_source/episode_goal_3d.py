"""Episode anchors with delayed vertical goal blending, entirely network-driven."""
import torch
from relative_action import SCALE
from delivery_policy_contract import NORMALIZATION
from diffusion_overfit_common import vector_transform
EPISODE_3D_NORMALIZATION=dict(low=[-v for v in SCALE]+[-1.,-1.,-1.,-1.,0.,0.],high=SCALE+[1.,1.,1.,1.,.04,1.],
    quaternion='xyzw absolute FR3 base',source='fixed initial measured TCP / measured goal anchors; XY alpha, Z beta=clip((7alpha-4)/3); ninth learned alpha')

def blend(alpha):
    beta=((7*alpha-4)/3).clamp(0,1)
    return torch.stack([alpha,alpha,beta],dim=-1)

def targets(batch,frame):
    initial=vector_transform(batch['observation.reset_state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
    goal=batch['observation.goal_features'][:,-1,:3];valid=batch['observation.goal_features'][:,-1,6]
    future=(frame[:,None]+torch.arange(-1,15,device=frame.device)).clamp(0,39)
    alpha=((future-19)/7).clamp(0,1).to(initial.dtype)*valid[:,None]
    target=vector_transform(batch['action'],NORMALIZATION,inverse=True).clone()
    target[:,:,:3]-=initial[:,None]+blend(alpha)*(goal-initial)[:,None]
    out=dict(batch);out['action']=vector_transform(torch.cat([target,alpha[:,:,None]],dim=-1),EPISODE_3D_NORMALIZATION)
    return out

def decode(normalized,initial,goal,valid):
    target=vector_transform(normalized,EPISODE_3D_NORMALIZATION,inverse=True)
    alpha=target[:,:,8].clamp(0,1)*valid[:,None]
    result=target[:,:,:8].clone();result[:,:,:3]+=initial[:,None]+blend(alpha)*(goal-initial)[:,None]
    return result
