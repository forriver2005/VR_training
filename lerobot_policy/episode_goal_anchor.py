"""Episode-fixed robot/goal XY anchors; Z relative to measured initial robot TCP.

Targets no longer change just because a new planning block starts from a slightly
different actual pose. Both anchors come from observations, not reset commands.
"""
import torch
from relative_action import SCALE
from delivery_policy_contract import NORMALIZATION
from diffusion_overfit_common import vector_transform
EPISODE_NORMALIZATION=dict(low=[-v for v in SCALE]+[-1.,-1.,-1.,-1.,0.,0.],high=SCALE+[1.,1.,1.,1.,.04,1.],
    quaternion='xyzw absolute FR3 base',source='episode initial measured TCP / measured RGB-D goal XY anchors; Z relative initial measured TCP; learned ninth anchor weight')

def targets(batch,frame):
    initial=vector_transform(batch['observation.reset_state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
    goal=batch['observation.goal_features'][:,-1,:3];valid=batch['observation.goal_features'][:,-1,6]
    future=(frame[:,None]+torch.arange(-1,15,device=frame.device)).clamp(0,39)
    alpha=((future-19)/7).clamp(0,1).to(initial.dtype)*valid[:,None]
    target=vector_transform(batch['action'],NORMALIZATION,inverse=True).clone()
    target[:,:,:3]-=initial[:,None]
    target[:,:,:2]-=alpha[:,:,None]*(goal-initial)[:,None,:2]
    out=dict(batch);out['action']=vector_transform(torch.cat([target,alpha[:,:,None]],dim=-1),EPISODE_NORMALIZATION)
    return out

def decode(normalized,initial,goal,valid):
    target=vector_transform(normalized,EPISODE_NORMALIZATION,inverse=True)
    alpha=target[:,:,8:9].clamp(0,1)*valid[:,None,None]
    result=target[:,:,:8].clone();result[:,:,:3]+=initial[:,None]
    result[:,:,:2]+=alpha*(goal-initial)[:,None,:2]
    return result
