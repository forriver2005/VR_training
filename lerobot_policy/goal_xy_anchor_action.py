"""Learned horizontal goal anchor; vertical actions remain TCP-relative.

Keeping Z separate prevents a target at another height from pulling a lifted
object down during an early, still goal-masked transport block.
"""
import torch
from relative_action import SCALE
from delivery_policy_contract import NORMALIZATION
from diffusion_overfit_common import vector_transform
NORMALIZATION_XY=dict(low=[-v for v in SCALE]+[-1.,-1.,-1.,-1.,0.,0.],high=SCALE+[1.,1.,1.,1.,.04,1.],
    quaternion='xyzw absolute FR3 base',source='XY residual around learned TCP/observed-goal mixture; Z relative to planning TCP; ninth channel anchor weight')

def targets(batch,frame):
    anchor=vector_transform(batch['observation.state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
    goal=batch['observation.goal_features'][:,-1,:3];valid=batch['observation.goal_features'][:,-1,6]
    future=(frame[:,None]+torch.arange(-1,15,device=frame.device)).clamp(0,39)
    alpha=((future-19)/7).clamp(0,1).to(anchor.dtype)*valid[:,None]
    target=vector_transform(batch['action'],NORMALIZATION,inverse=True).clone()
    target[:,:,:3]-=anchor[:,None]
    target[:,:,:2]-=alpha[:,:,None]*(goal-anchor)[:,None,:2]
    out=dict(batch);out['action']=vector_transform(torch.cat([target,alpha[:,:,None]],dim=-1),NORMALIZATION_XY)
    return out

def decode(normalized,anchor,goal,valid):
    target=vector_transform(normalized,NORMALIZATION_XY,inverse=True)
    alpha=target[:,:,8:9].clamp(0,1)*valid[:,None,None]
    result=target[:,:,:8].clone();result[:,:,:3]+=anchor[:,None]
    result[:,:,:2]+=alpha*(goal-anchor)[:,None,:2]
    return result
