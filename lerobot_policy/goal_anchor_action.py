"""Learned goal-anchor weights plus residual actions; no runtime phase input.

Training-only anchor labels describe the frozen expert's transport interval.
At inference the diffusion model predicts the anchor weight for each target.
"""
import torch
from relative_action import SCALE
from delivery_policy_contract import NORMALIZATION
from diffusion_overfit_common import vector_transform
GOAL_ANCHOR_NORMALIZATION=dict(low=[-v for v in SCALE]+[-1.,-1.,-1.,-1.,0.,0.],
    high=SCALE+[1.,1.,1.,1.,.04,1.],quaternion='xyzw absolute FR3 base',
    source='XYZ residual around learned mixture of planning TCP and measured initial target; finger absolute; ninth dimension learned anchor weight')


def goal_anchor_targets(batch,frame):
    anchor=vector_transform(batch['observation.state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
    goal=batch['observation.goal_features'][:,-1,:3]
    valid=batch['observation.goal_features'][:,-1,6]
    future=(frame[:,None]+torch.arange(-1,15,device=frame.device)).clamp(0,39)
    # Collection phase contract: lift ends at19, transport20..26, place27 onward.
    alpha=((future-19)/7).clamp(0,1).to(anchor.dtype)*valid[:,None]
    target=vector_transform(batch['action'],NORMALIZATION,inverse=True)
    target=target.clone();target[:,:,:3]-=anchor[:,None]+alpha[:,:,None]*(goal-anchor)[:,None]
    out=dict(batch);out['action']=vector_transform(torch.cat([target,alpha[:,:,None]],dim=-1),GOAL_ANCHOR_NORMALIZATION)
    return out


def decode_goal_actions(normalized,anchor,goal,valid):
    target=vector_transform(normalized,GOAL_ANCHOR_NORMALIZATION,inverse=True)
    alpha=target[:,:,8:9].clamp(0,1)*valid[:,None,None]
    action=target[:,:,:8].clone();action[:,:,:3]+=anchor[:,None]+alpha*(goal-anchor)[:,None]
    return action
