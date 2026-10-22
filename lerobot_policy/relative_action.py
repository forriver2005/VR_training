"""Optional XYZ residual at the replanning TCP; orientations/fingers stay absolute."""
import torch
from delivery_policy_contract import NORMALIZATION
from diffusion_overfit_common import vector_transform

SCALE=[hi-lo for lo,hi in zip(NORMALIZATION['low'][:3],NORMALIZATION['high'][:3])]
RELATIVE_NORMALIZATION=dict(low=[-v for v in SCALE]+[-1.,-1.,-1.,-1.,0.],
    high=SCALE+[1.,1.,1.,1.,.04],quaternion='xyzw, absolute in FR3 base',
    source='XYZ displacement from current planning observation TCP; fixed full task spans; finger absolute metres')

def relative_targets(batch):
    """Transform labels only; condition observations remain exactly unchanged."""
    anchor=vector_transform(batch['observation.state'][:,-1],NORMALIZATION,inverse=True)[:,:3]
    target=vector_transform(batch['action'],NORMALIZATION,inverse=True)
    target=target.clone();target[:,:,:3]-=anchor[:,None,:]
    out=dict(batch);out['action']=vector_transform(target,RELATIVE_NORMALIZATION)
    return out

def decode_actions(normalized_actions,anchor,normalization):
    """Add the one planning anchor once, before putting absolute targets in queue."""
    target=vector_transform(normalized_actions,normalization,inverse=True)
    target=target.clone();target[:,:,:3]+=anchor[:,None,:]
    return target
