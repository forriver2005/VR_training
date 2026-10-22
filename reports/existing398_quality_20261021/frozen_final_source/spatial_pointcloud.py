"""Measured XYZ spatial tokens and a common planning-TCP observation frame.

This is a local experimental encoder, not a reproduction of RISE or iDP3.
No object poses, semantic masks, phase labels or future observations are inputs.
"""
import torch
from torch import nn
from delivery_policy_contract import CROP,NORMALIZATION


def tcp_centered_inputs(points,state):
    """Both history frames use the current planning TCP, preserving relative motion."""
    crop=torch.as_tensor(CROP,device=points.device,dtype=points.dtype)
    lo=torch.as_tensor(NORMALIZATION['low'],device=state.device,dtype=state.dtype)
    hi=torch.as_tensor(NORMALIZATION['high'],device=state.device,dtype=state.dtype)
    xyz=(state[...,:3]+1)*.5*(hi[:3]-lo[:3])+lo[:3]
    anchor=xyz[:,-1:,:]
    metric=(points+1)*.5*(crop[1]-crop[0])+crop[0]
    pc=(metric-anchor[:,:,None,:])/(crop[1]-crop[0])
    s=state.clone();s[...,:3]=(xyz-anchor)/(hi[:3]-lo[:3])
    return pc,s


class SpatialPointEncoder(nn.Module):
    """128 local neighborhoods, positional tokens, two attention layers, 64D output."""
    def __init__(self):
        super().__init__()
        self.point_mlp=nn.Sequential(nn.Linear(3,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,128),nn.LayerNorm(128),nn.ReLU())
        self.local=nn.Sequential(nn.Linear(131,128),nn.LayerNorm(128),nn.ReLU())
        self.position=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128))
        self.readout=nn.Parameter(torch.zeros(1,1,128));nn.init.normal_(self.readout,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=256,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.projection=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.LayerNorm(64))

    def forward(self,points):
        b,n,d=points.shape
        if n!=1024 or d!=3:raise ValueError('Spatial encoder expects 1024 measured XYZ points')
        # Fixed coverage across incoming deterministic FPS samples, no object-based sampling.
        idx=torch.linspace(0,n-1,128,device=points.device).long()
        anchors=points[:,idx]
        with torch.no_grad():neighbors=torch.cdist(anchors,points).topk(16,largest=False,sorted=True).indices
        batch=torch.arange(b,device=points.device)[:,None,None]
        xyz=points[batch,neighbors]-anchors[:,:,None]
        feature=self.point_mlp(points)[batch,neighbors]
        token=self.local(torch.cat([feature,xyz],dim=-1)).amax(dim=2)+self.position(anchors)
        token=torch.cat([self.readout.expand(b,-1,-1),token],dim=1)
        return self.projection(self.attention(token)[:,0])
