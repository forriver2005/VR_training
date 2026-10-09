#!/usr/bin/env python3
"""Train one recorded RGB trajectory with per-step EMA and resumable checkpoints."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from diffusion_overfit_common import RAW, make_config, make_normalization, vector_transform, observation_tensors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from pointcloud_diffusion import PointCloudPolicy
from rgbd_geometry import geometry_contract, normalize_points
from lerobot.configs.types import FeatureType, PolicyFeature

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=RAW)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--resume',type=Path)
    p.add_argument('--modality',choices=['rgb','pointcloud'],default='rgb')
    p.add_argument('--window-sampling',choices=['all','replan'],default='all',
                   help='all 40 anchors or the five anchors used by A=8 online replanning')
    p.add_argument('--history-validity',action='store_true',help='mark padded versus measured observation history')
    p.add_argument('--prediction-type',choices=['epsilon','sample'],default='epsilon',
                   help='diffusion target: noise or clean normalized action; record as an independent ablation')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=args.resume is not None)
    with np.load(args.source) as data: episode = {k:data[k] for k in data.files}
    if episode['state'].shape!=(40,8) or episode['action'].shape!=(40,8):
        raise ValueError('Exactly one 40-frame episode is required')
    norm = make_normalization(episode)
    if args.modality=='pointcloud':
        if not bool(episode.get('success',False)):
            raise ValueError('Point-cloud overfit requires a physically successful replay')
        times=episode['observation_time_s']
        control_dt=float(episode['control_dt_s'])
        if times.shape!=(40,) or not np.allclose(np.diff(times),control_dt,rtol=0,atol=1e-6):
            raise ValueError('Observation timestamps do not match control period')
        if episode['point_cloud'].shape!=(40,1024,3) or not np.isfinite(episode['point_cloud']).all():
            raise ValueError('Expected finite point cloud (40,1024,3)')
        obs={'observation.state':vector_transform(torch.tensor(episode['state'],device=args.device),norm),
             'observation.point_cloud':torch.tensor(normalize_points(episode['point_cloud']),device=args.device)}
    else:
        obs = observation_tensors(episode['state'],[episode['camera_1_rgb'],episode['camera_2_rgb']],norm,args.device)
    actions = vector_transform(torch.tensor(episode['action'],device=args.device),norm)
    if actions.abs().max()>1.00001: raise ValueError('Expert actions exceed fixed bounds')
    config=make_config(args.device)
    config.prediction_type=args.prediction_type
    if args.modality=='pointcloud':
        config.input_features={'observation.state':config.input_features['observation.state']}
        if args.history_validity:
            config.input_features['observation.history_valid']=PolicyFeature(FeatureType.STATE,(1,))
    elif args.history_validity:
        raise ValueError('History validity currently supported only for pointcloud')
    model = (PointCloudPolicy(config) if args.modality=='pointcloud' else DiffusionPolicy(config)).to(args.device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    def lr_factor(step):
        if step<500: return (step+1)/500
        return .5*(1+math.cos(math.pi*min(1.,(step-500)/max(1,args.steps-500))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    start = 0
    if args.resume:
        saved = torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['source_sha256']!=source_hash or saved['normalization']!=norm:
            raise ValueError('Resume dataset/normalization mismatch')
        if saved.get('modality','rgb')!=args.modality or saved.get('steps',args.steps)!=args.steps:
            raise ValueError('Resume modality/training schedule mismatch')
        if saved.get('window_sampling','all')!=args.window_sampling:
            raise ValueError('Resume window sampling mismatch')
        if saved.get('history_validity',False)!=args.history_validity:
            raise ValueError('Resume history validity mismatch')
        if saved.get('prediction_type','epsilon')!=args.prediction_type:
            raise ValueError('Resume diffusion target mismatch')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); start=saved['step']
    contract = dict(source=str(args.source.resolve()),source_sha256=source_hash,steps=args.steps,
        batch_size=args.batch_size,seed=args.seed,H=2,K=16,A=8,frames=40,
        windows=40 if args.window_sampling=='all' else 5,window_sampling=args.window_sampling,
        history_validity=args.history_validity,
        prediction_type=args.prediction_type,
        window_anchors=list(range(40)) if args.window_sampling=='all' else list(range(0,40,8)),
        observation_indices=[-1,0],action_indices=list(range(-1,15)),padding='repeat boundary frames',
        augmentation=False,crop=False,network='shared ResNet18 GroupNorm + UNet [64,128,256]',
        ema_power=.75,ddpm_steps=100,optimizer='AdamW lr=1e-4 betas=(.95,.999) wd=1e-6',
        normalization=norm,device=args.device,modality=args.modality,
        pointcloud_contract='XYZ FR3 base; fixed crop [.15,-.45,0] to [.95,.25,.95]; deterministic FPS 1024')
    if args.modality=='pointcloud':
        contract['network']='PointNet [64,128,256] LN ->64; state MLP 8->64->64; UNet [64,128,256]'
        contract['pointcloud_contract']=geometry_contract()
        contract['control_dt_s']=control_dt
        source_manifest=args.source.parent/'replay_manifest.json'
        contract['replay_manifest']=str(source_manifest.resolve())
        contract['replay_manifest_sha256']=hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        replay=json.loads(source_manifest.read_text())
        contract['reset_settle_steps']=replay['reset_settle_steps']
        contract['scene_sha256']=replay['source_sha256']
    (args.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    wall=time.monotonic(); losses=[]
    print(json.dumps(dict(parameters=sum(v.numel() for v in model.parameters()),
        **{k:contract[k] for k in ['H','K','A','windows','batch_size']})),flush=True)
    params,shadow=list(model.parameters()),list(ema.parameters())
    offsets=torch.arange(-1,15,device=args.device)
    anchors=torch.tensor(contract['window_anchors'],device=args.device)
    for step in range(start+1,args.steps+1):
        t=anchors[torch.randint(len(anchors),(args.batch_size,),device=args.device)]
        history=(t[:,None]+torch.tensor([-1,0],device=args.device)).clamp(0,39)
        future=t[:,None]+offsets
        batch={k:v[history] for k,v in obs.items()}
        if args.history_validity:
            batch['observation.history_valid']=(t[:,None]+torch.tensor([-1,0],device=args.device)>=0).float().unsqueeze(-1)
        batch['action']=actions[future.clamp(0,39)]
        batch['action_is_pad']=(future<0)|(future>=40)
        optimizer.zero_grad(set_to_none=True)
        loss,_=model(batch)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,10.)
        optimizer.step(); scheduler.step()
        decay=min(.9999,1-(1+max(0,step-1))**(-.75))
        with torch.no_grad():
            torch._foreach_lerp_(shadow,params,1-decay)
            for a,b in zip(ema.buffers(),model.buffers()): a.copy_(b)
        losses.append(float(loss.detach()))
        if step%args.log_every==0 or step==1:
            record=dict(step=step,loss=float(np.mean(losses)),grad_norm=float(grad),lr=scheduler.get_last_lr()[0],
                ema_decay=decay,elapsed_s=time.monotonic()-wall,steps_per_second=(step-start)/(time.monotonic()-wall))
            print(json.dumps(record),flush=True)
            with (args.output/'train_metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            losses=[]
        if step%args.save_every==0 or step==args.steps:
            checkpoint=args.output/'checkpoints'/f'{step:06d}'
            checkpoint.mkdir(parents=True,exist_ok=True)
            ema.save_pretrained(checkpoint/'ema')
            (checkpoint/'ema'/'normalization.json').write_text(json.dumps(norm,indent=2)+'\n')
            (checkpoint/'ema'/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
            torch.save(dict(step=step,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),normalization=norm,source_sha256=source_hash,
                modality=args.modality,steps=args.steps,window_sampling=args.window_sampling,
                history_validity=args.history_validity,prediction_type=args.prediction_type),
                checkpoint/'training_state.pt')
            (args.output/'latest_checkpoint.txt').write_text(str(checkpoint.resolve())+'\n')
            print(f'Saved EMA and resumable checkpoint at step {step}',flush=True)
    (args.output/'training_complete.json').write_text(json.dumps(dict(step=args.steps,elapsed_s=time.monotonic()-wall))+'\n')

if __name__=='__main__': main()
