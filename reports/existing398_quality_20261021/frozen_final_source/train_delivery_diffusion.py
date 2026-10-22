"""Train PointNet diffusion from audited 4 Hz JSONL lists, including hold history."""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from diffusion_overfit_common import make_config, vector_transform
from pointcloud_diffusion import PointCloudPolicy
from delivery_policy_contract import ROOT, CROP, NORMALIZATION, TIMER, BOUNDED_TIMER, POINT_SOURCES, load_entries, load_episode, sha256
from relative_action import RELATIVE_NORMALIZATION,relative_targets

def make_batch(obs, actions, episode, frame):
    history=frame[:,None]+torch.tensor([-1,0],device=frame.device)
    future=frame[:,None]+torch.arange(-1,15,device=frame.device)
    batch={k:v[episode[:,None],history.clamp(0,39)] for k,v in obs.items()}
    batch['observation.history_valid']=(history>=0).float().unsqueeze(-1)
    batch['action']=actions[episode[:,None],future.clamp(0,39)]
    batch['action_is_pad']=(future<0)|(future>=40)
    return batch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--list',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prediction-type',choices=['epsilon','sample'],default='epsilon')
    p.add_argument('--timer-mode',choices=['anchor','bounded'],default='anchor')
    p.add_argument('--pointcloud-source',choices=list(POINT_SOURCES),default='wrist_depth')
    p.add_argument('--relative-actions',action='store_true',help='predict XYZ offset from the planning TCP; execute absolute world targets')
    p.add_argument('--observation-frame',choices=['base','current_tcp','initial_tcp'],default='base')
    p.add_argument('--encoder',choices=['pointnet','spatial_tokens','proprioceptive'],default='pointnet')
    p.add_argument('--goal-memory',action='store_true',help='initial green target measured from fixed RGB-D; no simulator truth')
    p.add_argument('--goal-recipe',choices=['v1','v2'],default='v1')
    p.add_argument('--goal-anchor-actions',action='store_true',help='learn transport anchor weights and XYZ residual; ninth training-only output channel')
    p.add_argument('--goal-xy-anchor-actions',action='store_true')
    p.add_argument('--episode-goal-actions',action='store_true',help='fixed measured initial TCP and goal anchors; avoids replanning-reference drift')
    p.add_argument('--episode-goal-3d-actions',action='store_true')
    p.add_argument('--goal-state-switch',action='store_true')
    p.add_argument('--learned-goal-gate',action='store_true',help='robot-history phase classifier gates goal conditioning; no runtime expert stage')
    p.add_argument('--position-loss-weight',type=float,default=1.)
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    anchored=args.goal_anchor_actions or args.goal_xy_anchor_actions or args.episode_goal_actions or args.episode_goal_3d_actions
    if not math.isfinite(args.position_loss_weight) or args.position_loss_weight<=0:raise ValueError('Invalid loss weight')
    if sum([args.relative_actions,args.goal_anchor_actions,args.goal_xy_anchor_actions,args.episode_goal_actions,args.episode_goal_3d_actions])>1:raise ValueError('Choose one action representation')
    if (args.episode_goal_actions or args.episode_goal_3d_actions) and args.observation_frame!='initial_tcp':raise ValueError('Episode anchors require measured initial TCP memory')
    if args.goal_state_switch and not args.learned_goal_gate:raise ValueError('Goal state switch requires learned gate')
    if args.observation_frame!='base' and not (args.relative_actions or anchored):
        raise ValueError('TCP-centered observations require relative XYZ action targets')
    if anchored and not args.goal_memory:
        raise ValueError('Goal anchor requires goal memory and replaces relative-actions')
    if args.learned_goal_gate and (args.observation_frame!='initial_tcp' or not args.goal_memory):raise ValueError('Learned gate requires initial TCP memory and goal memory')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    entries=load_entries(args.list)
    if any(e.get('evaluation_only') or e.get('scene_only_evaluation') for e in entries):raise ValueError('Evaluation-only scene manifests cannot be used for training')
    timer_config=TIMER if args.timer_mode=='anchor' else BOUNDED_TIMER
    output_normalization=RELATIVE_NORMALIZATION if args.relative_actions else NORMALIZATION
    if anchored:
        if args.episode_goal_3d_actions:
            from episode_goal_3d import EPISODE_3D_NORMALIZATION as output_normalization,targets as goal_anchor_targets
        elif args.episode_goal_actions:
            from episode_goal_anchor import EPISODE_NORMALIZATION as output_normalization,targets as goal_anchor_targets
        elif args.goal_xy_anchor_actions:
            from goal_xy_anchor_action import NORMALIZATION_XY as output_normalization,targets as goal_anchor_targets
        else:
            from goal_anchor_action import GOAL_ANCHOR_NORMALIZATION as output_normalization,goal_anchor_targets
        expected=dict(approach=6,pregrasp_hold=1,descend=4,close=3,grip_hold=1,lift=5,transport=7,place=4,place_hold=2,release=3,retreat=4)
        for package in {e['package_root'] for e in entries}:
            if json.loads((ROOT/package/'meta/collection_config.json').read_text())['phases']!=expected:raise ValueError('Goal anchor phase labels require the frozen 40-frame expert schedule')
    from delivery_policy_contract import normalize_points
    episodes=[load_episode(e,timer_config,args.pointcloud_source) for e in entries]
    obs={'observation.state':vector_transform(torch.tensor(np.stack([e['state'] for e in episodes]),device=args.device),NORMALIZATION),
         'observation.point_cloud':torch.tensor(normalize_points(np.stack([e['point_cloud'] for e in episodes])),device=args.device),
         'observation.elapsed_same_target_s':torch.tensor(np.stack([e['elapsed_same_target_s'] for e in episodes])[:,:,None]/TIMER['normalization_seconds'],device=args.device).clamp(0,1)}
    actions=vector_transform(torch.tensor(np.stack([e['action'] for e in episodes]),device=args.device),NORMALIZATION)
    if args.observation_frame=='initial_tcp':obs['observation.reset_state']=obs['observation.state'][:,:1].expand(-1,40,-1)
    if args.goal_memory:
        if args.goal_recipe=='v2':from rgbd_goal_geometry_v2 import load_goal_features,GOAL_RECIPE
        else:from rgbd_object_features import load_goal_features,GOAL_RECIPE
        obs['observation.goal_features']=torch.tensor(np.stack([load_goal_features(e) for e in entries]),device=args.device)
    cfg=make_config(args.device)
    if anchored:cfg.output_features['action']=PolicyFeature(FeatureType.ACTION,(9,))
    cfg.input_features={'observation.state':cfg.input_features['observation.state'],
                        'observation.history_valid':PolicyFeature(FeatureType.STATE,(1,)),
                        'observation.elapsed_same_target_s':PolicyFeature(FeatureType.STATE,(1,))}
    if args.goal_memory:cfg.input_features['observation.goal_features']=PolicyFeature(FeatureType.STATE,(8,))
    if args.observation_frame=='initial_tcp':cfg.input_features['observation.reset_state']=PolicyFeature(FeatureType.STATE,(8,))
    cfg.prediction_type=args.prediction_type
    model=PointCloudPolicy(cfg,encoder_type=args.encoder,observation_frame=args.observation_frame,learned_goal_gate=args.learned_goal_gate,position_loss_weight=args.position_loss_weight,goal_state_switch=args.goal_state_switch).to(args.device).train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    def lr_factor(step):
        if step<500:return (step+1)/500
        return .5*(1+math.cos(math.pi*min(1.,(step-500)/max(1,args.steps-500))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    contract=dict(protocol_version='delivery_4hz_v1',modality='pointcloud',H=2,K=16,A=8,
        ddpm_steps=100,prediction_type=args.prediction_type,history_validity=True,timer=timer_config,
        pointcloud_contract=dict(source=args.pointcloud_source,crop=CROP.tolist(),pixel_center_offset=.5,
            fps=POINT_SOURCES[args.pointcloud_source],view_quotas=[256,256,512] if args.pointcloud_source=='fused_depth' else [1024]),
        normalization=output_normalization,control_dt_s=.25,physics_steps_per_frame=30,reset_settle_steps=240,
        cube_a_size_m=.04,cube_b_size_m=.06,success_xy_m=.005,success_z_delta_m=.05,
        entries=entries,list_sha256=sha256(args.list),steps=args.steps,seed=args.seed,batch_size=args.batch_size,
        optimizer='AdamW lr=1e-4 betas=(.95,.999) wd=1e-6',ema_power=.75,
        history_offsets=[-1,0],action_offsets=list(range(-1,15)),boundary_padding='repeat',mask_padding_loss=False,
        network='PointNet 3-64-128-256 ->64; state 8-64-64; history + timer; UNet [64,128,256]',
        augmentation=False,training_scene_ids=sorted({e['scene_id'] for e in entries}),
        preprocessing_sha256=sha256(Path(__file__).with_name('delivery_policy_contract.py')))
    contract['derived_point_clouds']=[dict(path=e['derived_point_cloud_path'],sha256=sha256(e['derived_point_cloud_path'])) for e in episodes]
    contract['position_action_mode']='goal_anchor_residual' if args.goal_anchor_actions else ('relative_to_current_tcp' if args.relative_actions else 'absolute_base')
    if args.goal_xy_anchor_actions:contract['position_action_mode']='goal_xy_anchor_residual'
    if args.episode_goal_3d_actions:
        contract['position_action_mode']='episode_goal_3d_anchor'
        contract['goal_anchor_actions']=dict(code_sha256=sha256(Path(__file__).with_name('episode_goal_3d.py')),
            training_only_anchor_labels='alpha=clip((expert_frame-19)/7,0,1); not an inference input',
            decoder='initial measured TCP + [alpha,alpha,clip((7alpha-4)/3)]*(observed goal-initial TCP)+residual')
    if args.episode_goal_actions:
        contract['position_action_mode']='episode_goal_xy_anchor'
        contract['goal_anchor_actions']=dict(code_sha256=sha256(Path(__file__).with_name('episode_goal_anchor.py')),
            training_only_anchor_labels='alpha=clip((expert_frame-19)/7,0,1); not an inference input',
            decoder='XY initial measured TCP + predicted alpha*(observed goal-initial TCP)+residual; Z initial measured TCP+residual')
    if args.goal_xy_anchor_actions:
        contract['goal_anchor_actions']=dict(code_sha256=sha256(Path(__file__).with_name('goal_xy_anchor_action.py')),
            training_only_anchor_labels='alpha=clip((expert_frame-19)/7,0,1); not an inference input',
            decoder='XY TCP + learned_alpha*(measured_goal-TCP) + residual; Z TCP+residual')
    if args.goal_anchor_actions:
        contract['goal_anchor_actions']=dict(code_sha256=sha256(Path(__file__).with_name('goal_anchor_action.py')),
            training_only_anchor_labels='alpha=clip((expert_frame-19)/7,0,1); not an inference input',
            decoder='TCP + learned_alpha*(measured_goal-TCP) + learned_residual; output orientation/grip absolute')
    contract['encoder_type']=args.encoder
    contract['observation_frame']=args.observation_frame
    contract['learned_goal_gate']=args.learned_goal_gate
    contract['goal_state_switch']=args.goal_state_switch
    contract['position_loss_weight']=args.position_loss_weight
    if args.learned_goal_gate:contract['goal_gate_supervision']=dict(label='expert_frame>=20, training loss only',weight=.05,gate='predicted from reset-relative robot history, gripper and executed-command timer; hard straight-through mask')
    if args.goal_memory:
        from goal_observation_registry import goal_estimator
        _,_,goal_code=goal_estimator(GOAL_RECIPE)
        contract['goal_memory']=dict(recipe=GOAL_RECIPE,source='initial fixed-camera RGB-D; green channel geometry; known 6cm target size',
            code_sha256=sha256(goal_code))
    contract['network']=('Spatial local neighborhoods128x16 + Transformer128x2 ->64' if args.encoder=='spatial_tokens' else 'PointNet 3-64-128-256 ->64')+'; state 8-64-64; history + timer; UNet [64,128,256]'
    if args.encoder=='proprioceptive':contract['network']='No point-cloud encoder; measured-goal/state MLPs; history + timer; Diffusion UNet [64,128,256]'
    if args.goal_memory:contract['network']+='; measured RGB-D goal MLP 8-64-64'
    contract['model_code_sha256']={name:sha256(Path(__file__).with_name(name)) for name in ['pointcloud_diffusion.py','spatial_pointcloud.py']}
    contract['observation_normalization']=NORMALIZATION
    contract['action_limits']=NORMALIZATION
    if args.relative_actions:contract['relative_action_code_sha256']=sha256(Path(__file__).with_name('relative_action.py'))
    start=0
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['contract']!=contract:raise ValueError('Resume contract mismatch')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device!='cpu':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);start=saved['step']
    args.output.mkdir(parents=True,exist_ok=args.resume is not None)
    (args.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    parameters=list(model.parameters());shadow=list(ema.parameters());losses=[];wall=time.monotonic()
    print(json.dumps(dict(episodes=len(entries),windows=len(entries)*40,parameters=sum(p.numel() for p in parameters))),flush=True)
    for step in range(start+1,args.steps+1):
        ep=torch.randint(len(entries),(args.batch_size,),device=args.device)
        t=torch.randint(40,(args.batch_size,),device=args.device)
        batch=make_batch(obs,actions,ep,t)
        if anchored:batch=goal_anchor_targets(batch,t)
        elif args.relative_actions:batch=relative_targets(batch)
        if args.learned_goal_gate:batch['transport_started_target']=(t>=20).float()
        opt.zero_grad(set_to_none=True);loss,details=model(batch)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,10.)
        opt.step();scheduler.step()
        decay=min(.9999,1-(1+max(0,step-1))**(-.75))
        with torch.no_grad():
            torch._foreach_lerp_(shadow,parameters,1-decay)
            for x,y in zip(ema.buffers(),model.buffers()):x.copy_(y)
        losses.append(float(loss.detach()))
        if step==1 or step%100==0:
            row=dict(step=step,loss=float(np.mean(losses)),grad_norm=float(grad),lr=scheduler.get_last_lr()[0],elapsed_s=time.monotonic()-wall)
            if details:row.update({k:float(v) for k,v in details.items()})
            with (args.output/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);losses=[]
        if step%args.save_every==0 or step==args.steps:
            target=args.output/'checkpoints'/f'{step:06d}';target.mkdir(parents=True,exist_ok=True)
            ema.save_pretrained(target/'ema')
            for name,value in [('run_config',contract),('normalization',output_normalization)]:
                (target/'ema'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            torch.save(dict(step=step,contract=contract,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device!='cpu' else [],
                python_rng=random.getstate(),numpy_rng=np.random.get_state()),target/'training_state.pt')
            (args.output/'latest_checkpoint.txt').write_text(str((target/'ema').resolve())+'\n')
            print('CHECKPOINT',target,flush=True)
    (args.output/'training_complete.json').write_text(json.dumps(dict(step=args.steps,elapsed_s=time.monotonic()-wall))+'\n')

if __name__=='__main__':main()
