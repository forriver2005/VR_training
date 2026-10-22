"""Teacher-forced delivery evaluation with identical queue/timer preprocessing."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from delivery_policy_contract import load_entries,load_episode
from diffusion_overfit_common import Inference

def metrics(pred,target):
    pos=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)
    qa=pred[:,3:7].astype(float);qa/=np.linalg.norm(qa,axis=1,keepdims=True)
    qb=target[:,3:7].astype(float);qb/=np.linalg.norm(qb,axis=1,keepdims=True)
    angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=1)),0,1)))
    grip=np.abs(pred[:,7]-target[:,7])
    return dict(position_mean_m=float(pos.mean()),position_max_m=float(pos.max()),quaternion_mean_deg=float(angle.mean()),
        quaternion_max_deg=float(angle.max()),gripper_mean_m=float(grip.mean()),gripper_max_m=float(grip.max()),
        per_frame_position_m=pos.tolist(),per_frame_gripper_m=grip.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--list',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)));p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4)
    inference=Inference(args.checkpoint,args.device)
    args.output.mkdir(parents=True,exist_ok=True);rows=[];predictions=[]
    for e in load_entries(args.list):
        if e.get('scene_only_evaluation'):raise ValueError('Scene-only evaluation has no matching4Hz expert samples; evaluate in closed loop')
        ep=load_episode(e,inference.contract['timer'],inference.contract['pointcloud_contract']['source'])
        if inference.goal_memory:
            from goal_observation_registry import goal_estimator
            _,load_goal_features,_=goal_estimator(inference.contract['goal_memory']['recipe'])
            ep['goal_features']=load_goal_features(e)
        for seed in args.seeds:
            inference.reset(seed);pred=[];wall=time.monotonic()
            for t in range(40):
                a,_=inference.predict(ep['state'][t:t+1],point_cloud=ep['point_cloud'][t:t+1],elapsed_same_target_s=ep['elapsed_same_target_s'][t:t+1],
                    goal_features=ep['goal_features'][t:t+1] if inference.goal_memory else None)
                pred.append(a[0])
            pred=np.asarray(pred)
            row=dict(uid=e['uid'],seed=seed,**metrics(pred,ep['action']),inference_calls=inference.inference_calls,
                action_clipping_count=inference.clip_count,elapsed_s=time.monotonic()-wall)
            rows.append(row);predictions.append(pred)
            print(json.dumps({k:v for k,v in row.items() if not k.startswith('per_frame')}),flush=True)
    aggregate={k:float(np.mean([r[k] for r in rows])) for k in ['position_mean_m','quaternion_mean_deg','gripper_mean_m']}
    report=dict(checkpoint=str(args.checkpoint.resolve()),mode='teacher-forced observations and executed-expert-history timer; not closed loop',mean=aggregate,trials=rows)
    (args.output/'offline_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'offline_predictions.npz',predicted_action=np.asarray(predictions))

if __name__=='__main__':main()
