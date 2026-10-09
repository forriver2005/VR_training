#!/usr/bin/env python3
"""Teacher-forced observation reconstruction using the same A=8 online inference."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from diffusion_overfit_common import RAW, Inference

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,default=RAW)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.source) as data: ep={k:data[k] for k in data.files}
    inference=Inference(args.checkpoint,args.device)
    results=[]; predictions=[]; raw_predictions=[]
    for seed in args.seeds:
        inference.reset(seed); predicted=[]; raw=[]; wall=time.monotonic()
        for t in range(40):
            action,diag=inference.predict(ep['state'][t:t+1],
                [ep['camera_1_rgb'][t:t+1],ep['camera_2_rgb'][t:t+1]])
            predicted.append(action[0]); raw.append(diag['raw_action'][0])
        pred=np.asarray(predicted); target=ep['action']
        pos=np.linalg.norm(pred[:,:3]-target[:,:3],axis=-1)
        qa=pred[:,3:7]/np.maximum(np.linalg.norm(pred[:,3:7],axis=-1,keepdims=True),1e-8)
        qb=target[:,3:7]/np.maximum(np.linalg.norm(target[:,3:7],axis=-1,keepdims=True),1e-8)
        angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=-1)),0,1)))
        grip=np.abs(pred[:,7]-target[:,7])
        result=dict(seed=seed,position_mean_m=float(pos.mean()),position_max_m=float(pos.max()),
            position_rmse_m=float(np.sqrt(np.mean(pos**2))),quaternion_mean_deg=float(angle.mean()),
            quaternion_max_deg=float(angle.max()),gripper_mean_m=float(grip.mean()),gripper_max_m=float(grip.max()),
            inference_calls=inference.inference_calls,action_clipping_count=inference.clip_count,
            quaternion_fallback_count=inference.quaternion_fallbacks,elapsed_s=time.monotonic()-wall)
        results.append(result); predictions.append(pred); raw_predictions.append(raw)
        print(json.dumps(result),flush=True)
    report=dict(checkpoint=str(args.checkpoint.resolve()),source=str(args.source.resolve()),
        protocol='recorded observations before each action; H=2 K=16 A=8; EMA DDPM100; 5 replans over 40 steps',
        trials=results,mean={k:float(np.mean([v[k] for v in results])) for k in
            ['position_mean_m','quaternion_mean_deg','gripper_mean_m','position_max_m','quaternion_max_deg','gripper_max_m']})
    (args.output/'offline_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'offline_predictions.npz',seeds=args.seeds,predicted_action=predictions,
                        raw_action=raw_predictions,target_action=ep['action'])

if __name__=='__main__':main()
