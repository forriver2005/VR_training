"""Diagnostic only: target geometry never enters the policy or cloud sampler."""
import argparse,json
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,BOUNDED_TIMER,timer_sequence,fused_point_cloud


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);args=p.parse_args()
    plan=json.loads((args.collection/'experiment_plan.json').read_text());pairs=[]
    for background in range(4):
        for a in range(2):
            paths=[args.collection/f'background_{background}_a{a}_b{b}' for b in range(2)]
            if not all((p/'completed.json').is_file() for p in paths):continue
            zs=[np.load(p/'rollout.npz') for p in paths]
            rows=[]
            for t in [0,8,16,24,32]:
                hs=[max(0,t-1),t];future=np.arange(t,t+8).clip(0,39)
                counts=[];fcounts=[];fclouds=[]
                for z in zs:
                    # Ground truth is used ONLY for this visibility audit.
                    world=np.array(z['cube_position_world'][max(0,t-1),1]);base=z['base_to_world']
                    centre=np.linalg.inv(base)[:3,:3]@world+np.linalg.inv(base)[:3,3]
                    counts.append([int(np.sum(np.all(np.abs(z['point_cloud'][h]-centre)<.033,axis=1))) for h in hs])
                    fs=np.stack([fused_point_cloud(z['depth'][h],z['camera_intrinsics'],z['camera_to_world'],z['wrist_depth'][h],z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][h],base)[0] for h in hs])
                    fclouds.append(fs);fcounts.append([int(np.sum(np.all(np.abs(c-centre)<.033,axis=1))) for c in fs])
                row=dict(frame=t,history_frames=hs,wrist_target_counts=counts,fused_target_counts=fcounts,
                    state_max_abs_difference=float(np.max(np.abs(zs[0]['state'][hs]-zs[1]['state'][hs]))),
                    wrist_cloud_max_abs_difference_m=float(np.max(np.abs(zs[0]['point_cloud'][hs]-zs[1]['point_cloud'][hs]))),
                    fused_cloud_max_abs_difference_m=float(np.max(np.abs(fclouds[0]-fclouds[1]))),
                    wrist_depth_identical_fraction=float(np.mean(zs[0]['wrist_depth'][hs]==zs[1]['wrist_depth'][hs])),
                    executed_block_xyz_max_difference_m=float(np.linalg.norm(zs[0]['action'][future,:3]-zs[1]['action'][future,:3],axis=1).max()),
                    timer_history_equal=bool(np.array_equal(timer_sequence(zs[0]['action'],.25,BOUNDED_TIMER)[hs],timer_sequence(zs[1]['action'],.25,BOUNDED_TIMER)[hs])))
                row['identical_wrist_condition_different_actions']=bool(row['state_max_abs_difference']==0 and row['wrist_cloud_max_abs_difference_m']==0 and row['timer_history_equal'] and row['executed_block_xyz_max_difference_m']>.005)
                rows.append(row)
            equal_frames=[t for t in range(20) if np.array_equal(zs[0]['state'][t],zs[1]['state'][t]) and np.array_equal(zs[0]['point_cloud'][t],zs[1]['point_cloud'][t])]
            pairs.append(dict(background=background,a_variant=a,collection_success=[bool(z['success']) for z in zs],
                equal_state_cloud_frames_before_transport=equal_frames,observations=rows))
            for z in zs:z.close()
    report=dict(note='Diagnostic only; axis-aligned 33mm halfwidth is a near-target count, not semantic segmentation. Counts do not enter training.',
        pairs=pairs,exact_conflicts=sum(r['identical_wrist_condition_different_actions'] for p in pairs for r in p['observations']))
    (args.collection/'paired_observability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(pairs),exact_conflicts=report['exact_conflicts'])))

if __name__=='__main__':main()
