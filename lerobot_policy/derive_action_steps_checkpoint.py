"""Explicit inference-only action-block ablation; copied weights are unmodified."""
import argparse,json,shutil
import numpy as np
from pathlib import Path
from delivery_policy_contract import ROOT,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--action-steps',type=int,required=True)
    p.add_argument('--action-samples',type=int,default=1);p.add_argument('--binary-gripper',action='store_true');p.add_argument('--fixed-down-quaternion',action='store_true')
    p.add_argument('--anchor-snap-epsilon',type=float,default=0.)
    p.add_argument('--goal-terminal-constraints',action='store_true',help='after predicted arrival: goal XY and no TCP target below the stack contact height')
    args=p.parse_args()
    config=json.loads((args.checkpoint/'config.json').read_text());contract=json.loads((args.checkpoint/'run_config.json').read_text())
    if config['n_obs_steps']!=2 or config['horizon']!=16 or not 1<=args.action_steps<=15:raise ValueError('Unsupported action-block contract')
    if not 1<=args.action_samples<=16:raise ValueError('Invalid ensemble size')
    if not 0<=args.anchor_snap_epsilon<=.05 or (args.anchor_snap_epsilon and 'anchor' not in contract['position_action_mode']):raise ValueError('Invalid anchor endpoint projection')
    if args.goal_terminal_constraints and (not contract.get('goal_memory') or 'anchor' not in contract['position_action_mode']):raise ValueError('Terminal constraints require observed-goal anchor actions')
    terminal_audit=[]
    if args.binary_gripper or args.fixed_down_quaternion or args.goal_terminal_constraints:
        # These constraints must follow the frozen training action alphabet.
        # They are not a hand-written phase policy and never consult simulator truth.
        for entry in contract['entries']:
            with np.load(ROOT/entry['npz']) as z:a=z['action'];base=z['base_to_world']
            if args.binary_gripper and not np.all(np.minimum(np.abs(a[:,7]-.016),np.abs(a[:,7]-.04))<1e-6):raise ValueError('Training gripper actions are not binary')
            if args.fixed_down_quaternion and not np.allclose(a[:,3:7],[1,0,0,0],atol=1e-6,rtol=0):raise ValueError('Training orientation is not fixed down')
            if args.goal_terminal_constraints:
                # Training metadata is used only to verify the action contract.
                # The deployed projection obtains its goal from RGB-D, never this metadata.
                diag=json.loads((ROOT/entry['diagnostics']).read_text())
                b=(np.linalg.inv(base)@np.r_[diag['initial_cube_states'][1]['position'],1.])[:3]
                height=.5*(contract['cube_a_size_m']+contract['cube_b_size_m'])
                xy=float(np.linalg.norm(a[26:,:2]-b[:2],axis=1).max());z_margin=float((a[26:,2]-b[2]-height).min())
                if xy>.001 or z_margin<-.001:raise ValueError('Training terminal actions violate proposed goal constraints')
                terminal_audit.append(dict(uid=entry['uid'],max_xy_m=xy,min_height_margin_m=z_margin))
    if args.output.exists():raise ValueError('Output must be new')
    shutil.copytree(args.checkpoint,args.output)
    previous=config['n_action_steps'];config['n_action_steps']=args.action_steps
    contract['A']=args.action_steps
    contract['action_samples']=args.action_samples;contract['binary_gripper']=args.binary_gripper;contract['fixed_down_quaternion']=args.fixed_down_quaternion
    contract['anchor_snap_epsilon']=args.anchor_snap_epsilon
    contract['goal_terminal_constraints']=args.goal_terminal_constraints
    contract['inference_ablation']=dict(kind='action_steps',previous=previous,value=args.action_steps,
        parent=str(args.checkpoint.resolve()),parent_contract_sha256=sha256(args.checkpoint/'run_config.json'),weights_retrained=False)
    contract['inference_ablation'].update(action_samples=args.action_samples,binary_gripper=args.binary_gripper,fixed_down_quaternion=args.fixed_down_quaternion,
        anchor_snap_epsilon=args.anchor_snap_epsilon,
        goal_terminal_constraints=args.goal_terminal_constraints,
        ensemble='mean of normalized latent trajectories before geometric decoding',constraints_source='verified constant/binary action labels of every training episode')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');(args.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    weights=[p for p in args.checkpoint.iterdir() if p.suffix=='.safetensors']
    if not weights or not all(sha256(p)==sha256(args.output/p.name) for p in weights):raise RuntimeError('Weight copy verification failed')
    (args.output/'inference_ablation.json').write_text(json.dumps(contract['inference_ablation'],indent=2)+'\n')
    if terminal_audit:(args.output/'terminal_action_contract_audit.json').write_text(json.dumps(terminal_audit,indent=2)+'\n')

if __name__=='__main__':main()
