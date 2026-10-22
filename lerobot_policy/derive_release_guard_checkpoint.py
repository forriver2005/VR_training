"""Add a measured-pose execution guard; keep every learned weight unchanged."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,sha256
from release_guard import DEFAULT_RELEASE_GUARD

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.checkpoint/'run_config.json').read_text())
    if not all(c.get(k) for k in ['monotonic_goal_gate','goal_terminal_constraints','binary_gripper','fixed_down_quaternion']):raise ValueError('Requires monotonic stage memory and verified action/goal constraints')
    if c['control_dt_s']!=.25:raise ValueError('Guard timing requires the4Hz protocol')
    checked=[];height=.5*(c['cube_a_size_m']+c['cube_b_size_m'])
    for e in c['entries']:
        with np.load(ROOT/e['npz']) as z:act=z['action'];base=z['base_to_world']
        closed=np.flatnonzero(act[:,7]<.028)
        if not len(closed) or len(closed)!=closed[-1]-closed[0]+1 or closed[-1]>=37:raise ValueError('Expert gripper sequence is not one grasp followed by release')
        first=int(closed[-1]+1);diag=json.loads((ROOT/e['diagnostics']).read_text())
        goal=(np.linalg.inv(base)@np.r_[diag['initial_cube_states'][1]['position'],1])[:3];goal[2]+=height
        if not np.allclose(act[first:first+3,:3],goal,atol=.001,rtol=0):raise ValueError('Expert release is not a three-frame hold at the nominal stack height')
        checked.append(dict(uid=e['uid'],first_release_frame=first))
    if a.output.exists():raise ValueError('Output must be new')
    shutil.copytree(a.checkpoint,a.output)
    c['release_guard']=dict(DEFAULT_RELEASE_GUARD,recipe='settled_nominal_pose_before_learned_release_v1',code_sha256=sha256(ROOT/'lerobot_policy/release_guard.py'))
    c['release_guard_ablation']=dict(parent=str(a.checkpoint.resolve()),parent_contract_sha256=sha256(a.checkpoint/'run_config.json'),weights_retrained=False,
        inputs='measured robot pose, measured RGB-D goal, policy commands and their history; no object/contact ground truth',
        rule='defer a learned release request until two stable intervals at the nominal placement command; hold pose for opening plus two more intervals; never force release without a model request')
    (a.output/'run_config.json').write_text(json.dumps(c,indent=2)+'\n');(a.output/'release_action_contract_audit.json').write_text(json.dumps(checked,indent=2)+'\n')
    for f in a.checkpoint.glob('*.safetensors'):
        if sha256(f)!=sha256(a.output/f.name):raise RuntimeError('Weights changed')

if __name__=='__main__':main()
