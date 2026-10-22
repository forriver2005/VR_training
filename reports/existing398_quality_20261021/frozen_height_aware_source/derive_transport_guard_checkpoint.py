import argparse,json,shutil
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,sha256
from transport_clearance_guard import DEFAULT_TRANSPORT_GUARD

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.checkpoint/'run_config.json').read_text())
    if not c.get('monotonic_goal_gate') or not c.get('release_guard') or not c.get('goal_terminal_constraints'):raise ValueError('Requires the validated goal/phase/release contract')
    checked=[]
    for e in c['entries']:
        with np.load(ROOT/e['npz']) as z:act=z['action'];base=z['base_to_world']
        d=json.loads((ROOT/e['diagnostics']).read_text());goal=(np.linalg.inv(base)@np.r_[d['initial_cube_states'][1]['position'],1])[:3]
        offset=float(act[26,2]-goal[2])
        if abs(offset-DEFAULT_TRANSPORT_GUARD['hover_offset_m'])>.001:raise ValueError('Expert approach height differs from guard contract')
        checked.append(dict(uid=e['uid'],approach_height_above_goal_m=offset))
    if a.output.exists():raise ValueError('Output must be new')
    shutil.copytree(a.checkpoint,a.output)
    c['transport_clearance_guard']=dict(DEFAULT_TRANSPORT_GUARD,recipe='observed_goal_height_before_learned_horizontal_transport_v1',code_sha256=sha256(ROOT/'lerobot_policy/transport_clearance_guard.py'))
    c['transport_guard_ablation']=dict(parent=str(a.checkpoint.resolve()),parent_contract_sha256=sha256(a.checkpoint/'run_config.json'),weights_retrained=False,
        inputs='actual robot state, RGB-D target and policy transport/release events only; no obstacle/cube ground-truth transforms',
        purpose='preserve the learned horizontal route while preventing transport below the destination approach height')
    (a.output/'run_config.json').write_text(json.dumps(c,indent=2)+'\n');(a.output/'transport_height_contract_audit.json').write_text(json.dumps(checked,indent=2)+'\n')
    for f in a.checkpoint.glob('*.safetensors'):
        if sha256(f)!=sha256(a.output/f.name):raise RuntimeError('Weights changed')

if __name__=='__main__':main()
