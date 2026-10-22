"""Preserve the learned 'transport has started' event until an explicit reset."""
import argparse,json,shutil
from pathlib import Path
from delivery_policy_contract import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.checkpoint/'run_config.json').read_text())
    if not c.get('learned_goal_gate') or not c.get('goal_memory'):raise ValueError('Requires learned transport gate and measured goal')
    if a.output.exists():raise ValueError('Output must be new')
    shutil.copytree(a.checkpoint,a.output);c['monotonic_goal_gate']=True
    c['gate_memory_ablation']=dict(parent=str(a.checkpoint.resolve()),parent_contract_sha256=sha256(a.checkpoint/'run_config.json'),weights_retrained=False,
        rule='latch predicted gate>0.5 OR an actually returned valid-goal anchor weight>1/14; clear only on explicit episode reset',
        threshold_source='1/14 is half the smallest positive supervised transport anchor (1/7); no runtime expert phase or frame number',
        motivation='has-started is a historical event, not a classifier that may revert after gripper opening')
    (a.output/'run_config.json').write_text(json.dumps(c,indent=2)+'\n')
    for f in a.checkpoint.glob('*.safetensors'):
        if sha256(f)!=sha256(a.output/f.name):raise RuntimeError('Weights changed')

if __name__=='__main__':main()
