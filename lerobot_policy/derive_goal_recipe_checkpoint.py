"""Versioned perception correction ablation, preserving neural weights."""
import argparse,json,shutil
from pathlib import Path
from delivery_policy_contract import sha256
from rgbd_goal_geometry_v2 import GOAL_RECIPE
from goal_observation_registry import goal_estimator

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    contract=json.loads((args.checkpoint/'run_config.json').read_text())
    if not contract.get('goal_memory'):raise ValueError('Parent requires measured goal conditioning')
    if args.output.exists():raise ValueError('Output must be new')
    shutil.copytree(args.checkpoint,args.output)
    previous=contract['goal_memory'];_,_,code=goal_estimator(GOAL_RECIPE)
    contract['goal_memory']=dict(recipe=GOAL_RECIPE,code_sha256=sha256(code),source='fixed RGB-D surface normals and depth-supported silhouettes; first valid measurement; known axis-aligned green6cm cube')
    contract['perception_ablation']=dict(parent=str(args.checkpoint.resolve()),parent_contract_sha256=sha256(args.checkpoint/'run_config.json'),previous_goal_memory=previous,weights_retrained=False,
        reason='occlusion/raster rows were mistaken for physical side faces; source indices and simulator object poses are not detector inputs')
    (args.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    for f in args.checkpoint.glob('*.safetensors'):
        if sha256(f)!=sha256(args.output/f.name):raise RuntimeError('Weights changed')
    shutil.copy2(code,args.output/'goal_estimator_source.py')

if __name__=='__main__':main()
