import argparse,json,shutil
from pathlib import Path
from delivery_policy_contract import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tau',type=float,default=.04);a=p.parse_args()
    if not 0<a.tau<=.1:raise ValueError('Invalid reference filter time constant')
    c=json.loads((a.checkpoint/'run_config.json').read_text())
    if c['control_dt_s']!=.25 or c['physics_steps_per_frame']!=30:raise ValueError('Requires validated4Hz/120Hz timing')
    if a.output.exists():raise ValueError('Output must be new')
    shutil.copytree(a.checkpoint,a.output)
    c['controller_filter']=dict(joint_tau_s=a.tau,gripper_tau_s=a.tau,physics_dt_s=1/120,
        code_sha256=sha256(Path(__file__).with_name('smooth_joint_targets.py')),
        semantics='same desired Cartesian targets and static drive gains; exponential reference convergence each physics tick; original reset unchanged')
    c['controller_filter_ablation']=dict(parent=str(a.checkpoint.resolve()),parent_contract_sha256=sha256(a.checkpoint/'run_config.json'),weights_retrained=False)
    (a.output/'run_config.json').write_text(json.dumps(c,indent=2)+'\n')
    for f in a.checkpoint.glob('*.safetensors'):
        if sha256(f)!=sha256(a.output/f.name):raise RuntimeError('Weights changed')

if __name__=='__main__':main()
