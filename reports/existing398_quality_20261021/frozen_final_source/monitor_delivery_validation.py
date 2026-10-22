"""Evaluate fixed held-out trajectories at saved checkpoints, then freeze a choice."""
import argparse,json,os,subprocess,time
from pathlib import Path
from delivery_policy_contract import ROOT,load_entries

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--list',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',nargs='+',type=int,default=[10000,20000,30000,40000]);p.add_argument('--gpu',default='2')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    validation=load_entries(args.list);rows=[]
    for step in args.steps:
        checkpoint=args.training/'checkpoints'/f'{step:06d}'/'ema';deadline=time.monotonic()+3600
        while not (checkpoint/'run_config.json').is_file():
            if time.monotonic()>deadline:raise TimeoutError(f'Checkpoint not produced: {checkpoint}')
            time.sleep(3)
        contract=json.loads((checkpoint/'run_config.json').read_text())
        if set(contract['training_scene_ids']) & {e['scene_id'] for e in validation}:raise ValueError('Validation scene leakage')
        target=args.output/f'step_{step:06d}'
        print('VALIDATE',step,flush=True)
        with (args.output/f'step_{step:06d}.log').open('w') as log:
            subprocess.run(['/home/houshengyuan/anaconda3/envs/lerobot/bin/python',str(ROOT/'lerobot_policy/evaluate_delivery_diffusion.py'),
                '--checkpoint',str(checkpoint.resolve()),'--list',str(args.list.resolve()),'--output',str(target.resolve()),'--seeds','0'],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((target/'offline_metrics.json').read_text())
        row=dict(step=step,checkpoint=str(checkpoint.resolve()),**metrics['mean'],episodes=len(metrics['trials']))
        rows.append(row);print(json.dumps(row),flush=True)
        report=dict(selection_rule='minimum validation25 mean position error; fixed seed0',trials=rows,
            best=min(rows,key=lambda r:r['position_mean_m']),complete=len(rows)==len(args.steps))
        (args.output/'validation_curve.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'selected_checkpoint.txt').write_text(report['best']['checkpoint']+'\n')

if __name__=='__main__':main()
