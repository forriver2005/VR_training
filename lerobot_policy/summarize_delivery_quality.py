"""Summarize completed overfit trials without dropping failures or relaxing gates."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report={'scope':'one training trajectory; no held-out generalization claim','experiments':[]}
    for closed_path in sorted(args.root.glob('closed_loop_*/summary.json')):
        tag=closed_path.parent.name.removeprefix('closed_loop_')
        offline_path=args.root/f'offline_{tag}/offline_metrics.json'
        if not offline_path.is_file():continue
        closed=json.loads(closed_path.read_text());offline=json.loads(offline_path.read_text())
        if closed['completed']!=closed['requested']:continue
        trials=closed['trials'];successful=[r for r in trials if r['success']]
        record=dict(tag=tag,checkpoint=closed['checkpoint'],completed=closed['completed'],successes=closed['successes'],
            success_rate=closed['success_rate'],runtime_errors=closed['runtime_errors'],
            offline_position_mean_mm=offline['mean']['position_mean_m']*1000,
            offline_position_worst_mm=max(r['position_max_m'] for r in offline['trials'])*1000,
            offline_quaternion_mean_deg=offline['mean']['quaternion_mean_deg'],
            offline_gripper_mean_mm=offline['mean']['gripper_mean_m']*1000,
            ik_failures=sum(r.get('ik_failures',0) for r in trials),
            action_clipping_count=sum(r.get('action_clipping_count',0) for r in trials),
            failure_categories=dict(Counter(f for r in trials for f in r.get('failure_reasons',[]))),
            xy_mean_mm=float(np.mean([r['final_xy_error_m'] for r in trials if 'final_xy_error_m' in r]))*1000,
            xy_max_mm=max((r.get('final_xy_error_m',0) for r in trials),default=0)*1000,
            success_example=successful[0]['result'] if successful else None,
            trial_results=trials)
        record['single_scene_gate_passed']=(record['completed']>=10 and record['success_rate']>=.9
            and not record['runtime_errors'] and not record['ik_failures'] and not record['action_clipping_count'])
        report['experiments'].append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in e.items() if k!='trial_results'} for e in report['experiments']],indent=2))

if __name__=='__main__':main()
