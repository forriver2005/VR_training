"""Small physically recollected layout pilot; never synthesize policy observations."""
from repro_paths import learning_python, isaac_python
import argparse,copy,hashlib,json,os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,NORMALIZATION,load_entries,sha256


def shifted_episode(entry,data,diag,spec):
    if spec['source_uid']!=entry['uid'] or spec['source_npz_sha256']!=entry['source_npz_sha256']:
        raise ValueError('Layout/source mismatch')
    a,b=np.asarray(spec['a_delta_base_m'],dtype=np.float64),np.asarray(spec['b_delta_base_m'],dtype=np.float64)
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():raise ValueError('Invalid deltas')
    if max(np.linalg.norm(a),np.linalg.norm(b))>.035 or a[2]!=0 or b[2]!=0:
        raise ValueError('Pilot permits only <=35 mm horizontal offsets')
    data=copy.deepcopy(data);diag=copy.deepcopy(diag)
    data['action']=data['action'].copy()
    # Keep the archived curved obstacle-avoidance path; blend its endpoint shift.
    shifts=np.repeat(a[None],40,axis=0)
    shifts[20:27]=a+(b-a)*np.linspace(1/7,1,7)[:,None]
    shifts[27:]=b
    data['action'][:,:3]+=shifts
    data['reset_target']=data['reset_target'].copy();data['reset_target'][:3]+=a
    for state,delta in zip(diag['initial_cube_states'],[a,b]):
        state['position']=(np.asarray(state['position'])+data['base_to_world'][:3,:3]@delta).tolist()
        state['linear_velocity']=[0.,0.,0.];state['angular_velocity']=[0.,0.,0.]
    if np.any(data['action']<np.asarray(NORMALIZATION['low'])-1e-6) or np.any(data['action']>np.asarray(NORMALIZATION['high'])+1e-6):
        raise ValueError('Shifted actions exceed frozen numerical bounds')
    return data,diag


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-gpus',nargs='+',default=['0','0']);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-prepared',action='store_true');args=p.parse_args()
    source=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021/pilot_scene_checks_4hz.jsonl'
    train=load_entries(ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021/train_4hz.jsonl')
    entries=load_entries(source)[:4];out=args.output.resolve()
    if not args.run_prepared:
        out.mkdir(parents=True,exist_ok=False);(out/'specs').mkdir();specs=[]
        for i,e in enumerate(entries):
            if e['uid'] not in {r['uid'] for r in train}:raise ValueError('Nontraining layout source')
            for a_id,dx in enumerate([0.,.012]):
                for b_id,dy in enumerate([-.025,.025]):
                    name=f'background_{i}_a{a_id}_b{b_id}'
                    identity=dict(source_scene=e['scene_id'],a_delta_base_m=[dx,0.,0.],b_delta_base_m=[0.,dy,0.])
                    spec=dict(identity,layout_id=name,scene_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16],
                        source_index=i,source_uid=e['uid'],source_npz_sha256=e['source_npz_sha256'],
                        action_rule='archive phases 0:20 + deltaA; 20:27 linear blend deltaA->deltaB; 27:40 + deltaB',
                        partition='train',policy_observation_rule='new synchronous Isaac Sim RGB-D, measured joints/TCP; depth-only cloud')
                    (out/'specs'/f'{name}.json').write_text(json.dumps(spec,indent=2)+'\n');specs.append(spec)
        (out/'experiment_plan.json').write_text(json.dumps(dict(sources=entries,specs=specs,
            gate='strict collection success AND independent expert IK replay; failed attempts retained',
            baseline='wrist absolute actions',training=dict(steps=40000,batch_size=16,seed=42,H=2,K=16,A=8),
            validation='unchanged validation_quick_4hz; select minimum mean position error; test84 unused',
            scope='local independent A/B perturbations of four training backgrounds; does not cover a new obstacle/background family'),indent=2)+'\n')
    else:specs=json.loads((out/'experiment_plan.json').read_text())['specs']
    if args.prepare_only:return
    def queue(gpu,jobs):
        rows=[]
        for spec in jobs:
            name=spec['layout_id'];target=out/name
            if target.exists():raise ValueError('Refuse to overwrite '+str(target))
            command=[isaac_python(),str(ROOT/'lerobot_policy/run_delivery_episode.py'),
                '--list',str(source),'--index',str(spec['source_index']),'--mode','expert_ik','--layout-spec',str(out/'specs'/f'{name}.json'),'--output',str(target)]
            with (out/f'{name}.log').open('w') as log:
                try:
                    proc=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1'),stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    if proc.returncode or not (target/'completed.json').is_file():raise RuntimeError('Incomplete collection')
                    result=json.loads((target/'result.json').read_text());row=dict(layout_id=name,success=result['success'],failure_reasons=result['failure_reasons'],final_xy_error_m=result['final_xy_error_m'])
                except Exception as exc:row=dict(layout_id=name,success=False,error=repr(exc))
            if row['success']:
                e=entries[spec['source_index']]
                row['entry']=dict(e,uid=f'layout_pilot/{name}',group='layout_pilot',episode_id=name,scene_id=spec['scene_id'],
                    npz=str((target/'rollout.npz').relative_to(ROOT)),source_npz_sha256=sha256(target/'rollout.npz'),
                    diagnostics=str((target/'result.json').relative_to(ROOT)),scene_metadata=str((target/'scene.json').relative_to(ROOT)),
                    replay_report=None,parent_uid=e['uid'],parent_scene_id=e['scene_id'],generated_layout=True)
            (out/f'{name}.status.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(json.dumps(row),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=len(args.sim_gpus)) as pool:
        futures=[pool.submit(queue,gpu,specs[i::len(args.sim_gpus)]) for i,gpu in enumerate(args.sim_gpus)]
        rows=[row for f in futures for row in f.result()]
    rows.sort(key=lambda r:r['layout_id'])
    (out/'collection_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'collected_candidates.jsonl').write_text(''.join(json.dumps(r['entry'])+'\n' for r in rows if r['success']))
    print('COLLECTION',sum(r['success'] for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
