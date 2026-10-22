"""Report completed fixed-protocol results; do not double count reused trials."""
import json
from pathlib import Path
from delivery_policy_contract import ROOT,sha256
OUT=ROOT/'lerobot_policy/outputs/existing398_20261021'
REPORT=ROOT/'reports/existing398_quality_20261021'
MODELS=['wrist_absolute','fused_tcp_pointnet','fused_tcp_spatial','goal_wrist_tcp','goal_fused_tcp','goal_anchor_proprio',
    'episode_gate_proprio','episode_gate_wrist','canonical_gate_proprio','canonical_gate_wrist',
    'episode_gate_wrist_goal_v2','canonical_gate_proprio_goal_v2']
rows=[]
for name in MODELS:
    p=OUT/(name+'_closed')/'summary.json'
    if not p.exists():continue
    s=json.loads(p.read_text());trials=s['trials']
    if len(trials)!=30 or len({(r['index'],r['seed']) for r in trials})!=30:raise ValueError('Unexpected final trial coverage '+name)
    cp=Path(s['checkpoint']);contract=json.loads((cp/'run_config.json').read_text())
    seen=[r for r in trials if r['index']<4];val=[r for r in trials if r['index']==4]
    result=dict(name=name,checkpoint=str(cp),checkpoint_contract_sha256=sha256(cp/'run_config.json'),
        encoder=contract.get('encoder_type','pointnet'),observation_frame=contract.get('observation_frame','base'),
        action_mode=contract.get('position_action_mode'),goal_recipe=contract.get('goal_memory',{}).get('recipe'),A=contract['A'],
        train_successes=sum(r['success'] for r in seen),train_count=len(seen),validation_successes=sum(r['success'] for r in val),validation_count=len(val),
        runtime_errors=sum('error' in r for r in trials),ik_failures=sum(r.get('ik_failures',0) for r in trials),action_clipping_count=sum(r.get('action_clipping_count',0) for r in trials),
        validation_max_xy_mm=max(r.get('final_xy_error_m',0) for r in val)*1000,
        seen_by_layout={str(i):sum(r['success'] for r in seen if r['index']==i) for i in range(4)},summary=str(p.relative_to(ROOT)))
    rows.append(result)
report=dict(training_episodes=398,training_layouts=4,validation_layouts=1,test_used=False,
    limits=dict(control_dt_s=.25,actions=40,xy_m=.005,height_difference_m=.05,height_error_m=.004,drift_m=.001,speed_m_s=.05),
    note='Validation was used for selection; this is not an independent test estimate. RGB-D goal geometry uses known green6cm axis-aligned cube priors; no simulator object pose is fed to the policy.',
    models=rows)
REPORT.mkdir(exist_ok=True,parents=True);(REPORT/'quality_summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| 方案 | 训练布局闭环 | 未见验证布局闭环 | A | 运行错误 |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['name']} | {r['train_successes']}/{r['train_count']} | {r['validation_successes']}/{r['validation_count']} | {r['A']} | {r['runtime_errors']} |")
(REPORT/'results_table.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
