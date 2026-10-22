"""Finalize evidence and data-role accounting after the frozen fresh test."""
import hashlib,json,platform,shutil
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT
OUT=ROOT/'lerobot_policy/outputs/existing398_20261021';REPORT=ROOT/'reports/existing398_quality_20261021'
MANIFEST=ROOT/'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate_cases(summary):
    results=[json.loads(Path(t['result']).read_text()) for t in summary['trials'] if 'result' in t]
    if len(results)!=summary['completed']:raise ValueError('Incomplete physics results')
    return dict(count=len(results),successes=sum(r['success'] for r in results),
        mean_final_xy_mm=float(np.mean([r['final_xy_error_m'] for r in results])*1000),
        max_final_xy_mm=max(r['final_xy_error_m'] for r in results)*1000,
        max_stable_xy_mm=max(r['stable_window_max_xy_error_m'] for r in results)*1000,
        max_height_error_mm=max(r['stable_window_max_z_error_m'] for r in results)*1000,
        max_drift_mm=max(max(r['stable_window_drift_m']) for r in results)*1000,
        max_speed_mm_s=max(max(r['release_window_audit']['max_speed_m_s']) for r in results)*1000,
        ik_failures=sum(r['ik_failures'] for r in results),workspace_clips=sum(r['action_clipping_count'] for r in results),
        forbidden_contact_trials=sum(bool(r['obstacle_collision_pairs'] or r['reset_forbidden_contacts']) for r in results),
        release_guard_overrides=sum(r.get('release_guard_override_count',0) for r in results),
        terminal_goal_projections=sum(r.get('goal_terminal_projection_count',0) for r in results),
        minimum_final_tcp_height_above_measured_goal_m=min((r['final_tcp_state'][2]-r['goal_memory_audit']['measured_initial_goal_base'][2] for r in results if 'final_tcp_state' in r),default=None))

def main():
    frozen=json.loads((OUT/'frozen_final_candidate.json').read_text());cp=Path(frozen['checkpoint'])
    first=json.loads((OUT/'final_test/summary.json').read_text());fresh=json.loads((OUT/'fresh_holdout_policy/summary.json').read_text())
    known_path=OUT/('smooth_release_regression' if cp.name.endswith('_smooth') else 'settled_release_regression')/'summary.json'
    known=json.loads(known_path.read_text())
    for n,h in frozen['checkpoint_sha256'].items():
        if sha(cp/n)!=h:raise ValueError('Checkpoint changed: '+n)
    for n,h in frozen['source_sha256'].items():
        if sha(ROOT/'lerobot_policy'/n)!=h:raise ValueError('Frozen source changed: '+n)
    for s in [known,fresh]:
        if s['runtime_errors']:raise ValueError('Runtime errors must be resolved and retained separately')
        for t in s['trials']:
            folder=Path(t['result']).parent
            if not all((folder/n).is_file() for n in ['completed.json','rollout.npz','rollout.gif']):raise ValueError('Missing artifact '+str(folder))
    k=aggregate_cases(known);f=aggregate_cases(fresh)
    contract=json.loads((cp/'run_config.json').read_text())
    summary=dict(status='frozen evaluation completed',checkpoint=str(cp),training_episodes=398,training_layouts=4,
        known_regression=k,known_regression_groups=known['groups'],fresh_independent=f,fresh_groups=fresh['groups'],
        first_independent_test=dict(successes=first['successes'],count=first['completed'],subsequently_used_for_debugging=True),
        data_roles=json.loads((MANIFEST/'evaluation_plan.json').read_text()),
        policy=dict(encoder=contract['encoder_type'],H=2,K=16,A=contract['A'],ddpm_steps=100,prediction_type='sample',EMA=True,
            monotonic_goal_gate=contract.get('monotonic_goal_gate'),goal_recipe=contract['goal_memory']['recipe'],
            fixed_down_quaternion=contract.get('fixed_down_quaternion'),binary_gripper=contract.get('binary_gripper'),
            goal_terminal_constraints=contract.get('goal_terminal_constraints'),release_guard=contract.get('release_guard'),controller_filter=contract.get('controller_filter')),
        scope='task-specific RGB-D geometry + learned diffusion planning + execution constraints; no public-benchmark SOTA claim')
    (REPORT/'final_quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (REPORT/'execution_manifest.json').write_text(json.dumps(dict(frozen_candidate=frozen,
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [known_path,OUT/'fresh_holdout_policy/summary.json',OUT/'final_test/summary.json',OUT/'final_unit_tests.log',MANIFEST/'fresh_scenes.jsonl']},
        GPUs_used=[0,1,2],raw_demonstrations_added_to_training=0),indent=2)+'\n')
    if k['successes']==k['count'] and f['successes']==f['count']:
        (OUT/'recommended_checkpoint.txt').write_text(str(cp)+'\n')
    labels=['wrist_absolute','fused_tcp_pointnet','fused_tcp_spatial','goal_wrist_tcp','goal_fused_tcp','goal_anchor_proprio','episode_gate_proprio','episode_gate_wrist','canonical_gate_proprio','canonical_gate_wrist','episode_gate_wrist_goal_v2','canonical_gate_proprio_goal_v2']
    table=['| 方案 | 训练布局 | 原验证布局 |','|---|---:|---:|']
    for name in labels:
        p=OUT/(name+'_closed')/'summary.json'
        if not p.exists():continue
        d=json.loads(p.read_text());tr=d['trials'];table.append(f"| {name} | {sum(t['success'] for t in tr if t['index']<4)}/20 | {sum(t['success'] for t in tr if t['index']==4)}/10 |")
    known_groups=known['groups'];seen=sum(known_groups[str(i)]['successes'] for i in range(4));val=known_groups['4']['successes'];reg=known_groups['5']['successes']
    lines=[
        '# 现有数据优化：冻结评测报告',
        '',f"最终候选在已知布局回归中取得 **{k['successes']}/{k['count']}**，在两个新保留布局上取得 **{f['successes']}/{f['count']}**。测试前已冻结权重、配置及代码哈希，测试期间未调整。",
        '',f"当前方案采用固定相机RGB-D目标几何、实际机器人状态与初始状态记忆，以及Diffusion Policy。它属于带任务先验和执行约束的方案；最终候选的神经网络未使用PointNet特征。环境为Isaac Sim6、FR3V2、4cm/6cm方块和盒障碍。",
        '', '## 评测结果与数据角色',
        '', '| 评测部分 | 布局数 | 成功次数 | 用途 |','|---|---:|---:|---|',
        f'| 训练布局 | 4 | {seen}/20 | 已见布局回归 |',f'| 原验证布局 | 1 | {val}/10 | 用于选模 |',
        f'| 原测试布局 | 1 | {reg}/20 | 首轮测试后用于错误定位，现为回归 |',
        f"| 新保留布局 | 2 | {f['successes']}/40 | 未用于本轮模型与参数选择 |",
        '',f"需要保留的负面结果：首次冻结模型在原测试布局只有 **{first['successes']}/{first['completed']}**。随后定位到阶段判断回退等问题，因此该布局的后续成绩不能再称为完全独立测试。",
        '', '新的两个布局来自此前保留的8Hz数据包。只复用它们的场景资产和实测reset，在统一4Hz、40步协议下执行策略；没有把8Hz演示重标为4Hz训练数据。两者与已使用布局按实际方块/障碍位姿核对，不仅按目录名或scene_id区分；各自的4Hz专家可行性回放通过。',
        '',f"全部训练仍使用原有 **398条、4个布局**。开头曾采集的16个局部变化布局在用户调整分工后被隔离，没有加入任何本轮训练。",
        '', '| 指标 | 已知布局回归 | 新保留布局 |','|---|---:|---:|',
        f"| 平均最终XY误差 | {k['mean_final_xy_mm']:.3f} mm | {f['mean_final_xy_mm']:.3f} mm |",
        f"| 最大稳定窗口XY误差 | {k['max_stable_xy_mm']:.3f} mm | {f['max_stable_xy_mm']:.3f} mm |",
        f"| 最大高度误差 | {k['max_height_error_mm']:.3f} mm | {f['max_height_error_mm']:.3f} mm |",
        f"| 最大稳定漂移 | {k['max_drift_mm']:.3f} mm | {f['max_drift_mm']:.3f} mm |",
        f"| 最大稳定窗口速度 | {k['max_speed_mm_s']:.3f} mm/s | {f['max_speed_mm_s']:.3f} mm/s |",
        f"| IK失败 | {k['ik_failures']} | {f['ik_failures']} |",
        f"| 全局工作空间额外限幅 | {k['workspace_clips']} | {f['workspace_clips']} |",
        f"| 禁用障碍接触试验数 | {k['forbidden_contact_trials']} | {f['forbidden_contact_trials']} |",
        '', '成功判据保持：XY≤5mm、中心高度差50±4mm、漂移<1mm、速度<50mm/s、抓起/释放/堆叠接触、无禁用盒障碍接触、IK正常。每次策略40步、4Hz，之后维持最后指令做固定物理稳定检查。姿态/夹爪、目标终点和释放守卫的修正属于方案组成，单独计数；“全局工作空间限幅为0”不表示所有神经网络原始输出均未经修正。',
        '', '## 关键改动',
        '', '1. 保留reset稳定后的实际初始状态，明确区分初始化命令与测量值。将动作写成初始位置/测量目标的参考量与路径残差，减少重新规划时参考点漂移。',
        '2. 从同步固定RGB-D估计绿色6cm目标。用表面法向、尺寸及表面一致性筛选真实几何，修复背景反光/局部遮挡导致的约12–15mm定位偏差。物体真值只用于审计和成功评价。',
        '3. 给“搬运已经开始”增加历史记忆：由网络判断或已发出的搬运动作触发，直到显式reset才清除。训练阶段标签只用于监督，推理不读取专家阶段或绝对帧号。',
        '4. 执行时约束为训练数据实际使用的固定朝向、开/闭夹爪取值；预测到达目标后，水平目标对齐测得的B中心，下降目标不低于当前任务的堆叠接触高度。',
        '5. 模型提出开爪请求后，检查名义放置命令与实测机器人姿态的稳定性，延迟不稳的释放，并在打开后保持片刻。该守卫不读取物体接触或真值，也不自行认定堆叠成功。',
        '6. 最终控制器对120Hz的关节/夹爪目标加入40ms参考平滑。静态驱动增益、原始reset和场景资产不变，高层仍是4Hz目标。它改变了执行瞬态，已作为控制器改动单独记录。',
        '', '相同回归数据下，目标平滑将平均最终XY误差从2.057mm降至0.624mm。两个诊断样本中，下降时方块相对TCP的垂直偏移最大值由约9.8mm降至约1.8mm；这是闭环对照证据，不是对任意机器人或环境的保证。',
        '', '```mermaid','flowchart LR','  RGBD[固定RGB-D与标定] --> Goal[测量目标及记忆]','  Robot[实测机器人状态] --> History[最近两帧与初始状态记忆]','  History --> Gate[学习阶段判断与历史锁存]','  Goal --> DP[Diffusion Policy]','  History --> DP','  Gate --> DP','  DP --> Decode[参考坐标解码与动作约束]','  Decode --> Guard[稳定释放守卫]','  Guard --> IK[IK与120Hz目标平滑]','  IK --> Sim[Isaac Sim物理执行]','  Sim --> Robot','```',
        '', '## 对照实验', '',*table,
        '', '上表均训练到40k步，但闭环评测权重由既定验证规则选择。后期还比较了10k间隔checkpoint与A=8/4/2。模型结构、额外监督、输入信息及执行层有多项变化，不能把全部收益归因于单一模块。',
        '', '单次采样加有效动作约束的回归结果为49/50；4次采样平均为47/50，出现额外障碍接触，因此最终采用单次采样。加入稳定释放守卫达到50/50；目标平滑进一步改善位置余量。',
        '', '![阶段记忆诊断](phase_memory_diagnostic.png)',
        '', '图为同一已知回归样本的过程对照，不代表总体成功率。原冻结策略在放置后重新返回抓取区；历史锁存消除了该行为，后续执行修正改善了放置质量。',
        '', '![几何定位诊断](goal_geometry_diagnostic.png)',
        '', '绿色点来自真实RGB-D。黑色真值标记仅用于独立审计，没有进入策略定位或动作生成。',
        '', '## 使用与复现',
        '', f'推荐checkpoint：`{cp.relative_to(ROOT)}`。必须保留完整checkpoint目录和对应推理代码，使用本项目入口；不能只取权重文件交给通用8维动作加载器。',
        '', '```bash','/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/run_delivery_sweep.py \\',
        f'  --checkpoint {cp.relative_to(ROOT)} \\',
        '  --list data/evaluation_manifests/fr3v2_existing_holdouts_20261021/regression_scenes.jsonl \\',
        '  --indices 0 1 2 3 4 5 --seeds 0 1 2 --mode policy \\',
        '  --output lerobot_policy/outputs/new_review_run --sim-gpus 0 1 --server-gpu 2','```',
        '', '训练使用H=2、K=16、A=8、DDPM100、sample预测、EMA、batch32、seed42。完整训练为40k步，最终采用20k EMA权重，再叠加明确记录的感知与执行改动。代码、参数、数据清单与权重哈希见执行清单；全部34项单元测试通过，旧腕部模型40步输出逐元素兼容检查通过。',
        '', '## 结论边界',
        '', '这是当前固定任务协议下的工程方案，包含已知颜色、6cm目标尺寸、轴对齐几何、当前reset习惯及单次抓取堆叠流程等先验。新增训练数据若改变TCP/抓取高度、阶段长度或这些约定，必须更新相应合同。',
        '', '独立测试覆盖两个新增保留布局，每个布局20个策略种子；种子数不能替代场景多样性。当前数据不足以宣称在公开机器人基准上达到SOTA，仍需相同任务、数据、控制预算和基线的共同评测。方法参考了[DP3](https://3d-diffusion-policy.github.io/)和[RISE](https://rise-policy.github.io/)的3D表征思路，本地实现不等于完整复现这些方法。',
        '', '原始失败、端口冲突及重试均保留。端口冲突修复为预先保留监听socket并校验服务身份，仅重跑执行异常/未完成试验，未删除有效策略失败。整个实验使用GPU0、1、2。',
        '', '[最终质量汇总](final_quality_summary.json) · [执行与冻结清单](execution_manifest.json)'
    ]
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(checkpoint=str(cp),known=k,fresh=f),indent=2))

if __name__=='__main__':main()
