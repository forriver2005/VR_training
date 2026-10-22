"""Publish a completed regression, preserving every earlier independent-test failure."""
import hashlib
import json
from pathlib import Path

from delivery_policy_contract import ROOT
from finalize_existing_study_report import aggregate_cases

OUT = ROOT / 'lerobot_policy/outputs/existing398_20261021'
REPORT = ROOT / 'reports/existing398_quality_20261021'
MANIFEST = ROOT / 'data/evaluation_manifests/fr3v2_existing_holdouts_20261021'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT / 'frozen_ensemble_candidate.json').read_text())
    checkpoint = Path(frozen['checkpoint'])
    for name, expected in frozen['checkpoint_sha256'].items():
        assert sha(checkpoint / name) == expected, name
    for name, expected in frozen['source_sha256'].items():
        assert sha(ROOT / 'lerobot_policy' / name) == expected, name
    result_path = OUT / 'complete_ensemble_regression/summary.json'
    result = json.loads(result_path.read_text())
    assert result['completed'] == result['requested'] == 138
    assert result['runtime_errors'] == 0
    actual_pairs = {(t['index'], t['seed']) for t in result['trials']}
    expected_pairs = {(t['index'], t['seed']) for t in
                      json.loads((MANIFEST / 'all_known_tasks.json').read_text())}
    expected_pairs |= {(i, seed) for i in range(8, 24) for seed in range(3)}
    assert len(result['trials']) == len(actual_pairs) == 138
    assert actual_pairs == expected_pairs
    scenes = [json.loads(line) for line in
              (MANIFEST / 'complete_regression_scenes.jsonl').read_text().splitlines()]
    assert result['successes'] == 138, 'Do not promote an incomplete or failed candidate'
    for trial in result['trials']:
        folder = Path(trial['result']).parent
        assert all((folder / n).is_file() for n in ['completed.json', 'rollout.npz', 'rollout.gif'])
        physical = json.loads(Path(trial['result']).read_text())
        assert trial['uid'] == physical['uid'] == scenes[trial['index']]['uid']
        assert trial['seed'] == physical['seed']
        assert json.loads((folder / 'completed.json').read_text())['frames'] == 40
        assert physical['success'] == trial['success']
        assert physical['action_samples'] == 4
        assert physical['control_dt_s'] == .25
    def subgroup(predicate):
        trials = [t for t in result['trials'] if predicate(t['index'])]
        return aggregate_cases(dict(completed=len(trials), trials=trials))
    total = aggregate_cases(result)
    known = subgroup(lambda i: i < 8)
    combinations = subgroup(lambda i: i >= 8)
    physical_results = [json.loads(Path(t['result']).read_text()) for t in result['trials']]
    maximum_goal_error_mm = max(r['goal_memory_audit']['initial_position_error_m']
                                for r in physical_results) * 1000
    contract = json.loads((checkpoint / 'run_config.json').read_text())
    latency = json.loads((REPORT / 'inference_latency_n4.json').read_text())
    single_latency = json.loads((REPORT / 'inference_latency_n1.json').read_text())
    assert Path(latency['checkpoint']) == checkpoint
    assert latency['reference_action_max_absolute_difference'] < 1e-6
    quality = dict(
        checkpoint=str(checkpoint), status='complete regression; no public SOTA claim',
        training_episodes=398, training_layouts=4, new_training_demonstrations=0,
        complete_regression=total, known_layout_regression=known,
        maximum_goal_position_error_mm_diagnostic_only=maximum_goal_error_mm,
        local_combination_regression=combinations, groups=result['groups'],
        first_frozen_test=dict(successes=2, count=20, subsequently_used_for_debugging=True),
        second_frozen_test=dict(successes=19, count=40, subsequently_used_for_debugging=True),
        previous_single_sample=dict(known_successes=90, known_count=90,
                                    local_successes=46, local_count=48),
        independent_backgrounds_remaining=0, public_SOTA_claim=False,
        policy_latency={k: v for k, v in latency.items() if k != 'timings'},
        actual_algorithm=dict(
            encoder=contract['encoder_type'], visual_input='measured fixed RGB-D goal geometry',
            H=2, K=16, A=8, action_samples=4, DDPM_steps=100, EMA=True,
            selected_training_step=20000, total_training_steps=40000,
            control_dt_s=.25, action_count=40,
            goal_memory=contract['goal_memory'],
            monotonic_goal_gate=contract['monotonic_goal_gate'],
            release_guard=contract['release_guard'], controller_filter=contract['controller_filter'],
            transport_clearance_guard=contract['transport_clearance_guard']),
        limits=['All 24 configurations now have development feedback; repeated seeds are not new scenes.',
                '16 combinations share four backgrounds; no broad unseen-background claim.',
                'Known green 6cm axis-aligned target and task-specific action/control constraints.',
                'No simulator object/contact truth is used as policy or guard input.'])
    evidence = [result_path, OUT / 'ensemble_final_plan.json',
                OUT / 'ensemble_full_remaining/summary.json', OUT / 'ensemble_polish_probe/summary.json',
                OUT / 'height_aware_regression/summary.json', OUT / 'local_combination_policy/summary.json',
                OUT / 'final_test/summary.json', OUT / 'fresh_holdout_policy/summary.json',
                OUT / 'height_policy_unit_tests.log', MANIFEST / 'complete_regression_scenes.jsonl',
                MANIFEST / 'complete_ensemble_remaining_tasks.json',
                REPORT / 'inference_latency_n1.json', REPORT / 'inference_latency_n4.json',
                REPORT / 'learning_environment.json', REPORT / 'simulation_environment.json',
                REPORT / 'final_policy_examples.png',
                ROOT / 'lerobot_policy/finalize_ensemble_report.py',
                ROOT / 'lerobot_policy/benchmark_delivery_inference.py',
                ROOT / 'lerobot_policy/run_final_ensemble_check.py']
    execution = dict(frozen=frozen, evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in evidence},
                     GPUs_used=[0, 1, 2], source_assets_modified=False,
                     coverage='136 full-regression cases plus two identical-contract probes; no duplicate cases')
    (REPORT / 'final_quality_summary.json').write_text(json.dumps(quality, indent=2) + '\n')
    (REPORT / 'execution_manifest.json').write_text(json.dumps(execution, indent=2) + '\n')
    (OUT / 'recommended_checkpoint.txt').write_text(str(checkpoint) + '\n')
    relative_checkpoint = str(checkpoint.relative_to(ROOT))
    text = f'''# DemoGrasp现有数据优化：完整回归结果

当前推荐系统在24个场景配置、138次固定种子闭环运行中取得 **138/138**：8个原始布局90/90，16组局部A/B组合48/48。平均最终水平误差 **{total['mean_final_xy_mm']:.3f} mm**，最大稳定窗口水平误差 **{total['max_stable_xy_mm']:.3f} mm**。成功门槛仍为5mm。

这是当前已完成实验中表现最好的任务专用方案。所有这些场景现在都产生过开发反馈，因此结果是回归成绩，**不是独立泛化成绩，也不是公开基准SOTA证明**。

## 数据与评估范围

| 部分 | 场景配置 | 成功次数 | 当前用途 |
|---|---:|---:|---|
| 原训练布局 | 4 | 20/20 | 已见布局回归 |
| 原验证布局 | 1 | 10/10 | 曾用于选模 |
| 原测试及保留布局 | 3 | 60/60 | 首轮测试后用于错误定位，现为回归 |
| 局部A/B组合 | 16 | 48/48 | 共享4个背景，最终采样选择已使用其反馈 |

训练始终只使用原有 **398条、4个布局**。16组局部布局在用户接手数据生成前已采集并隔离，本轮没有加入训练。两个保留包的原演示为8Hz，只复用资产和实测reset，在统一4Hz、40步协议下执行策略；未重标演示或混入训练。原17号数据组未通过当前运行时的专家回放，继续隔离。

必须保留的历史结果：第一版冻结模型独立测试 **2/20**；第二版在另两个保留布局的首次测试 **19/40**。这些失败引导了后续修复，最终结果不能重新称为独立测试。单次采样完整执行版为90/90和46/48，两次失败水平误差5.23mm、5.33mm；它们保留在原始结果中，没有调整门槛或删除。

当前四次采样版包含136次整组复测及2次相同配置探针，合计138个不同的场景/种子对；没有把探针重复计数。不同随机种子并不等同于新场景。

## 物理质量

| 指标 | 原始8布局，90次 | 局部16组合，48次 |
|---|---:|---:|
| 平均最终水平误差 | {known['mean_final_xy_mm']:.3f} mm | {combinations['mean_final_xy_mm']:.3f} mm |
| 最大稳定窗口水平误差 | {known['max_stable_xy_mm']:.3f} mm | {combinations['max_stable_xy_mm']:.3f} mm |
| 最大高度误差 | {known['max_height_error_mm']:.3f} mm | {combinations['max_height_error_mm']:.3f} mm |
| 最大稳定漂移 | {known['max_drift_mm']:.3f} mm | {combinations['max_drift_mm']:.3f} mm |
| 最大稳定窗口速度 | {known['max_speed_mm_s']:.3f} mm/s | {combinations['max_speed_mm_s']:.3f} mm/s |
| IK失败 | {known['ik_failures']} | {combinations['ik_failures']} |
| 全局工作空间额外限幅 | {known['workspace_clips']} | {combinations['workspace_clips']} |
| 禁用障碍接触试验数 | {known['forbidden_contact_trials']} | {combinations['forbidden_contact_trials']} |

验收条件保持：水平误差≤5mm、中心高度差50±4mm、漂移<1mm、速度<50mm/s、实际抓起/释放/堆叠接触、无禁用盒障碍接触、IK正常。40次4Hz动作后使用原固定物理稳定窗口。执行约束有单独计数；全局限幅为0不代表网络原始输出未经修正。

仍存在接近阈值的样本：RGB-D初始目标定位的最大三维误差为 {maximum_goal_error_mm:.3f}mm（仅在执行后与真值审计）。原保留布局的一次最终水平误差约4.72mm，其目标定位误差约4.35mm，说明满分回归不代表已经没有感知误差，也不能推断任意新遮挡下仍有足够余量。

## 当前网络与执行方法

环境为Isaac Sim6、FR3V2、4cm/6cm方块和盒障碍。最终方案是 **RGB-D目标几何＋机器人状态与初始状态记忆＋Diffusion Policy＋执行约束**。神经网络使用状态/目标MLP，没有采用PointNet特征。腕部PointNet、融合PointNet和空间注意力模型的对照保留，未将其成绩包装成最终模型成绩。

H=2、K=16、A=8，DDPM100、EMA、sample预测；训练batch32、seed42、40k步，选取20k EMA。每个控制周期更新最近两帧观测，每8个动作重新生成16步轨迹并执行对齐后的8步。当前每次并行采样4条轨迹，在归一化动作空间平均后解码；初始实测状态和首次有效RGB-D目标跨动作块保留。

1. **目标测量**：使用真实固定相机RGB-D的深度面法向与边界，修复反光/遮挡导致的12–15mm定位偏差。已知绿色、6cm和轴对齐是明确任务先验。
2. **动作参考与阶段记忆**：用初始位置、测量目标和路径残差表示动作；网络预测进度，锁存“搬运已开始”，避免开爪后返回抓取区。专家帧号只用于训练监督，不是推理输入。
3. **有效动作约束**：固定朝向、开/闭夹爪取值遵循训练数据；预测终点时对齐测量目标，下降不低于名义堆叠高度。
4. **稳定释放**：模型提出开爪后，等待实测机器人位姿稳定，再打开并保持片刻。执行层不读取物体真值或接触，不自行宣布成功。
5. **执行平滑**：120Hz关节/夹爪目标使用40ms参考平滑，减少下降时滑移；原始reset、资产和静态驱动增益不变。
6. **搬运净空**：根据测量目标高度先升高，再沿模型方向横向移动，抵达上方后下降。未使用障碍真值地图，也不是任意障碍排列的完整运动规划器。
7. **多次采样**：在完整执行方案下重新比较并采用4次平均，降低小幅抓取/放置偏差。早期缺少完整约束的4次采样曾出现更多碰撞，其47/50结果仍保留。

## 主要对照

| 方案 | 原4个训练布局 | 原验证布局 |
|---|---:|---:|
| 398条腕部PointNet基线 | 17/20 | 0/10 |
| 融合点云＋TCP参考PointNet | 3/20 | 0/10 |
| 空间注意力编码 | 0/20 | 0/10 |
| 初始/目标参考＋腕部编码 | 15/20 | 10/10 |
| 目标几何修复＋状态参考 | 20/20 | 10/10 |
| 当前完整执行系统 | 20/20 | 10/10 |

后期原验证布局用于闭环选模。架构、额外监督、输入和执行层均有变化，不能将全部收益归于某一个模块。单次采样的稳定释放版达到50/50；关节目标平滑将这组平均水平误差从2.057mm降至0.624mm。高目标布局暴露净空问题后修复，最终扩大到上述24配置回归。

![阶段记忆诊断](phase_memory_diagnostic.png)

同一回归样本的过程对照，不代表总体成功率。

![目标几何诊断](goal_geometry_diagnostic.png)

绿色点来自实际RGB-D；图中的真值标记只用于审计，不参与控制。

![当前策略的实际执行片段](final_policy_examples.png)

上行为本轮最终水平误差最大的样本；下行为单次采样阶段失败、四次采样后通过的局部组合。均为实际相机帧，展示抓取、搬运、释放和撤离。

## 复现与交付

推荐checkpoint：`{relative_checkpoint}`。包含权重、完整合同、归一化及各执行约束配置，必须结合项目推理入口加载。推荐路径也保存到 `lerobot_policy/outputs/existing398_20261021/recommended_checkpoint.txt`。

```bash
/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/run_delivery_sweep.py \\
  --checkpoint {relative_checkpoint} \\
  --list data/evaluation_manifests/fr3v2_existing_holdouts_20261021/complete_regression_scenes.jsonl \\
  --indices 0 4 5 6 7 22 23 --seeds 0 --mode policy \\
  --output lerobot_policy/outputs/new_review_run --sim-gpus 0 1 --server-gpu 2
```

35项相关单元测试通过。闭环结果逐条保留result.json、rollout.npz和rollout.gif；权重/配置/源码及证据摘要见执行清单。使用GPU0、1、2，未超过六卡上限。原始NPZ、USD、URDF和训练清单未修改。

推理耗时在 `{latency['device']}` 上完成预热后测量：单次采样重新规划均值 {single_latency['model_replanning']['mean_ms']:.1f}ms，四次采样均值 {latency['model_replanning']['mean_ms']:.1f}ms、P95 {latency['model_replanning']['p95_ms']:.1f}ms；四次采样队列内取动作均值 {latency['queued_action']['mean_ms']:.2f}ms。这里只测策略，包括CUDA同步，排除相机、RGB-D几何、通信、IK和物理。复用实测输入重新推理与保存动作最大差 {latency['reference_action_max_absolute_difference']:.3g}。仿真在等待推理时暂停，4Hz仿真控制频率不等于已验证墙钟实时4Hz。

目前没有完全未参与调试的新背景。下一批数据应先留出完整新布局再冻结系统评测，才能判断广泛泛化；公开SOTA需要明确外部基准及同协议对照。当前可准确汇报为“现有场景回归138/138、毫米级放置精度”，不能汇报为“任意新场景100%”或“已证明SOTA”。

[最终质量汇总](final_quality_summary.json) · [执行与冻结清单](execution_manifest.json) · [单次采样阶段报告](height_n1_report.md)
'''
    (REPORT / 'report.md').write_text(text)
    print(json.dumps(quality['complete_regression'], indent=2))


if __name__ == '__main__':
    main()
