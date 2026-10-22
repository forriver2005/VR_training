> 跨设备使用请以 [repro/README.md](../repro/README.md) 为准；本页保留历史实验路径和入口说明。

# FR3V2 / Isaac Sim / Diffusion Policy 运行入口

所有命令从 `/home/houshengyuan/DexResearch/DemoGrasp` 执行。当前质量、推荐权重、数据角色和局限以
[实验报告](../reports/existing398_quality_20261021/report.md) 和
[机器可读汇总](../reports/existing398_quality_20261021/final_quality_summary.json) 为准。
`outputs/existing398_20261021/recommended_checkpoint.txt` 保存已完成验证的推荐目录；运行中的候选不会提前替换它。

## 直接运行当前推荐策略

```bash
policy_checkpoint=$(cat lerobot_policy/outputs/existing398_20261021/recommended_checkpoint.txt)
/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/run_delivery_sweep.py \
  --checkpoint "$policy_checkpoint" \
  --list data/evaluation_manifests/fr3v2_existing_holdouts_20261021/all_known_scenes.jsonl \
  --indices 0 4 5 6 7 --seeds 0 --mode policy \
  --output lerobot_policy/outputs/my_review_run --sim-gpus 0 1 --server-gpu 2
```

输出目录必须不存在，避免覆盖已有证据。脚本分别启动 Isaac Sim 环境与 LeRobot 推理环境，通过本机 socket 连接。
`--sim-gpus` 是物理GPU编号，`--server-gpu` 是推理GPU编号；请选空闲设备，总使用量不要超过6张。
多次填写同一GPU可启动多个独立仿真进程；这与多GPU训练不同。

每次运行先恢复场景和实测关节reset，稳定后读取真实首帧；随后40次4Hz策略动作，最后固定物理窗口检查稳定性。
H=2、K=16、A=8：每步更新两帧历史；动作队列空时预测16步，取对齐后的8步执行。目标记忆、阶段锁存和执行守卫每次episode显式reset。
物体真值只用于场景恢复、诊断和评分，不作为策略/守卫输入。

当前最佳方向是固定RGB-D目标几何与机器人状态条件下的Diffusion Policy，带动作约束、稳定释放和搬运净空守卫。
它有明确任务先验：绿色轴对齐6cm目标、4cm被抓方块、固定朝向和开闭夹爪标签。
它不是纯PointNet模型，也不能直接保证任意物体颜色、尺寸、障碍排列或真实机器人表现。

## 重训当前神经网络的原始配置

这是现有权重的训练来源，不会自动得到全部后处理配置。选择新输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
/home/houshengyuan/anaconda3/envs/lerobot/bin/python lerobot_policy/train_delivery_diffusion.py \
  --list data/training_manifests/fr3v2_cube40_cube60_20261021/train_4hz.jsonl \
  --output lerobot_policy/outputs/my_canonical_training \
  --steps 40000 --save-every 10000 --batch-size 32 --seed 42 \
  --prediction-type sample --timer-mode bounded --pointcloud-source wrist_depth \
  --observation-frame initial_tcp --encoder proprioceptive \
  --goal-memory --goal-recipe v1 --episode-goal-3d-actions \
  --goal-state-switch --learned-goal-gate --position-loss-weight 6
```

训练只使用398条、4个布局，严格按episode构造窗口。当前权重来自20k EMA，由原验证协议选取。
原模型训练使用v1目标估计，后续推理修复为v2；这里保留原始训练来源，不将未经验证的v2重训冒充已测模型。
`run_config.json` 保存数据校验和及网络/动作约定。训练阶段使用专家帧号构造额外监督，推理不输入帧号；
新数据若改变阶段长度、TCP、接触高度、颜色或尺寸，必须重新审核这些标签和约定。

最终推理配置的来源依次为：

1. `derive_goal_recipe_checkpoint.py`：更新RGB-D目标几何。
2. `derive_monotonic_gate_checkpoint.py`：增加搬运阶段记忆。
3. `derive_action_steps_checkpoint.py`：动作块长度、采样次数和有效动作约束。
4. `derive_release_guard_checkpoint.py`：稳定释放。
5. `derive_smooth_controller_checkpoint.py`：120Hz参考平滑。
6. `derive_transport_guard_checkpoint.py`：目标高度相关的搬运净空。

这些工具生成新目录并保留父配置哈希，不修改神经网络权重。
新权重必须重新验证，不能仅复制旧模型的成绩。现成策略请直接使用推荐目录，避免漏掉任一执行配置。

## 查看质量和保留独立测试

每次运行的 `summary.json` 保存全部试验；每条轨迹保存 `result.json`、`rollout.npz`、`rollout.gif`。
成功需要水平误差≤5mm、高度差50±4mm、稳定、抓起和释放、堆叠接触、无禁用障碍接触以及IK正常。
执行约束的修正次数单独记录，不能将全局限幅为0解释为网络输出未被修正。

当前已提供且合格的布局全部产生过开发反馈，后续重跑属于回归。`evaluation_only` / `scene_only_evaluation` 清单禁止用于训练。
下一批数据需按完整物理布局留出未调试集合，冻结模型后只测试一次并保留全部有效失败；重复seed不能替代增加新布局。
公共SOTA需要指定共同基准、相同观测和评分协议，并比较对应方法。

仿真在等待策略时暂停物理步进。仿真4Hz不等于墙钟实时4Hz；如需部署，需另外测量感知、通信、规划和IK的完整延迟。
