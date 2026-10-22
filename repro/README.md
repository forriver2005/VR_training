# Isaac Sim FR3V2 Diffusion Policy：当前最佳方案复现

本目录交付2026-10-22冻结的 **RGB-D目标几何＋状态条件Diffusion Policy＋执行约束**。
原始完整系统在24个场景配置、138个场景/种子对上通过138/138，平均最终水平误差1.013mm，最大4.715mm。
所有这些场景都已产生开发反馈，因此这是**回归成绩，不是独立泛化成绩或公开SOTA证明**。
神经网络没有使用PointNet特征；不能将该成绩描述为纯PointNet或纯端到端模型的成绩。

## 交付内容

| 内容 | 位置／获取方式 |
|---|---|
| 网络、训练、推理、控制、评估及测试源码 | `lerobot_policy/`，随Git提交 |
| Isaac Sim运行时代码 | `lerobot_policy/isaacsim_runtime/`，随Git提交，保留原始字节 |
| 推荐20k EMA权重及完整配置 | `repro/checkpoint/`，约22MB，随Git提交 |
| 训练/验证/评测划分 | `data/*_manifests/`，随Git提交 |
| 原138次完整物理评分JSON | `repro/evidence/`，随Git提交 |
| 场景USD、贴图、URDF及相关资产 | GitHub Release的`runtime`资源组 |
| 24个评测reset/输入与一个动作复现样本 | Release的`evaluation`资源组 |
| 398条原始训练＋100条原始验证轨迹 | Release的`training`资源组，与evaluation中的重叠文件去重 |
| 可选预处理缓存 | Release的`cache`资源组；原始深度重建的点云和v1目标特征，含源数据/代码校验和 |
| 所有资源的文件级/压缩包级SHA256 | `repro/artifacts.json` |
| 发行源码与权重SHA256 | `repro/source_manifest.json` |

默认资源约1.83GB压缩文件，解压约3GB；完整训练数据另需十余GB。具体字节数以`artifacts.json`为准。
资源包保留原始NPZ、USD、URDF字节。它们不包含Isaac Sim软件本身；仿真软件通过NVIDIA官方安装源安装。
原始资产沿用各自来源许可，本分支没有为第三方资产重新授予许可。

## 1. 克隆分支

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  --branch repro/isaacsim-diffusion-best-20261022 --single-branch \
  https://github.com/forriver2005/VR_training.git
cd VR_training
```

该方法的权重直接在Git中。`GIT_LFS_SKIP_SMUDGE=1`用于跳过仓库历史ACT分支的无关LFS导出包，不影响当前方案。
后续所有命令在仓库根目录执行，路径可以与原作者机器完全不同。

## 2. 环境

当前交付入口针对 **Linux x86_64**。需要NVIDIA RTX GPU及可用Vulkan/驱动；原测试设备为RTX 6000 Ada、驱动595.84。
支持一张GPU同时运行仿真和策略，也支持分开设备；建议至少16GB显存，并按实际场景启动情况调整并发数。
原测试为Isaac Sim **6.0.0.0**；不要用5.1、6.0.0.1或其他版本替代后直接声称复现了相同物理协议。
硬件、驱动和渲染器差异可能影响RGB-D与物理结果，应保留实际成绩，不能保证任意机器逐位一致或必然138/138。

两个Python环境必须分开：学习环境Python3.10，仿真环境Python3.12。
如系统缺少相应Python/venv，请先安装，或用Conda创建对应版本的独立环境。
学习环境的`evdev`依赖需要C编译器和Linux输入设备头文件；Ubuntu通常由`build-essential`、`linux-libc-dev`及对应Python开发头文件提供。

```bash
python3.10 -m venv .venvs/learning
.venvs/learning/bin/python -m pip install --upgrade pip
.venvs/learning/bin/python -m pip install setuptools==80.9.0 wheel==0.45.1
.venvs/learning/bin/python -m pip install --no-build-isolation evdev==1.9.3
.venvs/learning/bin/python -m pip install -r repro/learning-environment.lock.txt

python3.12 -m venv .venvs/isaacsim
.venvs/isaacsim/bin/python -m pip install --upgrade pip
.venvs/isaacsim/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com -r repro/sim-environment.lock.txt
```

锁文件记录实际运行环境的完整包版本，而不只是顶层依赖。主要版本为PyTorch2.10.0、LeRobot0.4.4、Diffusers0.35.2；
学习NumPy2.2.6；仿真NumPy2.3.1、IKPy3.4.2。原训练使用TF32，不启用AMP。
先安装指定构建工具，再安装`evdev`，是干净环境验证中发现的必要步骤；默认隔离构建曾因找不到`genecodes_c.py`失败。
这不修改策略或依赖版本，只固定该外设包的构建方式。
Isaac Sim的安装和运行受NVIDIA软件条款约束；启动器按原流程设置`OMNI_KIT_ACCEPT_EULA=YES`。
参阅[NVIDIA Python安装文档](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)。

如果已有对应环境，直接指定其解释器，无需安装到`.venvs`：

```bash
export DEMOGRASP_LEARNING_PYTHON=/absolute/path/to/learning/bin/python
export DEMOGRASP_ISAAC_PYTHON=/absolute/path/to/isaacsim/bin/python
```

未设置时，学习端使用启动脚本的Python；仿真端使用当前仓库`.venvs/isaacsim/bin/python`。
下面命令按默认`.venvs`布局书写；使用已有环境时，将`.venvs/learning/bin/python`替换为其解释器。

## 3. 下载资源、检查环境

```bash
python3 repro/fetch_artifacts.py --sets runtime evaluation
.venvs/learning/bin/python repro/run.py doctor
.venvs/learning/bin/python repro/run.py unit
```

下载器支持重用已验证文件和HTTP断点续传，检查每个压缩包及解压文件SHA256；遇到不同内容的本地文件会停止，不覆盖用户数据。
已有离线资源包时用`--local-source /path/to/archives`。再次核对可加`--verify-only`。
`doctor`检查发行文件哈希、两个环境的主要版本及GPU。`unit`运行不依赖完整原始数据的几何、时序、执行约束和资源安装测试。

## 4. 快速测试和完整回归

先运行3个代表配置：普通布局、高目标布局和局部组合。

```bash
.venvs/learning/bin/python repro/run.py smoke \
  --output repro-runs/smoke --sim-gpus 0 --server-gpu 0
```

完整复测138个固定场景/种子对：

```bash
.venvs/learning/bin/python repro/run.py regression \
  --output repro-runs/regression --sim-gpus 0 1 --server-gpu 2
```

单卡机器可改为`--sim-gpus 0 --server-gpu 0`。编号是物理GPU编号；不要将已有`CUDA_VISIBLE_DEVICES`映射后的相对编号混用。
重复填写同一仿真GPU可增加独立进程数，例如`--sim-gpus 0 0 0 1 1 1 --server-gpu 2`；显存不足时降低并发。
入口限制最多使用六张不同的GPU。输出目录必须不存在。

每次运行先恢复场景和实测reset，240步稳定，然后40个4Hz动作，最后120个物理步评估，取末60步稳定窗口。
成功必须同时满足：实际抓起超过初始高度5cm、水平误差≤5mm、中心高度差50±4mm、漂移<1mm、速度<50mm/s、
两指最终分别≥35mm、检测到堆叠接触、无禁用盒障碍接触、无IK失败。预先登记的合法支撑接触除外。
每条轨迹保存`result.json`、`rollout.npz`、`rollout.gif`；根目录`summary.json`保留所有成功、有效失败和运行异常。
运行异常不能混同策略失败，不能通过删除有效失败或放松5mm阈值修改成绩。

四次采样重新规划原测均值314ms。仿真等待推理时暂停物理步进，因此仿真4Hz不表示墙钟实时4Hz。

## 5. 从头训练与生成最终配置

先补充原始训练和验证数据；必须同时保留runtime/evaluation，因为存在文件共享。

```bash
python3 repro/fetch_artifacts.py --sets runtime evaluation training cache
.venvs/learning/bin/python repro/run.py train \
  --output repro-runs/training --gpu 0
```

默认准确对应原训练方案：398条/4布局、40k步、batch32、seed42、AdamW1e-4、500步warmup＋余弦下降、
EMA power0.75、H2/K16/A8、DDPM100、sample预测、位置损失权重6、阶段BCE权重0.05；每10k保存。
窗口不跨episode，边界重复补齐且参与损失，无数据增强。训练目标和观测的固定归一化写入checkpoint。
`cache`不是新增训练数据，只省去确定性的深度反投影/FPS及目标特征预处理。不下载它也能训练，但首次处理398条轨迹会较慢。

历史最佳选在20k EMA；它来自**计划总步数40k**的学习率调度。把`--steps`改成20k会改变调度，不能视为相同训练。
训练时使用v1目标几何，最终推理切换v2；这里保留实际权重来源，不把尚未评估的v2重训称为已验证方案。
调试可使用`--steps 2 --save-every 1 --batch-size 2`，但这种权重不具有报告中的质量。

根据选定EMA生成相同最终执行配置：

```bash
.venvs/learning/bin/python repro/run.py derive \
  --checkpoint repro-runs/training/checkpoints/020000/ema \
  --output repro-runs/derived-policy
.venvs/learning/bin/python repro/run.py smoke \
  --checkpoint repro-runs/derived-policy \
  --output repro-runs/retrained-smoke --sim-gpus 0 --server-gpu 0
```

派生工具依次加入目标几何v2、阶段锁存、四次采样及有效动作约束、稳定释放、关节平滑、搬运净空；不修改神经权重。
它保留各阶段父配置和校验和。新训练权重应独立验证，不能直接继承原权重的138/138成绩。
原选模协议是对10k/20k/30k/40k在原验证布局种子0/1/2进行闭环检查，再比较A8/4/2；
依据成功数、实际抓起数、最终XY误差排序。验证集不得用当前回归测试集替代。

## 方法的组成与适用范围

网络为状态MLP8→64→64、目标MLP8→64→64、阶段MLP20→64→64→1，以及条件时序U-Net[64,128,256]。
动作9维＝XYZ残差＋四元数＋夹爪＋搬运进度，解码为8维物理动作。
参考位置＝初始实测TCP＋[α,α,β]×(测量B−初始TCP)，β=clip((7α−4)/3,0,1)。
训练进度标签α=clip((专家帧号−19)/7,0,1)，阶段标签为专家帧号≥20；推理不输入专家帧号。
改变轨迹阶段长度时必须重新审核这些标签。

当前任务先验及规则包括：绿色轴对齐6cm目标、4cm被抓方块、首次有效RGB-D目标记忆、初始实测状态记忆、
两帧有效标记和指令保持计时、搬运阶段锁存、朝下固定姿态、开/闭夹爪投影、α端点修正、终点XY对齐及高度下界、
稳定释放等待、目标高度相关净空、40ms关节目标平滑、四次采样平均及工作空间限幅。
它不持续用点云跟踪A/B，不读取物体真值或接触作为策略/守卫输入。真值仅用于reset、诊断和独立评分。
这些规则会实际改变网络输出，尤其是路线、高度、终点和释放时机。该系统不能代表纯端到端视觉学习的性能。

## 证据与历史路径

[原始实验报告](../reports/existing398_quality_20261021/report.md)保留了最初独立测试2/20、后续保留布局首次测试19/40、
单次采样局部组合46/48等结果；这些失败用于调试后，相应布局不再是独立测试集。
16组局部组合没有加入训练，但也已用于采样方案选择。重复seed不等于增加独立布局。

历史报告、checkpoint中的`parent`、缓存来源和评分JSON可能包含原机器绝对路径，它们是**原始来源记录**，不是新入口的运行依赖。
请使用本文件的命令、`repro/checkpoint`和已提交的相对清单。`run_*study.py`等脚本保留为完整实验源码，
可能依赖历史中间实验目录；重现当前方法使用`repro/run.py`，无需重跑所有研究分支。
完整资源包覆盖实际用于训练/验证和24配置回归的输入，不包括全部900条上传数据、所有弃用实验录像或NVIDIA运行时软件。

若用于“100条VR示教→1万条轨迹增强”的论文，请另行冻结公平对照配置和独立测试集；当前报告不提供该增强方法的效果证明。
