# H2R task、MJLab 训练与统一评估

与 original 的适配及参数对照见 [h2r_adaptation.md](h2r_adaptation.md)。

这是公开 Sharpa 入口 `f5efe18` 的 H2R 任务适配，配方为 `contrack-public-sharpa-h2r-v2`：保留仓库内 PyTorch PPO／CMDP、双 critic、奖励分组、adaptive reset library 和阈值课程，将 Isaac Lab 环境迁至 MJLab 的 `ManagerBasedRlEnv`，物理 backend 为 `mujoco_warp`。初始运动学参考使用与 FlashCHORD 相同的 V2D IK 实现和参数，复制来源见 [SOURCE.json](../contrack_h2r/_vendor/v2d/SOURCE.json)。接触准备仍为 ConTrack H2R 配方；这不是未发布的官方 HDF5 生成链复现。

## 环境

在 ConTrack 根目录运行。环境与 benchmark 分开；无需 Isaac Sim、Isaac Lab、Newton 或其他方法的私有模块。

```bash
bash scripts/h2r_environment.sh /opt/h2r/contrack
bash scripts/h2r_ik_environments.sh /opt/h2r/contrack-ik
/opt/h2r/contrack/bin/python -m contrack_h2r.prepare --help
```

输入只有 h2r_bench 的 task 目录（`h2r-task-v1`，可以是 suite 内的 task 或独立打包的 task），由 bench 发布的加载器 `h2r_io`（依赖包 `h2r-io`）读取。三个锁文件都固定 h2r_bench 的 tag `h2r-io-v1`（`ssh://git@github.com/dex-synth/h2r_bench.git`，子目录 `packages/h2r-io`），锁文件计入源码 hash。主进程和两个准备子进程都经它读取 task。

`h2r/requirements.lock` 固定 Python 3.12 环境的依赖，含 MuJoCo／MJWarp 4.0.1、MJLab、PyTorch 2.9.0 CUDA 12.8、Warp 1.16.0、rsl-rl-lib 5.5.2。MJLab 尚无支持 MuJoCo 4 的正式版，锁文件以 git commit 固定其 `main`（`1f48001802bd89257537c2dc16979d41b9686e86`，SSH 源）；该 commit 声明 `torch>=2.14`，本环境有意保留与 h2r_bench 相同的 PyTorch 2.9.0，测试与 MuJoCo 4 suite 上的端到端检查均在此组合下进行。MuJoCo 3.11 的环境与代码保留在分支 `mj3`。训练使用 CUDA；准备和在线推理可在 CPU 执行。Python 3.12 也满足现有 runner 的 f-string 语法。按实际驱动选择设备；只设置当前进程的 `CUDA_VISIBLE_DEVICES`，不要改变其他方法的环境。

原始 MANO 解码运行于独立 Python 3.10 环境，原始 IK 运行于 Python 3.11（Pinocchio 3.7.0、Pink 4.2.0、DAQP 0.10.3）。`mano-requirements.lock` 和 `ik-requirements.lock` 固定依赖；`manotorch` 和 `chumpy` 固定源码提交与压缩包 hash。它们通过文件与主准备进程交接，训练和在线推理不导入这些包。复制的源码位于本仓库内，不需要安装完整 V2D、FlashCHORD 或 Isaac Lab。

## 输入与准备

唯一输入是 bench 封存的 task 目录：一个 demo、一个 robot 及两者装配后的完整物理模型。task 不含控制频率、评估设置或初始化规则。

```bash
/opt/h2r/contrack/bin/python -m contrack_h2r.prepare \
  --task /path/to/task --output /path/to/new-prepared \
  --mano-root /path/to/mano \
  --loader-python /opt/h2r/contrack-ik/mano/bin/python \
  --ik-python /opt/h2r/contrack-ik/ik/bin/python
```

`h2r_io.load_task` 校验 task、demo、robot 的全部文件和身份；此前格式的 bundle／task 在此被拒绝（`Expected h2r-task-v1 in …`）。机器人按 `robot.name`、`robot.hands` 和 semantics 判断：IK 与接触槽映射目前只有 `sharpa_wave`，其他手型明确拒绝；腕关节、手指关节和腕 body 取自 semantics，手的集合取自 task，因此双手、仅右手、仅左手使用同一段代码。物体按 `demo.objects` 遍历：每个自由根取其根部件的参考位姿，每个 hinge 取 `position_rad`，固定根不占广义坐标；多物体、铰接物体和额外的固定支撑不需要分支。四个数据集的包结构相同，没有按数据集区分的读取或配方。

task 的 `compiled.json` 记录编译它的 MuJoCo 版本。`prepare` 和训练要求它与运行时版本相同，训练还要求 prepared 由同一版本生成：旧版本编译的 task 在新版本中仍能载入，但带着旧版本的执行器等编译参数，因此明确拒绝，须使用 bench 为当前版本生成的 task 重新准备。共同 MJCF 的执行器须是每个关节一个、无延迟的位置输入（MuJoCo 4 的 `ctrladr`、`ctrlnum`、`input`、`delay` 逐项核对）。

方法准备包括：

- 用原始 `MANO.forward()`／`AxisLayerFK` 从 `mano_quat`、`mano_trans`、`mano_shape`（`h2r-mano-v1`：平直平均手姿、绝对旋转、平移为腕位置）生成世界坐标中的关节位置和朝向。IK 需要 task 包不提供的 AxisLayerFK 关节朝向，因此保留原始解码器而不使用 `h2r_io.mano.decode`；`left_shapedirs_x_mirrored` 按加载器的同一规则处理，解码出的手部网格与 16 个 MANO 关节须与 task 的 `vertices_m`、`joints_m` 相差不超过 0.1 mm，否则拒绝。指尖目标是 FlashCHORD 配方自己的顶点（745、317、444／左手 445、556、673），对所有数据集相同，不是 task 的指尖顶点（744、320、443、554、671）；两者的最大距离记入 `ik/mano.json`。
- 原样调用 `SharpaHandKinematics`：腕位置／朝向、五指尖位置／朝向及原始中间指节目标与权重全部保留。使用 DAQP、100 次最大迭代、200 Hz 内部迭代步长、1e-6 位置误差变化阈值、尺度 1.0，关闭 relative-frame tasks；保留原始首帧初始化和上一帧 warm start。参考时间轴不变，也不追加平滑、碰撞优化或另一套收敛条件。
- 从 task 的 robot 包提取手部子树，添加原始 task sites，在 Pinocchio 自由腕上求解，然后用 FlashCHORD 的连续腕角转换映射为共同六关节腕。随机非零构型的 FK 和限位校验贯穿 Pinocchio／MuJoCo 边界。
- 将皮肤顶点按最近 MANO 指骨分成 15 个 segment，对每个部件表面计算 5 mm 阈值内的平均局部点（`--contact-distance=0.005`），再按原生 OR／有效点均值规则合并为 10 槽。固定 distal pad 归入 DP 槽，静态桌面不占 actor 的物体接触槽。
- 使用共同 root＋hinge 广义坐标验证全部 part 的 FK。完整有限桌面、支撑、质量惯量、碰撞体、执行器和被动关节直接来自 task MJCF，不从示范重新拟合。task 的名义步长（300 Hz）不是训练步长：训练在方法侧模型中显式设置自己的物理步长，不修改封存 task。

`prepared/`（`contrack-h2r-v3`）包含 `task/`（`h2r_io.export_task` 导出的独立 task 副本，`task_id` 不变）、方法 `reference.npz`、`binding.json`、`prepare_report.json` 和 `training_physics.json`。`binding.json` 绑定 `task_id`、`demo_id`、`robot_id`、布局和方法文件的 hash；task 副本由自己的 `task_id` 封存，读取时逐文件校验。`ik/` 封存原始 MANO 特征、IK 模型、逐帧求解结果、来源、配方与环境版本；`ik/penetration.json` 记录整段碰撞诊断，包含物体之间的接触。诊断阈值 20 mm 是方法常量（`retarget.PENETRATION_REPORT_M`），只用于报告：评估的初始穿透阈值属于评估设置，方法读不到。诊断不会移动物体或修改 IK，eval 独立执行初始化检查。准备输出不覆盖输入或已有目录，失败时清理本次未完成输出。

IK 配方为 `flashchord-h2r-sharpa-ik-v1`，求解参数固定在配方中。切换 task 或 IK 后须生成新的 prepared 并重新训练，不能将 checkpoint 改绑到新参考。

## 训练与续训

```bash
CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.train \
  --prepared /path/to/prepared --output /path/to/new-training \
  --num-envs 6000 --iterations 5000 --seed 0

CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.train \
  --prepared /path/to/prepared --output /path/to/new-resumed-training \
  --num-envs 6000 --iterations 1000 --seed 0 --resume /path/to/training/model_999.pt
```

训练默认 **6000 个环境、5000 iterations、每 200 次更新保存一次**。更新数采用[论文附录 F](https://arxiv.org/html/2606.03177v3#Pt0.A6)的固定预算；并行数及保存间隔沿用原生 Sharpa README。16 steps/env、16 minibatches，每次更新采样 96,000 transitions，总计 4.8 亿 control transitions。论文按 clip 调整并行数，本文只对齐 update 数，不声明全部论文配方一致。`rsl_rl_contrack/sharpa_recipe.py` 供本 fork 的原生 Sharpa 入口与 H2R 共用；公开上游原来的 100000 默认值保留在预算来源记录中。其他预算须显式指定，`run_config.json` 保存 `budget_provenance` 和 `budget_override`；日志使用 TensorBoard。从零训练 5000 次保存最终 `model_4999.pt`。

训练时钟独立于 task 和评估：默认 120 Hz 物理、固定 60 Hz 策略，都由训练入口显式设置；task 的名义步长（300 Hz）和评估步长不进入训练。奖励的控制区间、episode 时长、参考差分和未来参考间隔都按 60 Hz 策略周期计算，不随 task 步长变化。`--physics-hz` 只接受与 60 Hz 精确兼容的频率；非 60 Hz `--control-hz` 明确拒绝。MJLab 负责物理步进、CUDA graph、manager 调度与批量 reset，官方 `RslRlVecEnvWrapper` 对接原 PPO/CMDP；不维护自写 VecEnv。共同 Sharpa 每手 6 腕＋22 手指，原生 58 维双臂 checkpoint 不能复用。

`rsl_rl_contrack/mimic.py` 从原生 Sharpa MDP 提取共同函数，原生入口和 H2R 使用同一参考采样、中心差分及 reset 抽样次序。先采样到 60 Hz 控制网格：位置线性插值、四元数最短弧 nlerp、接触最近邻（半帧取后一帧），然后按原生首尾单边／内部中心差分计算速度。`reset_reference.py` 将世界角速度转换为自由根局部广义速度，保留共同铰链约束，并在准备报告中记录部件速度与独立差分参考的差异。完整任务时长必须能被控制周期整除，否则拒绝准备，不取整。

策略观察当前真实状态／接触、当前＋5 个未来控制帧参考、参考接触和 phase。参考预采样一次，运行时按整数控制帧索引，训练与客户端共用同一特征和动作解码。
腕平移／旋转／手指残差尺度分别为 0.06666666666666667 m、0.21333333333333335 rad、0.25 rad；动作裁剪到 `[-1,1]`，因此每轴范围为上述尺度的正负值。腕范围对齐 LeanRetarget 的 Sharpa `joint_scales`，属于 H2R 参数适配，策略仍为 60 Hz。原 arm 跟踪与惩罚项改用完整六维腕，平移／旋转的归一化尺度显式保存在 layout，默认分别为 1 m／1 rad。奖励延续原 manager 的 action-time 参考：动作施加后的状态与该动作对应的参考比较，再推进下一观测时间。训练按原生顺序使用动作帧判断终止并记录 post-step 历史，再推进非终止帧；包含末帧参考动作，timeout 与 pose failure 可同时成立。声明的 episode 时长沿用 `raw_len/raw_fps`。这些方法私有训练时钟不改变共同任务时长，评估客户端在 bench 终点不再输出动作。实际接触点使用最后求解时的物体局部接触中点均值；这是从 PhysX sensor 到 MuJoCo manifold 的接触表示适配。

接触阈值与动作尺度分别封存在 `binding.json` 的 `preparation.contact_distance_m` 和 `layout.action_scales`。

`rg/rs/rp` 各乘一次实际控制区间，PPO 使用 `(rg+rp, rs+rp)`。训练 reset 沿用原生配方：从 adaptive reset library 抽取起始帧（含第 0 帧），并恢复该帧的参考速度。评估固定从参考第 0 帧、全部速度为零开始，没有 warm-up；客户端只提交首帧机器人位置。这一训练／评估初态差异是已接受的原生行为，训练配方不为此修改；`native_rollout --initial-velocity zero` 可在训练 setup 中单独观察它的影响。原生训练随机化默认关闭，当前 H2R 配方也关闭；共同 `sim-dr` 扰动由 bench 执行。

默认路径保留两项原生限制：速度历史缓存只在 reset 时更新（原生 indexed `copy_` 不写回），空 episode return 窗口触发原生 `ZeroDivisionError`。PPO 文件已恢复为 `f5efe18` 原实现；不再隐式启用 empty-return guard 或逐步更新速度缓存的变体。失败按实际原因记录，不能暗改预算或 reset 来消除。

GPU 参数回读报告保存在训练目录。默认每 world 接触容量 512、约束容量 4096，可通过 `--contacts-per-world`／`--constraints-per-world` 显式增大，溢出直接报错。MJWarp 4 在 `Data.overflow` 中同时标记求解器和 line search 达到迭代上限；这是共同求解器设置（task MJCF 的 `iterations`）的正常结果，不计为容量溢出，只有容量位报错。MJLab 原生捕获 step／forward／reset 的 CUDA graph。每个物理子步通过 metrics term 在 GPU 累积容量溢出标志，控制步结束统一检查；真实接触统计和奖励留在任务层。

每个控制步只保留 MJLab 尾部的一次完整 `forward`，用于可能发生的 reset 后刷新。`TaskState.finish_step()` 先提取最后求解时的局部接触点，再通过独立 CUDA graph 执行 `kinematics`、`com_pos`、`com_vel`，为奖励／终止提供当前 qpos/qvel 对应的 body 姿态和速度。轻量刷新不执行碰撞／约束求解，不写入动力学结果或 warmstart；非 graph 路径执行相同三个操作。接触相对积分后状态差一个物理子步，默认 8.33 ms，与最后 solve 的观测语义一致，不额外延迟一个 60 Hz 控制步。初始化和显式 reset 仍由 MJLab 完整刷新。`training_physics.json` 记录 `state_refresh` 与 `contact_timing`。

训练与评估物理频率可以不同。评估步长只从 `hello.dt_s` 得知（默认 300 Hz）；客户端以 `control_dt_s = 1/60` 声明 checkpoint 的策略周期，它不是评估步长的整数倍时 start 被拒绝，不改变策略频率。未来参考间隔来自 checkpoint 的训练时钟，源轨迹与计分时间仍由共同 task 决定。

部署格式为 `contrack-h2r-deployment-v4`：绑定 `prepared_id`、`task_id`、`demo_id`、`robot_id`，策略周期以秒保存（`training_physics.control_dt_s`），不保存 tick 数。每控制步只刷新运动学的实现仍由必有字段 `training_physics.state_refresh` 标识，不同刷新实现训练的 checkpoint 不混用。每个 `model_N.pt` 对应一个 `model_N.json`，包含 checkpoint hash、输入身份、观测／动作布局、normalizer 配方、训练配置和源码 hash；normalizer 数值随模型保存。checkpoint 同时保留 optimizer、CMDP、自适应库、课程与 RNG。续训核验环境、simulator（MuJoCo、MJWarp 版本与 MJLab 源 commit）、训练频率和并行规模，恢复学习状态并开始新的训练 episodes，不承诺恢复同一时刻的所有仿真缓存。仅加载自己生成或信任的 PyTorch checkpoint。此前格式的产物没有兼容读取：v2 及更早的 prepared、v3 及更早的 checkpoint 被明确拒绝，须从 task 目录重新 prepare 和 train。

`training_physics.json` 记录共同终点、方法声明的 episode 时长、首帧起训练动作数与实际物理参数。`run_config.json` 保存有效启动参数、预算 override、原生随机 episode-length 初始化及日志差异；`timing.json` 保存环境初始化、训练耗时、更新尝试数和实际控制／物理步数。6000 环境的开销须实际测量，不沿用旧 4096 环境的耗时。

## 独立统一评估

在 bench 的独立环境以同一 task 启动 `evaluate-server`（`sim`／`sim-dr`），然后运行常驻客户端：

共同资产编译版本须与评估 runtime 匹配；旧 MuJoCo 资产不能通过改写审计版本或改绑 checkpoint 进入新环境。需要迁移时，先由 bench 生成并核对新输入，再重新 prepare／train。

```bash
/opt/h2r/contrack/bin/python -m contrack_h2r.evaluate \
  --prepared /path/to/prepared --checkpoint /path/to/training/model_999.pt \
  --socket /path/to/shared/eval.sock --device cpu
```

客户端使用 `h2r-evaluation-v2` 协议（`h2r_io.client`）：核对 `hello.task_id` 与 prepared 绑定的相同，要求评估开启接触观测，按 `hello.dt_s` 换算每个策略周期的物理步数，并在 start 中声明 `control_dt_s = 1/60`、所需的物体 body 和接触。控制只使用 bench 的实际关节、body 原点世界速度及 contact records。start binding 的方法身份为 `ConTrack-H2R`，`controller` 记录 checkpoint 路径、sha256、完成更新数（`model_N` 为 `N+1`）和 `prepared_id`；bench 的评估清单须以同一身份声明。`contacts=null` 是错误，有效空列表是无接触。actor 的 normalizer 和 `tanh` 由原 `ActorCritic.act_inference()` 执行；共享解码输出所有具名关节位置目标。部署不启动物理 solver，不提交物体状态、reset、外力或方法终止条件。

## 原生闭环诊断

`contrack_h2r.native_rollout` 在训练 setup 中确定性回放一个 checkpoint，只反映训练效果：使用 checkpoint 记录的 MJLab／MJWarp 物理与 60 Hz 控制、原生观测和 `act_inference`；所有 world 从参考首帧开始，reset library 与 pose-break 阈值从 checkpoint 载入后冻结，不更新课程。

```bash
CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.native_rollout \
  --prepared /path/to/prepared --checkpoint /path/to/training/model_N.pt \
  --output /path/to/new-native-rollout --num-envs 4 --initial-velocity reference
```

`--initial-velocity reference` 使用训练 reset 的首帧参考速度，`zero` 对应 bench 的零初速。`summary.json` 记录各 world 的终止帧／原因、参考覆盖率和物体位置／角度误差，`rollout.npz` 保存逐控制步记录。GPU 接触求解不完全确定，多 world 会分叉，故分别报告。该诊断不替代 bench 的共同评估。

## 方法内 rollout 的导出

bench 在独立评估之外，还报告方法在自己训练所用仿真里的表现（`method-sim`，自报）。导出入口复用上面的原生闭环，不另写 rollout 循环：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.method_rollout \
  --prepared /path/to/prepared --checkpoint /path/to/training/model_N.pt --output /path/to/new-method-rollout
```

10 个 world 从同一个评估初态开始（MJWarp 在 GPU 上不完全确定，各算一条）：机器人取评估客户端在 start 中提交的首帧姿态，物体取 task 的参考首帧，全部速度为零，没有 warm-up；导出时逐项核对首个状态（`1e-6`），不符即拒绝。物理为 checkpoint 记录的 MJLab／MJWarp 训练步长（默认 120 Hz）与 60 Hz 控制，运行时的 MuJoCo、MJWarp 版本和 MJLab 源（commit）须与训练时相同；动作为 `act_inference`，场景只有 task 的执行器，没有随机化、外力或课程更新，world 结束后的自动重置不在导出范围内。

输出是 `h2r-method-rollout-v1` 包：每个物理步之后的机器人关节、物体自由根位姿和物体关节（首个状态为 0 s 的初态），方法身份和 `controller` 与评估客户端 start 中上报的相同。原生 pose-break 保留为方法自己的失败终止，该 world 的状态到此为止并记录 `ended_at_s`；原生闭环在参考末帧多执行的一个动作超出 task 时长，其状态不写入，但其间的 pose-break 仍计入判定。方法判定为“到达参考末帧且无 pose-break”，阈值取 checkpoint 中冻结的值。包由 bench 的 `import-method-rollout` 导入并用共同指标计分，见 h2r_bench 的 `docs/methods/contrack.md`。

## 代码与验证

| 模块 | 职责 |
| --- | --- |
| `inputs.py`、`layout.py` | 方法产物的身份与封存、具名布局 |
| `prepare.py`、`retarget.py`、`contacts.py` | 准备编排、隔离 IK 调用、穿透诊断、ConTrack 接触参考 |
| `mano_features.py`、`ik_worker.py`、`_vendor/v2d/` | 原始 MANO／IK 及输入、模型、具名关节边界转换 |
| `mjwarp_scene.py`、`mjlab_env.py` | 共同 MJCF 核验、MJLab 场景和任务 terms、真实接触；批量仿真与 VecEnv 由 MJLab 提供 |
| `rewards.py`、`resets.py` | 原生奖励公式、自适应库和课程 |
| `reference.py`、`reset_reference.py` | 控制网格参考、原生差分到共同广义状态的转换，以及训练／推理共用特征和解码 |
| `rsl_rl_contrack/mimic.py`、`sharpa_recipe.py` | 原生入口与 H2R 共用的纯张量参考／抽样逻辑及公开 Sharpa 配方 |
| `train.py` | 原 runner 的新入口与 checkpoint 上下文 |
| `policy.py` | checkpoint／JSON／prepared 身份核验与原 `ActorCritic` 构建，评估客户端与原生闭环共用 |
| `controller.py`、`evaluate.py` | 无 simulator 的反馈控制与 Unix socket 客户端 |
| `native_rollout.py` | 训练 setup 内首帧起始、冻结课程的确定性闭环诊断 |
| `method_rollout.py` | 把零初速的原生闭环按物理步导出为 bench 的方法内 rollout 包 |

```bash
/opt/h2r/contrack/bin/python -m pytest tests/h2r -q
CONTRACK_TEST_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  /opt/h2r/contrack/bin/python -m pytest tests/h2r -q
```

测试从固定 `f5efe18` Git 对象读取未改动的原生 callback 作为 oracle，覆盖参考采样／速度、动作、历史槽、随机流、终止及课程的多步对照；也包含原生奖励 callback 数值对照、参考终点与四元数符号、task 包输入与篡改、v2 握手、真实接触、body 原点速度、mask reset、MJLab graph／eager 数值对照、训练／评估时钟分离、子步容量溢出、PPO／CMDP 续训和完整控制输出一致性。GPU 测试需显式启用。短训练与闭环检查验证接入正确性，不证明论文性能或任务成功；实际任务结果与已验证组合保留在对应运行目录。

IK 测试还检查固定源码、世界坐标转换和只读穿透诊断。提供独立原 FlashCHORD 与 ConTrack 完整 ARCTIC 准备产物时，设置 `CONTRACK_IK_PARITY_ROOT=/path/to/check-directory` 启用全帧特征、关节解、时间轴和迭代次数对照；目录要求见 `tests/h2r/test_flashchord_ik.py`。单纯使用同一 solver 包不构成配方等价验证。
