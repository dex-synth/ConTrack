# ConTrack：H2R 与 original 对照

本文记录当前 H2R 适配相对公开 Sharpa 入口的差异，包括有效默认参数。运行方法见 [h2r.md](h2r.md)。表中“一致”仅指该行的配置或算法语义，不表示跨物理引擎数值等价。

| 对照对象 | 版本与入口 |
| --- | --- |
| original | [EmptyBlueBox/ConTrack `f5efe18`](https://github.com/EmptyBlueBox/ConTrack/tree/f5efe18e4b1f986566b252265c4bf63f3b67c6c6)，README 的 `scripts/rsl_rl/train.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0` |
| H2R | 基于本 fork `0c66474`，包含 5 mm 接触与 LeanRetarget 腕范围的参数适配；配方 `contrack-public-sharpa-h2r-v2`，有效参数另存于 prepared；`contrack_h2r.prepare` → `contrack_h2r.train` → `contrack_h2r.evaluate` |
| 配置口径 | original 使用 README 命令覆盖后的 Sharpa 配置；当前默认 update 数单独对齐论文附录 F 的 5000，其他论文配方对应关系不由此推定 |

## 1. 流水线与接口

| 项目 | original | H2R |
| --- | --- | --- |
| 输入 | HDF5：机器人 qpos、物体轨迹／网格、接触标注 | 一个 h2r_bench task 目录（demo、robot 与装配后的场景）；经 bench 的加载器 `h2r_io`（依赖包 `h2r-io`）校验身份和全部文件 |
| 运动学参考 | 读取 HDF5 中已生成的机器人 qpos；生成链未公开 | 从 MANO 生成参考，使用固定的 FlashCHORD／V2D Sharpa IK；具体参数见第 5 节 |
| 参考接触生成 | 读取 HDF5 的 15 段标注；生成参数未公开 | MANO 顶点按最近指骨分为 15 段，距物体表面 ≤ **5 mm** 的点参与局部接触点均值 |
| 15 段 → 10 槽 | 每指 proximal 单独一槽，middle＋distal 合并一槽；标志取 OR，点取有效段均值 | 一致 |
| 接触排列 | `object, hand, slot`；手指按 index、middle、ring、pinky、thumb | `hand, part, slot`；手指按 thumb、index、middle、ring、pinky；训练与部署共用映射 |
| 原生算法 | 仓库内 PyTorch PPO／CMDP、双 critic、adaptive reset library、阈值课程 | 一致 |
| 训练环境 | Isaac Lab `ManagerBasedRLEnv`＋PhysX | MJLab `ManagerBasedRlEnv`＋MJWarp；官方 `RslRlVecEnvWrapper` 连接原 runner |
| 部署 | `scripts/rsl_rl/play.py` 在原生环境中推理、推进物理 | 常驻 Unix socket 客户端；消费 bench 实际反馈，输出完整具名关节位置目标；客户端不创建 solver |
| 评估初态与终点 | play 从参考首帧及参考速度 reset，沿用原生环境终止逻辑 | 提交首帧机器人位置；bench 从参考第 0 帧、零速度开始，无 warm-up，并决定计分终点；训练 reset 仍按原生配方使用参考速度，这一训练／评估差异不做修正；方法训练 reset／提前终止不作用于评估世界 |
| checkpoint／续训 | 模型、normalizer、optimizer、CMDP、自适应库及课程 | 保留这些状态，增加 RNG、输入／布局／代码 hash 与同名 JSON；续训恢复学习状态并开始新 episodes，不恢复全部实时仿真缓存 |
| 日志／视频 | README 使用 W&B；保存时启动 play 录制视频 | TensorBoard＋配置／物理／耗时 JSON；训练入口不启动原生视频子进程 |

依据：[原生 Sharpa 环境](../source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/rby1a_sharpa_env_cfg.py)、[接触准备](../contrack_h2r/contacts.py)、[训练入口](../contrack_h2r/train.py)、[部署控制器](../contrack_h2r/controller.py)。

## 2. 机器人、动作与物理

| 项目／参数 | original | H2R |
| --- | --- | --- |
| 机器人与动作维度 | RB-Y1A＋Sharpa，每手 7 臂＋22 指关节；双手 58 维 | task 的 `sharpa_wave` 机器人包，每手 6 腕＋22 指关节；双手 56 维／单手 28 维；手的集合取自 task |
| 动作解码 | `q_target = q_ref + scale × clamp(action, -1, 1)`；actor 均值／推理输出经 `tanh` | 一致 |
| 臂／腕残差尺度 | 臂 `0.01 rad` | 腕平移 `0.06666666666666667 m`，腕旋转 `0.21333333333333335 rad`；每轴范围为正负该值，对齐 LeanRetarget Sharpa |
| 手指残差尺度 | `0.25 rad` | 一致 |
| 臂／腕跟踪与惩罚 | 7 维臂关节误差、速度和加速度 | 改为完整 6 维腕；平移／旋转分别除以 `1 m`／`1 rad` 的尺度，保留公式和权重 |
| PD 与执行器 | 臂 `kp=1000`、`kd=50`、力矩上限 `50 N·m`；手指按逐关节标定表 | 使用共同 MJCF 的位置执行器、PD、限位和力矩上限；不套用原生臂 PD，也不重新标定共同手指 |
| 物体与环境 | HDF5 网格生成 USD；从轨迹拟合铰链轴、锚点和限位；100×100 m ground plane | 完整 task MJCF：共同物体关节树、质量／惯量、碰撞体、有限桌面与支撑，不再从示范拟合物理资产 |
| 密度／摩擦／接触参数 | 物体密度 `200 kg/m³`；静／动摩擦均 `2.0`，combine=`multiply`；restitution=`0`；contact/rest offset=`0.01/0 m` | 从 task MJCF 读取；MuJoCo 接触参数不与 PhysX offset／摩擦合并规则作直接等价替换 |
| 碰撞过滤 | 原生 Sharpa 自碰撞排除表；HDF5 未使用侧关闭碰撞 | 使用共同 MJCF 碰撞规则与完整机器人能力 |
| 训练物理频率 | `120 Hz`，`dt=1/120 s` | 一致；由训练入口显式设置，不继承 task 的 300 Hz 名义步长 |
| 策略频率／decimation | `60 Hz`／`2` | 一致 |
| 频率可配置范围 | 选定配方固定上述频率 | 策略固定 `60 Hz`；物理频率可显式设为其整数倍，默认不变；任务时长须覆盖整数个控制区间 |
| 重力／求解器 | 重力 `(0,0,-9.81)`；PhysX，环境最小位置／速度迭代 `2/2`，机器人迭代 `16/0` | 重力、solver、integrator、iterations 等继承 task MJCF；只在训练副本中设置方法物理步长 |
| 批量环境布局 | 环境间距 `2 m` | 独立 MJWarp worlds，`env_spacing=0` |
| 接触／约束容量 | 每 sensor prim 接触容量 `32 × max(1, 物体数)`；PhysX rigid patch 容量 `2^19` | 每 world 默认 `contacts=512`、`constraints=4096`；可显式增大，溢出报错；与 PhysX 容量口径不同 |
| 实际接触反馈 | PhysX `ContactSensor` 的接触标志／点 | MuJoCo 有效 manifold；训练按物体局部坐标平均接触中点，固定 distal pads 归入 DP 槽；桌面不占物体接触槽 |
| 派生状态刷新 | PhysX／Isaac Lab 提供物理状态与传感器 | 方法通过轻量运动学获取积分后的 body 姿态／速度，MJLab 尾部保留一次完整 forward；接触保持最后物理 solve 的时刻，默认相差 8.33 ms |
| 评估物理频率 | play 沿用方法环境 | 评估设置（默认 300 Hz），只从 `hello.dt_s` 得知，可与训练不同；客户端声明 `control_dt_s = 1/60`，不能改变任务速度或计分时钟 |

依据：[原生机器人及手指标定表](../source/ConTrack/ConTrack/robots/rby1a_sharpa.py)、[H2R 布局](../contrack_h2r/layout.py)、[MJCF 边界](../contrack_h2r/mjwarp_scene.py)、[MJLab 环境](../contrack_h2r/mjlab_env.py)。共同资产参数以各次绑定的 task MJCF 和 `training_physics.json` 为准，没有一套方法私有固定值。

腕范围取自 LeanRetarget 的 `conf/physics_retarget/rl_tracking/robot/virtual_wrist_sharpa_wave_{bimanual,right}.yaml` 中 `control.joint_scales`。这里对齐每次动作相对参考的范围，ConTrack 策略频率仍为 60 Hz。

## 3. 参考、观测与训练参数

| 项目／参数 | original | H2R |
| --- | --- | --- |
| 控制网格采样 | 先重采样到 `60 Hz`；位置线性插值，四元数最短弧 nlerp，接触最近邻、半帧取后一帧 | 一致 |
| 参考速度 | 在控制网格上差分：首尾单边、内部中心差分；角速度在世界坐标系表达 | 一致 |
| reset 状态表示 | 机器人关节状态＋独立物体部件 pose／世界速度 | 转为共同广义 `qpos/qvel`；自由根角速度转局部坐标，铰链服从共同约束；部件速度与独立差分的差异写入准备报告 |
| 观测内容 | 当前机器人／物体位置和速度、实际接触、当前＋5 个未来控制帧 pose 参考、当前参考接触、phase | 一致 |
| 观测维度／顺序 | 原生臂／手关节及物体布局 | 按完整共同腕／手／part 具名布局转换；不能复用原生双臂 checkpoint |
| actor／critic 归一化 | 均启用，沿用原 normalizer | 一致 |
| Actor／双 Critic 隐层、激活 | `[256, 256, 128]`，`ELU` | 一致 |
| 初始动作噪声标准差 | `1.0` | 一致 |
| 并行环境数 | **6000**（README override；环境类裸默认 64） | 一致 |
| PPO updates／保存间隔 | **100000／200**（保存间隔为 README override；runner 类裸默认 1000） | **5000／200**；update 数采用论文固定预算，其他预算显式覆盖 |
| rollout／epochs／minibatches | `16 steps/env`／`10`／`16`；每 update `96000` transitions，每 minibatch `6000` 样本 | 一致 |
| optimizer／learning rate／schedule | Adam／`5e-4`／`adaptive` | 一致 |
| PPO clip／value loss | `clip_param=0.2`；`value_loss_coef=1.0`；启用 clipped value loss | 一致 |
| entropy／梯度裁剪／目标 KL | `0.001`／`max_grad_norm=1.0`／`desired_kl=0.01` | 一致 |
| gamma／GAE lambda | `0.99`／`0.95` | 一致 |
| CMDP | `alpha=0.9`，`lambda_lr=0.05`，`lambda_init=0`；warmup=`0`；`vg_window=100`，`vg_star_window=20` | 一致 |
| 训练随机化 | 物体 XY 扰动 `0`；PD 倍率 `[1,1]`；观测延迟 `[0,0]`，qpos／qvel 噪声 `0` | 一致 |
| 初始 episode-length buffer | `init_at_random_ep_len=True` | 一致 |
| seed 入口 | README 未显式传 `--seed`，由原 runner 配置提供 | CLI 默认 `--seed 0`，实际值写入 `run_config.json` |
| 预算／并行规模覆盖 | 可通过启动参数覆盖 | 相对 5000 updates／6000 envs 计算 `budget_override`，另存论文更新数来源 `budget_provenance` |

依据：[原生 PPO 配置](https://github.com/EmptyBlueBox/ConTrack/blob/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/agents/rsl_rl_ppo_cfg.py)、[共享配方](../rsl_rl_contrack/sharpa_recipe.py)、[参考处理](../rsl_rl_contrack/mimic.py)、[H2R 观测与解码](../contrack_h2r/reference.py)、[reset 坐标转换](../contrack_h2r/reset_reference.py)。

[论文附录 F](https://arxiv.org/html/2606.03177v3#Pt0.A6)规定 5000 updates/clip，并按 clip 调整并行数。当前 6000 环境对应总计 4.8 亿 control transitions；对齐 update 数不等于对齐论文所报 5.6–6.4 亿 simulator steps 或其余实验条件。

## 4. 奖励、reset 与终止

下表的“一致”指系数与计算规则；原 arm 项的输入已替换为第 2 节的六维腕。

| 项目／参数 | original | H2R |
| --- | --- | --- |
| 机器人跟踪 | 臂／手指权重 `10/10`；`exp(-4 × MSE)` | 一致 |
| 物体跟踪 | 位置／旋转权重 `500/500`；std=`0.05 m/0.5 rad` | 一致 |
| 接触奖励 | 接触匹配／接触点距离权重 `100/100`；距离核 std=`0.03 m` | 一致 |
| 速度惩罚 | 臂、指、物体线／角速度权重绝对值 `(1,1,1,0.1)`；sigma=`1` | 一致 |
| 加速度惩罚 | 同顺序权重绝对值 `(10,10,100,10)`；sigma=`50` | 一致 |
| CMDP 奖励分组 | `rg`=物体跟踪，`rs`=机器人／接触，`rp`=运动惩罚；各乘一次控制 dt，PPO 使用 `(rg+rp, rs+rp)` | 一致 |
| reset 抽样 | `topk=64`，回退 `delta=100` 帧，`epsilon=0.5`，softmax `tau=1.0`；保留随机抽样次序 | 一致 |
| 自适应库 | track EMA=`0.05`；best 衰减 `0.99999`；首帧不被历史状态替换 | 一致 |
| 终止阈值课程 | 位置 `0.1→0.05 m`，旋转 `1.0→0.5 rad`；成功 EMA=`0.05`，成功率区间 `[0.9,0.999]` | 一致 |
| 训练动作与时序 | 对 N 帧控制参考执行 `0…N-1`；动作后状态按动作帧计奖励／终止／历史，再推进非终止帧；timeout 与 pose failure 可同时成立 | 一致 |
| 声明的 episode 时长 | `raw_len/raw_fps`；末帧动作由参考 timeout 单独控制 | 一致 |
| 末帧失败的课程统计 | 到达末帧即计入 success EMA，即使该帧同时 pose failure | 一致 |
| 速度历史缓存 | 原生 indexed `copy_` 不写回，缓存实际只在 reset 时更新 | 一致 |
| 空 episode-return 窗口 | 原生 PPO 触发 `ZeroDivisionError` | 一致 |

当前默认没有加入 empty-return guard 或逐步更新速度缓存的算法修正。训练末帧动作只属于方法私有环境，统一评估在共同终点停止，不增加动作。

依据：[原生奖励](../source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/mdp/rewards.py)、[H2R 奖励](../contrack_h2r/rewards.py)、[reset library](../contrack_h2r/resets.py)、[原生 callback 多步对照](../tests/h2r/test_native_pipeline.py)。

## 5. H2R 新增的准备参数与依赖

ConTrack original 未公开 HDF5 的 IK／接触生成链，以下不能标为与 original 一致。H2R IK 来源为 V2D fork `a2b2099` 的固定源码，身份为 `flashchord-h2r-sharpa-ik-v1`，详见 [SOURCE.json](../contrack_h2r/_vendor/v2d/SOURCE.json)。

| 项目／参数 | original | H2R |
| --- | --- | --- |
| IK solver／迭代 | 未公开，无法核对 | Pink＋DAQP；最多 `100` 次迭代，内部 `frequency=200 Hz`（不是策略频率） |
| IK 收敛／尺度 | 未公开，无法核对 | 位置误差变化阈值 `1e-6`；`source_to_robot_scale=1.0`；`use_relative_frames=False` |
| IK 位置／朝向 cost | 未公开，无法核对 | 腕 `(0.2,0.2)`；拇指尖 `(1,0.05)`；食／中／无名指尖 `(1,0.1)`；小指尖 `(0.5,0.1)`；启用的中间指节目标 `(0.1,0)` |
| IK 初始化／后处理 | 未公开，无法核对 | 保留固定 V2D 首帧初始化、上一帧 warm start；自由腕解转换为连续六关节腕；不追加平滑或碰撞优化 |
| 参考接触阈值 | 未公开，无法核对 | 默认 `0.005 m`；15 段→10 槽合并规则见第 1 节 |
| 主环境 | Python `3.11`、PyTorch `2.7.0`、Isaac Sim `5.1.0`、Isaac Lab `42e61645` | Python `3.12`、PyTorch `2.9.0` CUDA `12.8`、MJLab `1.6.0`、MuJoCo／MJWarp `3.11.0`、Warp `1.16.0` |
| 准备环境 | 无公开独立配方 | MANO：Python `3.10`；IK：Python `3.11`、Pinocchio `3.7.0`、Pink `4.2.0`、DAQP `0.10.3`；与训练进程隔离 |

依据：[IK 配方](../contrack_h2r/retarget.py)、[IK 权重](../contrack_h2r/_vendor/v2d/robotic_grounding/retarget/params.py)、[主环境锁文件](../h2r/requirements.lock)、[IK 锁文件](../h2r/ik-requirements.lock)、[MANO 锁文件](../h2r/mano-requirements.lock)。

## 6. 源码改动与适用范围

| 代码范围 | 相对 original 的改动 |
| --- | --- |
| `rsl_rl_contrack/algorithms/ppo.py`、actor／normalizer／基础 runner | 一致 |
| 原生 `mimic_rby1a_sharpa/mdp/state.py`、`agents/rsl_rl_ppo_cfg.py` | 将采样、差分、reset 抽样和 PPO 常量抽到 `rsl_rl_contrack/mimic.py`、`sharpa_recipe.py`，供两条入口共用；默认 update 数改为论文的 5000，其余原生算法规则保留 |
| `contrack_h2r/`、`h2r/`、`scripts/h2r_*.sh` | 新增共同输入、MANO／IK、MJLab terms、反馈客户端、产物绑定及独立依赖环境 |
| `tests/h2r/` | 新增固定 original callback 对照、资产／协议检查、GPU 和训练／部署行为对照 |

此对照覆盖当前默认实现（配方 `contrack-public-sharpa-h2r-v2`，prepared `contrack-h2r-v3`，checkpoint `contrack-h2r-deployment-v4`）。接触阈值和动作尺度分别封存在 `binding.json` 的 `preparation.contact_distance_m` 与 `layout.action_scales`。此前格式的 prepared／checkpoint 被拒绝，须从 task 目录重新 prepare／train；原生双臂 checkpoint 也不能直接迁用。源码／callback 一致性及短训练检查不代表 Isaac 原生物理复现、跨引擎轨迹等价或论文性能复现；实际验证组合与结果以对应运行记录为准。
