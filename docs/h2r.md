# H2R bundle、直接 MJWarp 训练与统一评估

这是 ConTrack 的 H2R 任务适配：保留仓库内 PyTorch PPO／CMDP、双 critic、奖励分组、adaptive reset library 和阈值课程，将 Isaac Lab 环境迁至直接 `mujoco_warp`。初始运动学参考使用与 FlashCHORD 相同的 V2D IK 实现和参数，复制来源见 [SOURCE.json](../contrack_h2r/_vendor/v2d/SOURCE.json)。接触准备仍为 ConTrack H2R 配方；这不是未发布的官方 HDF5 生成链复现。

## 环境

在 ConTrack 根目录运行。环境与 benchmark 分开；无需 Isaac Sim、Isaac Lab、Newton 或其他方法的私有模块。

```bash
bash scripts/h2r_environment.sh /opt/h2r/contrack
bash scripts/h2r_ik_environments.sh /opt/h2r/contrack-ik
/opt/h2r/contrack/bin/python -m contrack_h2r.prepare --help
```

`h2r/requirements.lock` 固定 Python 3.12 环境的依赖，含 PyTorch 2.9.0 CUDA 12.8、MuJoCo／MJWarp 3.11.0、Warp 1.16.0。训练使用 CUDA；准备和在线推理可在 CPU 执行。Python 3.12 也满足现有 runner 的 f-string 语法。按实际驱动选择设备；只设置当前进程的 `CUDA_VISIBLE_DEVICES`，不要改变其他方法的环境。

原始 MANO 解码运行于独立 Python 3.10 环境，原始 IK 运行于 Python 3.11（Pinocchio 3.7.0、Pink 4.2.0、DAQP 0.10.3）。`mano-requirements.lock` 和 `ik-requirements.lock` 固定依赖；`manotorch` 和 `chumpy` 固定源码提交与压缩包 hash。它们通过文件与主准备进程交接，训练和在线推理不导入这些包。复制的源码位于本仓库内，不需要安装完整 V2D、FlashCHORD 或 Isaac Lab。

## 输入与准备

共同输入由 bench 预处理并封存。先在 bench 环境以相同 human bundle 和 robot bundle 创建开启 `--contacts` 的共同 task，然后运行：

```bash
/opt/h2r/contrack/bin/python -m contrack_h2r.prepare \
  --bundle /path/to/human/full --robot /path/to/robot \
  --task /path/to/common-task --output /path/to/new-prepared \
  --mano-root /path/to/mano \
  --loader-python /opt/h2r/contrack-ik/mano/bin/python \
  --ik-python /opt/h2r/contrack-ik/ik/bin/python
```

入口校验 bundle v1／v2／v3、task 身份、完整资源 hash、机器人和时钟。当前机器人布局限定为 `virtual_wrist_sharpa_wave_bimanual`／`virtual_wrist_sharpa_wave_right`；需要已解码的 MANO 21 joints（16 joints＋5 tips）、778 vertices、完整 MANO 参数及对应模型。未知机器人、未开启接触、缺失几何、不同参考、非有限数据和不一致关节树明确拒绝。ARCTIC 旧包未声明参与手时，以所绑定共同 task 的完整 robot hands 为兼容规则，并在 binding 中记录。

方法准备包括：

- 用原始 `MANO.forward()`／`AxisLayerFK` 生成世界坐标中的关节位置和朝向，保留 FlashCHORD 的指尖顶点定义；验证手部网格和非指尖关节与共同 bundle 一致。数据集适配仅转换参数表示、坐标与腕中心约定。
- 原样调用 `SharpaHandKinematics`：腕位置／朝向、五指尖位置／朝向及原始中间指节目标与权重全部保留。使用 DAQP、100 次最大迭代、200 Hz 内部迭代步长、1e-6 位置误差变化阈值、尺度 1.0，关闭 relative-frame tasks；保留原始首帧初始化和上一帧 warm start。参考时间轴不变，也不追加平滑、碰撞优化或另一套收敛条件。
- 从共同机器人提取手部子树，添加原始 task sites，在 Pinocchio 自由腕上求解，然后用 FlashCHORD 的连续腕角转换映射为共同六关节腕。随机非零构型的 FK 和限位校验贯穿 Pinocchio／MuJoCo 边界。
- 将皮肤顶点按最近 MANO 指骨分成 15 个 segment，对每个部件表面计算 8 mm 阈值内的平均局部点，再按原生 OR／有效点均值规则合并为 10 槽。固定 distal pad 归入 DP 槽，静态桌面不占 actor 的物体接触槽。
- 使用共同 root＋hinge 广义坐标验证全部 part 的 FK。完整有限桌面、支撑、质量惯量、碰撞体、执行器和被动关节直接来自 task MJCF，不从示范重新拟合。

`prepared/` 包含完整 task 副本、方法 `reference.npz`、`binding.json`、`prepare_report.json` 和 `training_physics.json`。`ik/` 封存原始 MANO 特征、IK 模型、逐帧求解结果、来源、配方与环境版本；`ik/penetration.json` 记录整段碰撞诊断，包含物体之间的接触。初始化诊断失败不会移动物体或修改 IK，eval 仍独立执行初始化检查。准备输出不覆盖输入或已有目录，失败时清理本次未完成输出。

新默认配方为 `flashchord-h2r-sharpa-ik-v1`，替换此前 `contrack-h2r-geometric-ik-v1`。旧 prepared／checkpoint 保持原绑定；切换 IK 后须生成新的 prepared 并重新训练，不能将旧 checkpoint 改绑到新参考。原 `--ik-iterations` 参数已移除，原始求解参数固定在配方中。

## 训练与续训

```bash
CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.train \
  --prepared /path/to/prepared --output /path/to/new-training \
  --num-envs 64 --iterations 100000 --seed 0

CUDA_VISIBLE_DEVICES=0 /opt/h2r/contrack/bin/python -m contrack_h2r.train \
  --prepared /path/to/prepared --output /path/to/new-resumed-training \
  --num-envs 64 --iterations 1000 --seed 0 --resume /path/to/training/model_999.pt
```

训练使用 task 的物理和控制周期（默认 240／30 Hz）。共同 Sharpa 每手 6 腕＋22 手指，原生 58 维双臂 checkpoint 不能复用。策略仍观察当前状态、实际接触、当前＋5 个未来控制时刻参考、参考接触和 phase。位置线性插值，四元数最短弧 nlerp，接触最近邻；最后不足一个策略区间时保留 task 的准确终点。

腕平移／旋转／手指残差尺度分别为 0.01 m、0.01 rad、0.25 rad。原 arm 跟踪与惩罚项改用完整六维腕，平移／旋转的归一化尺度显式保存在 layout，默认分别为 1 m／1 rad。奖励延续原 manager 的 action-time 参考：动作施加后的状态与该动作对应的参考比较，再推进下一观测时间。终止时不多执行一个 endpoint 动作。实际接触点使用最后求解时的物体局部接触中点均值；这是从 PhysX sensor 到 MuJoCo manifold 的接触表示适配。

`rg/rs/rp` 各乘一次实际控制区间，PPO 使用 `(rg+rp, rs+rp)`。训练初态可以随机起帧并恢复参考速度；部署只提交首帧机器人位置，bench 设零速度。原生训练随机化默认关闭，当前 H2R 配方也关闭；共同 `sim-dr` 扰动由 bench 执行。

原 PPO 只作一处边界修复：rollout 更新前尚无 episode 结束时，保持初始 CMDP 权重，不对空 return 窗口求均值；有真实 episode return 后继续原更新公式。

GPU 参数回读报告保存在训练目录。默认每 world 接触容量 512、约束容量 4096，可通过 `--contacts-per-world`／`--constraints-per-world` 显式增大，溢出直接报错。状态／奖励主要在 GPU；当前实现优先保证可核对性，没有 CUDA graph 或吞吐优化支持声明。

每个 `model_N.pt` 对应一个 `model_N.json`，包含 checkpoint hash、输入身份、观测／动作布局、normalizer 配方、训练配置和源码 hash；normalizer 数值随模型保存。checkpoint 同时保留 optimizer、CMDP、自适应库、课程与 RNG。续训恢复学习状态并开始新的训练 episodes，不承诺恢复同一时刻的所有仿真缓存。仅加载自己生成或信任的 PyTorch checkpoint。

## 独立统一评估

在 bench 的独立环境启动 `evaluate-server`，使用准备时绑定的 task 和共同 `sim`／`sim-dr` 配置。然后运行常驻客户端：

```bash
/opt/h2r/contrack/bin/python -m contrack_h2r.evaluate \
  --prepared /path/to/prepared --checkpoint /path/to/training/model_999.pt \
  --socket /path/to/shared/eval.sock --device cpu
```

客户端校验 task、robot、时钟、观测 schema 和 checkpoint，只使用 bench 的实际关节、body 原点世界速度及 contact records。`contacts=null` 是错误，有效空列表是无接触。actor 的 normalizer 和 `tanh` 由原 `ActorCritic.act_inference()` 执行；共享解码输出所有具名关节位置目标。部署不启动物理 solver，不提交物体状态、reset、外力或方法终止条件。

## 代码与验证

| 模块 | 职责 |
| --- | --- |
| `inputs.py`、`layout.py` | 输入身份、封存资源、具名布局 |
| `prepare.py`、`retarget.py`、`contacts.py` | 准备编排、隔离 IK 调用、穿透诊断、ConTrack 接触参考 |
| `mano_features.py`、`ik_worker.py`、`_vendor/v2d/` | 原始 MANO／IK 及输入、模型、具名关节边界转换 |
| `mjwarp_scene.py`、`mjwarp_env.py` | 共同 MJCF、GPU 批量物理、VecEnv、真实接触 |
| `rewards.py`、`resets.py` | 原生奖励公式、自适应库和课程 |
| `reference.py` | 训练／推理共享的时间、特征和动作解码 |
| `train.py` | 原 runner 的新入口与 checkpoint 上下文 |
| `controller.py`、`protocol.py`、`evaluate.py` | 无 simulator 的反馈控制与 Unix socket |

```bash
/opt/h2r/contrack/bin/python -m pytest tests/h2r -q
CONTRACK_TEST_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  /opt/h2r/contrack/bin/python -m pytest tests/h2r -q
```

测试包含原生奖励 callback 数值对照、参考终点与四元数符号、输入篡改、协议分包、真实接触、body 原点速度、mask reset、PPO／CMDP 续训和完整控制输出一致性。GPU 测试需显式启用。短训练与闭环检查验证接入正确性，不证明论文性能或任务成功；实际任务结果与已验证组合保留在对应运行目录。

IK 测试还检查固定源码、世界坐标转换和只读穿透诊断。提供独立原 FlashCHORD 与 ConTrack 完整 ARCTIC 准备产物时，设置 `CONTRACK_IK_PARITY_ROOT=/path/to/check-directory` 启用全帧特征、关节解、时间轴和迭代次数对照；目录要求见 `tests/h2r/test_flashchord_ik.py`。单纯使用同一 solver 包不构成配方等价验证。
