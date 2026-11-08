# 固定的 FlashCHORD / V2D IK

本目录仅包含 FlashCHORD H2R 准备实际使用的 IK、MANO 解码及必需依赖源码。来源、版本与逐文件 SHA-256 见 `SOURCE.json`。

- 完整源文件按字节复制，保留版权和许可证。`extracts` 条目按原文提取函数／常量，仅补齐独立模块的 imports；函数内容 hash 可核对。
- 空 `__init__.py` 只建立隔离命名空间，避免加载原仓库其他 runtime。
- Sharpa XML 仅提供原始 IK task-site 定义；运动学模型由共同 robot bundle 的手部子树生成，核验 FK 与限位。这里没有复制机器人 mesh 或 MANO 模型。
- `read_mano.py` 调用固定版本的外部 `manotorch`；MANO `.pkl` 由用户提供，并与 bundle 记录的 hash 核对。
- `contrack_h2r.mano_features` 与 `ik_worker` 是数据和模型边界适配，数值求解使用这里的原始代码。原始目标、权重、迭代、初始化和停止条件均保留。

运行时通过独立子进程的 `PYTHONPATH` 加载本目录。正常准备、训练和部署不需要相邻 V2D／FlashCHORD checkout。源码对照测试才使用原仓库。
