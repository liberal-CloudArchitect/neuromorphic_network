# Neuromorphic — 类脑神经网络研究工程

本仓库用于设计、训练和验证受神经科学启发的模块化神经网络。项目核心是可训练、可消融、可复现的网络系统；3D 脑模型仅是后续可选的 telemetry 展示层，不参与模型计算或网络 MVP 验收。

## 当前状态

- 工程阶段：`GATE-2` 已通过，项目进入 P3 多种子收益与消融准备阶段。
- Python 包：`neuromorphic`，版本 `0.3.0`。
- 本地环境：Conda `brain`、Python 3.12、PyTorch 2.12.1。
- 本机加速：Apple Silicon MPS；实现必须保持 CPU/CUDA 兼容。
- P0 状态：科学假说、公共契约、三任务协议和 `telemetry-v1` 已冻结并通过 `GATE-0`。
- P1 状态：三任务、弱基线、GRU/Transformer、训练/恢复/统计和产物体系已通过 `GATE-1`。
- P2 状态：六个人工计算模块、真实 top-2 稀疏执行、checkpoint-v2、telemetry 等价及 CPU/MPS 可移植性已通过 `GATE-2`。
- 模型状态：模块化工程闭环已实现；P2 结果不证明类脑收益或生物等价，正式多种子比较与消融属于 P3。

## 快速开始

```bash
make env
conda activate brain
make check
make smoke-mps
make smoke-p1
make smoke-p2-ci
```

若 `brain` 已存在，使用：

```bash
make env-update
```

不激活环境也可以运行门禁：

```bash
conda run -n brain make check
```

## 工程边界

- 网络主线：感觉编码、情景记忆、工作记忆、预测适配、动作选择和稀疏路由。
- 已包含：冻结契约、三个合成任务、P1 单体基线、六模块网络、真实稀疏路由、checkpoint-v2 和 `telemetry-v1`。
- 当前不包含：atlas、Three.js、Nilearn、WebSocket、3D viewer、SNN 或生成模型。
- `GATE-0`～`GATE-2` 只验收规格、基线和模块化工程闭环，不构成生物等价性或类脑收益结论。
- 详细边界见 [科学与产品边界](docs/scientific-boundaries.md)。

## 目录

- `src/neuromorphic/`：Python 包骨架。
- `configs/`：P1 基线与 P2 CPU/MPS suite 配置。
- `docs/`：已接受的 P0～P2 研究、架构和治理文档。
- `schemas/`：冻结的 telemetry 与运行产物 schema。
- `tests/`：单元、集成与端到端 smoke。
- `.omx/plans/`：可行性研究与实施追踪台账。
- `visualization/`：后续可选展示层；网络包不得依赖它。

## 开发命令

```bash
make lint          # Ruff 静态检查
make format-check  # Ruff 格式检查
make typecheck     # mypy
make test          # pytest
make smoke         # CPU/自动设备环境检查
make smoke-mps     # 强制 MPS 验收
make smoke-p1      # 三任务确定性训练 smoke
make smoke-p2-ci   # P2 CPU 微型资格测试
make smoke-p2-mps  # P2 完整 MPS Gate suite
make check         # 本地完整门禁
```

验收记录：[GATE-0](reports/gates/GATE-0.md)、[GATE-1](reports/gates/GATE-1.md)、[GATE-2](reports/gates/GATE-2.md)；P2 完整运行摘要见 [完整 MPS Suite](reports/p2/full_mps_suite.md)。

## 权利声明

本项目当前未开放授权，不允许在没有书面许可的情况下复制、修改、发布或分发，也不会发布到 PyPI。详见 [LICENSE](LICENSE)。
