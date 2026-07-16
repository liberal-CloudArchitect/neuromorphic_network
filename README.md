# Neuromorphic — 类脑神经网络研究工程

本仓库用于设计、训练和验证受神经科学启发的模块化神经网络。项目核心是可训练、可消融、可复现的网络系统；3D 脑模型仅是后续可选的 telemetry 展示层，不参与模型计算或网络 MVP 验收。

## 当前状态

- 工程阶段：`GATE-1` 已通过，P2 模块化计算闭环可开始。
- Python 包：`neuromorphic`，版本 `0.2.0`。
- 本地环境：Conda `brain`、Python 3.12、PyTorch 2.12.1。
- 本机加速：Apple Silicon MPS；实现必须保持 CPU/CUDA 兼容。
- P0 状态：科学假说、公共契约、三任务协议和 `telemetry-v1` 已冻结并通过 `GATE-0`。
- P1 状态：三任务、弱基线、GRU/Transformer、训练/恢复/统计和产物体系已通过 `GATE-1`。
- 模型状态：尚未实现类脑模块；P1 结果只是描述性单体基线，不证明类脑收益。

## 快速开始

```bash
make env
conda activate brain
make check
make smoke-mps
make smoke-p1
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
- 已包含：冻结的 `BrainPacket`/state/loss 契约、三个合成任务、训练基线和 `telemetry-v1`。
- 当前不包含：类脑模块实现、atlas、Three.js、Nilearn、WebSocket 或 3D viewer。
- `GATE-0` 与 `GATE-1` 只验收规格及基线基础设施，不构成生物等价性或类脑收益结论。
- 详细边界见 [科学与产品边界](docs/scientific-boundaries.md)。

## 目录

- `src/neuromorphic/`：Python 包骨架。
- `configs/`：P1 smoke 与正式基线配置。
- `docs/`：已接受的 P0 研究规格和治理文档。
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
make check         # 本地完整门禁
```

验收记录：[GATE-0](reports/gates/GATE-0.md)、[GATE-1](reports/gates/GATE-1.md)；正式基线汇总见 [Associative Recall GRU](reports/p1/associative_recall_gru.md)。

## 权利声明

本项目当前未开放授权，不允许在没有书面许可的情况下复制、修改、发布或分发，也不会发布到 PyPI。详见 [LICENSE](LICENSE)。
