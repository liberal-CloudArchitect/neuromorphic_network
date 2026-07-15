# Neuromorphic — 类脑神经网络研究工程

本仓库用于设计、训练和验证受神经科学启发的模块化神经网络。项目核心是可训练、可消融、可复现的网络系统；3D 脑模型仅是后续可选的 telemetry 展示层，不参与模型计算或网络 MVP 验收。

## 当前状态

- 工程阶段：仓库初始化与 P0 研究骨架。
- Python 包：`neuromorphic`，版本 `0.1.0`。
- 本地环境：Conda `brain`、Python 3.12、PyTorch 2.12.1。
- 本机加速：Apple Silicon MPS；实现必须保持 CPU/CUDA 兼容。
- 模型状态：尚未实现任何类脑模块，所有 P0 模板均为 `DRAFT`。

## 快速开始

```bash
make env
conda activate brain
make check
make smoke-mps
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
- 当前不包含：模块实现、训练任务、atlas、Three.js、Nilearn、WebSocket 或 3D viewer。
- `BrainPacket`、`ModuleOutput` 和 telemetry v1 尚未冻结；模板存在不等于 `GATE-0` 通过。
- 详细边界见 [科学与产品边界](docs/scientific-boundaries.md)。

## 目录

- `src/neuromorphic/`：Python 包骨架。
- `configs/`：草案配置；不得作为冻结实验协议。
- `docs/`：P0 研究模板和治理文档。
- `schemas/`：草案 schema。
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
make check         # 本地完整门禁
```

## 权利声明

本项目当前未开放授权，不允许在没有书面许可的情况下复制、修改、发布或分发，也不会发布到 PyPI。详见 [LICENSE](LICENSE)。
