# Neuromorphic — 类脑神经网络研究工程

本仓库用于设计、训练和验证受神经科学启发的模块化神经网络。项目核心是可训练、可消融、可复现的网络系统；3D 脑模型仅是后续可选的 telemetry 展示层，不参与模型计算或网络 MVP 验收。

## 当前状态

- 工程阶段：`GATE-2` 已通过，项目进入 P3 多种子收益与消融准备阶段。
- Python 包：`neuromorphic`，版本 `0.4.0`。
- 本地环境：Conda `brain`、Python 3.12、PyTorch 2.12.1。
- 本机加速：Apple Silicon MPS；实现必须保持 CPU/CUDA 兼容。
- P0 状态：科学假说、公共契约、三任务协议和 `telemetry-v1` 已冻结并通过 `GATE-0`。
- P1 状态：三任务、弱基线、GRU/Transformer、训练/恢复/统计和产物体系已通过 `GATE-1`。
- P2 状态：六个人工计算模块、真实 top-2 稀疏执行、checkpoint-v2、telemetry 等价及 CPU/MPS 可移植性已通过 `GATE-2`。
- P3 状态：81-cell、三 seed 正式矩阵与独立统计已完成；`GATE-3 PASSED`，`GATE-NN-MVP FAILED`。
- 模型状态：情景记忆与工作记忆获得任务内因果支持，但预测适配、双基线总体收益和稀疏非劣性未满足冻结条件；当前网络不是 qualified MVP。

## 快速开始

```bash
make env
conda activate brain
make check
make smoke-mps
make smoke-p1
make smoke-p2-ci
make smoke-p3-ci
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
make smoke-p3-ci   # P3 全 cell 类型 CPU 小样本资格矩阵
make qualify-p3-mps # P3 全 cell 类型 MPS 小样本资格矩阵
make check         # 本地完整门禁
```

P3 qualification、pilot 和正式实验由后台脚本管理：

```bash
./scripts/p3_full_run.sh freeze-qualification artifacts/runs/<qualification-run-id>
./scripts/p3_full_run.sh record-ci
./scripts/p3_full_run.sh start   # 首次：12-cell、每 cell 1,000 updates 的 pilot
./scripts/p3_full_run.sh status
./scripts/p3_full_run.sh logs
./scripts/p3_full_run.sh resume
./scripts/p3_full_run.sh stop
./scripts/p3_full_run.sh verify
./scripts/p3_full_run.sh freeze-pilot artifacts/runs/<pilot-run-id>
./scripts/p3_full_run.sh start   # 再次：使用冻结 preset 的三 seed 正式矩阵
```

`start` 会验证 clean SHA、`HEAD == origin/main`、qualification/CI lock、MPS、电源和磁盘。首次启动 pilot；pilot 完成并冻结选择后，再次启动才进入正式矩阵。后台运行只写入 ignored artifacts；关闭终端不会停止进程，机器重启后使用 `resume`。

验收记录：[GATE-0](reports/gates/GATE-0.md)、[GATE-1](reports/gates/GATE-1.md)、[GATE-2](reports/gates/GATE-2.md)、[GATE-3](reports/gates/GATE-3.md) 与 [GATE-NN-MVP](reports/gates/GATE-NN-MVP.md)。P3 完整结果见 [正式统计](reports/p3/formal.md)。

## 权利声明

本项目当前未开放授权，不允许在没有书面许可的情况下复制、修改、发布或分发，也不会发布到 PyPI。详见 [LICENSE](LICENSE)。
