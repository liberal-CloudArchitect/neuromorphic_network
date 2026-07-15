# 类脑神经网络项目实施方案与追踪台账

> 文档类型：网络优先的可执行实施计划 / 项目追踪主账
>
> 版本：0.3.0
>
> 基线调整日期：2026-07-15
>
> 建议启动日（T0）：2026-07-20，可在正式启动时整体平移
>
> 当前阶段：`GATE-0_PREPARATION`
>
> 实施状态：`ACTIVE`
>
> 科学与架构基线：[类脑神经网络设计与实现：可行性研究、计算架构与可视化附属方案](./brain-inspired-3d-model-feasibility.md)

---

## 0. 项目优先级声明

本项目的唯一核心主线是：

> **设计、训练并验证受神经科学启发的模块化神经网络。**

3D 脑模型是读取网络 telemetry 的可选展示层：

- 不参与模型 forward、loss、优化、路由或决策；
- 不得成为模块接口、训练流程或 `GATE-NN-MVP` 的依赖；
- viewer 关闭、断线或完全未安装时，网络必须仍可训练、评估、消融和复现；
- 3D 点亮不能替代任务指标、泛化、持续学习或模型内部因果证据；
- 展示层只有独立的 `GATE-VIZ`，失败不影响类脑网络 MVP 判定。

### 0.1 本台账回答的问题

1. 类脑网络的科学假说和计算模块是否明确？
2. 当前实现到哪个模块、任务和实验 Gate？
3. 模块化架构相对公平基线是否产生可重复收益？
4. 哪些消融、泛化和持续学习结果支持或否证设计？
5. 3D 展示是否忠实消费 telemetry，而没有反向绑架网络设计？

### 0.2 状态枚举

| 状态 | 含义 | 规则 |
|---|---|---|
| `NOT_STARTED` | 尚未开始 | 无实施产物 |
| `READY` | 依赖完成，可立即开始 | 必须已有负责人 |
| `IN_PROGRESS` | 正在实施 | 记录开始日和本周进展 |
| `IN_REVIEW` | 产物等待验收 | 必须有证据链接 |
| `BLOCKED` | 被明确问题阻塞 | 必须关联 `ISSUE-*` |
| `DONE` | 已通过验收 | 没有证据不得标记 |
| `DEFERRED` | 延期到后续阶段 | 写明恢复条件 |
| `CANCELLED` | 取消 | 关联 ADR 或 CR |

阶段状态：`NOT_STARTED / ACTIVE / GATE_REVIEW / PASSED / FAILED / ON_HOLD`。

### 0.3 更新纪律

- 每周更新项目快照、任务、指标、风险、问题、决策和变更。
- 每个 `DONE` 必须关联代码、测试、run manifest、统计报告或评审记录。
- 影响范围、日期、实验预算或验收阈值的变化必须建立 `CR-*`。
- 任何脑功能类比变化必须同步 evidence registry 和模块假说。
- `GATE-NN-MVP` 失败时保持失败，不允许以演示视频或 3D 效果替代。

---

## 1. 目标、范围与完成定义

### 1.1 类脑网络 MVP 目标

交付一个包含感觉编码、情景记忆、工作记忆、预测适配、动作选择和稀疏路由的可训练闭环，并通过容量匹配基线、多种子、消融、泛化和计算成本分析验证其功能分工。

### 1.2 `GATE-NN-MVP` 完成定义

- ≥5 个功能模块参与同一可训练闭环；
- 模块具有版本化 packet/state/loss/telemetry 接口；
- 三个任务与参数匹配、训练计算匹配、推理成本匹配的单体基线共享数据与评估协议；参数匹配为主比较，其余两类用于敏感性分析；
- 每个模块都有可测功能指标和单元测试；≥3 个关键模块的预注册任务级消融达到可重复阈值，编码器与选择器另有容量匹配替换/扰动对照；
- 至少在任务性能、OOD 泛化、样本效率、3 任务顺序学习或稀疏计算五项中的两项达到预注册的定量收益阈值；
- 路由不塌缩，模块状态不跨 episode 泄漏；
- 支持 checkpoint 恢复、fresh 环境复现和完整失败结果报告；
- telemetry 可关闭且关闭后不改变模型语义；
- **不要求** atlas、3D viewer 或脑区点亮。

### 1.3 核心范围内

- Evidence/Abstraction/Hypothesis registry；
- `BrainPacket`、`ModuleOutput`、模块状态和连接图；
- Associative Recall、Delayed Rule Switch、MiniGrid/小型图环境；
- 单体 RNN/Transformer、简单记忆、固定/随机路由基线；
- 感觉编码、情景记忆、工作区、预测器、选择器、稀疏路由；
- 分阶段训练、冻结/解冻、辅助损失和梯度冲突分析；
- 多种子、消融、OOD 泛化、3 任务顺序学习、样本效率与计算成本；
- 结构化 run manifest、checkpoint、telemetry 和离线分析。

### 1.4 可选范围

- atlas、ROI mesh、模块到脑区的多对多映射；
- WebSocket replay 与 Three.js 3D viewer；
- Activity、Learning、Model Causal Effect 三种展示；
- 展示性能、许可证和坐标 QA。

### 1.5 核心范围外

- 细胞/突触级人脑仿真；
- 全系统 SNN；
- 临床诊断、读心、意识或人格推断；
- 由 3D 脑区反向定义数百个网络模块；
- 把人工活动等同 BOLD 或真实放电；
- 在 `GATE-NN-MVP` 前扩展大规模多模态、完整世界模型或精细连接组。

---

## 2. 角色与治理

| 角色 ID | 角色 | 核心职责 | 建议投入 |
|---|---|---|---|
| `ROLE-PM` | 项目负责人 | 范围、Gate、风险、变更 | 0.3 FTE |
| `ROLE-ML` | 网络架构/训练 | 模块、损失、训练、基线、消融 | 1.0 FTE |
| `ROLE-EXP` | 实验与统计 | 任务、协议、多种子、泛化、报告 | 0.6 FTE |
| `ROLE-NEURO` | 神经科学顾问 | 证据、类比边界、研究支线 | 0.2 FTE |
| `ROLE-PLATFORM` | 平台/MLOps | CI、run manifest、checkpoint、telemetry | 0.5 FTE |
| `ROLE-QA` | 独立验证 | 测试、复现、Gate 审计 | 0.3 FTE |
| `ROLE-VIZ` | 可选可视化 | atlas、3D、回放和展示性能 | VIZ 启动后 1.0 FTE |

### 2.1 决策权

| 事项 | Accountable | Responsible | Consulted |
|---|---|---|---|
| 网络范围与 Gate | ROLE-PM | ROLE-PM | ROLE-ML, ROLE-QA |
| 模块接口与训练 | ROLE-ML | ROLE-ML | ROLE-EXP, ROLE-NEURO |
| 任务与统计协议 | ROLE-EXP | ROLE-EXP | ROLE-ML, ROLE-QA |
| 科学类比 | ROLE-NEURO | ROLE-NEURO | ROLE-ML |
| 复现与实验平台 | ROLE-PLATFORM | ROLE-PLATFORM | ROLE-ML, ROLE-QA |
| 网络 Gate 验收 | ROLE-QA | ROLE-QA | ROLE-PM, ROLE-ML, ROLE-EXP |
| 可视化 | ROLE-VIZ | ROLE-VIZ | ROLE-PLATFORM, ROLE-NEURO |

---

## 3. 时间基线与关键路径

建议 T0=2026-07-20；若正式启动日变化，整体平移。网络 MVP 预计 14 周。VIZ 可在 telemetry v1 稳定后分段并行：先实现 Activity/Learning，再在冻结消融结果可用后补 Model Causal Effect；其完成时间不得反向约束网络阶段。

| 阶段 | 计划日期 | 周 | 目标 | 状态 | Gate |
|---|---|---:|---|---|---|
| P0 科学与计算规格 | 2026-07-20—2026-07-31 | W1–W2 | 模块假说、接口、任务、基线协议 | `NOT_STARTED` | `GATE-0` |
| P1 任务基线与训练骨架 | 2026-08-03—2026-08-14 | W3–W4 | 三任务、单体基线、复现体系 | `NOT_STARTED` | `GATE-1` |
| P2 模块化计算闭环 | 2026-08-17—2026-09-11 | W5–W8 | ≥5 模块、路由、状态、损失 | `NOT_STARTED` | `GATE-2` |
| P3 因果、泛化与硬化 | 2026-09-14—2026-10-23 | W9–W14 | ≥3 seeds、消融、OOD、3 任务顺序学习、成本、复现 | `NOT_STARTED` | `GATE-NN-MVP` |
| VIZ 可选展示 | telemetry v1 后分段开展 | 非关键路径 | W9 起可做 Activity/Learning；P3-05 后补 Model Causal Effect | `DEFERRED` | `GATE-VIZ` |
| P4 完整持续学习与世界模型 | GATE-NN-MVP 后 4–8 周 | 后续 | replay、7 任务、生成、内部模拟 | `DEFERRED` | `GATE-4` |
| P5 生物合理性实验 | GATE-NN-MVP 后每项 6–12 周 | 后续 | SNN、局部学习、连接先验 | `DEFERRED` | 独立 Gate |

### 3.1 网络关键路径

```text
模块假说/接口/实验协议
  → 三任务与公平单体基线
  → 单模块单元测试
  → 模块化闭环与多损失训练
  → 1-seed smoke
  → ≥3 seeds 正式实验
  → 消融 + OOD 泛化 + 成本分析
  → fresh 环境复现
  → GATE-NN-MVP
```

`atlas → mesh → viewer` 不在关键路径上。

### 3.2 资源假设

- 2 名核心实施者：ROLE-ML 与 ROLE-EXP/PLATFORM；
- ROLE-NEURO、ROLE-QA 兼职；
- 至少 1 张单卡 GPU；
- 若只有 1 名全职实施者，日历时间预计扩大至 2.0–2.6 倍；
- VIZ 只有在不占用网络关键资源时并行启动。

---

## 4. 需求追踪矩阵

### 4.1 网络核心需求

| ID | 可验证需求 | 来源 | 实现任务 | 验收 | 状态 |
|---|---|---|---|---|---|
| `REQ-N01` | ≥5 模块参与闭环 | F§4, F§12 | P2-03—P2-11 | AT-P2-01 | `NOT_STARTED` |
| `REQ-N02` | 版本化 packet/state/loss 接口 | F§4 | P0-04, P2-01 | AT-P0-02 | `NOT_STARTED` |
| `REQ-N03` | 三个任务与公平单体基线 | F§7/P0–P1 | P0-06, P1-03—P1-08 | AT-P1-01 | `NOT_STARTED` |
| `REQ-N04` | 情景记忆快速绑定 | F§3, F§7/P3 | P2-04, P3-03 | AT-NN-02 | `NOT_STARTED` |
| `REQ-N05` | 工作区保持与规则控制 | F§3, F§7/P3 | P2-05, P3-04 | AT-NN-03 | `NOT_STARTED` |
| `REQ-N06` | 预测器改善 probe 或样本效率 | F§3, F§7/P3 | P2-06, P3-05 | AT-NN-04 | `NOT_STARTED` |
| `REQ-N07` | 路由稀疏且不塌缩 | F§3, F§7/P2 | P2-08, P3-06 | AT-P2-04 | `NOT_STARTED` |
| `REQ-N08` | 模块化至少两类定量收益达到预注册阈值 | F§7/P3 | P0-08, P3-02—P3-08 | AT-NN-05 | `NOT_STARTED` |
| `REQ-N09` | OOD 泛化、干扰鲁棒性和 3 任务顺序学习筛查 | F§7/P3, F§8 | P3-07 | AT-NN-06 | `NOT_STARTED` |
| `REQ-N10` | checkpoint 与 fresh 环境复现 | F§7/P1, F§12 | P1-02, P3-11 | AT-NN-07 | `NOT_STARTED` |
| `REQ-N11` | telemetry 可关闭且不改变语义 | F§5, F§12 | P2-12, P3-09 | AT-P2-05 | `NOT_STARTED` |
| `REQ-N12` | 失败、方差、成本和限制完整报告 | F§8, F§12 | P3-10—P3-12 | AT-NN-08 | `NOT_STARTED` |

### 4.2 可选展示需求

| ID | 可验证需求 | 实现任务 | 验收 | 状态 |
|---|---|---|---|---|
| `REQ-V01` | viewer 只消费 telemetry | VIZ-04—VIZ-07 | AT-VIZ-01 | `DEFERRED` |
| `REQ-V02` | viewer 断线不阻塞网络 | VIZ-05, VIZ-09 | AT-VIZ-02 | `DEFERRED` |
| `REQ-V03` | 三种展示语义分离 | VIZ-07—VIZ-08 | AT-VIZ-03 | `DEFERRED` |
| `REQ-V04` | atlas/映射/许可证可追溯 | VIZ-01—VIZ-03 | AT-VIZ-04 | `DEFERRED` |
| `REQ-V05` | 达到冻结性能阈值 | VIZ-09 | AT-VIZ-05 | `DEFERRED` |

---

## 5. 核心任务台账

### 5.1 治理与仓库初始化

| ID | 任务 | Owner | 估算 | 依赖 | 计划 | 状态 | 证据 |
|---|---|---|---:|---|---|---|---|
| `GOV-001` | 完成网络优先可行性研究与 ADR | ROLE-PM | 已完成 | 无 | 2026-07-15 | `DONE` | feasibility 文档 v0.2.1 |
| `GOV-002` | 重建网络优先实施追踪台账 | ROLE-PM | 已完成 | GOV-001 | 2026-07-15 | `DONE` | 本文件 v0.2.1 |
| `GOV-003` | 建立 README、许可证、贡献和科学边界说明 | ROLE-PLATFORM | 1.0 PD | 无 | 2026-07-15 | `DONE` | `README.md`, `LICENSE`, `CONTRIBUTING.md`, `docs/scientific-boundaries.md` |
| `GOV-004` | 锁定 Python/PyTorch、依赖和硬件环境 | ROLE-PLATFORM | 1.0 PD | GOV-003 | 2026-07-15 | `DONE` | `environment.yml`, `pyproject.toml`, `locks/`, `reports/environment/` |
| `GOV-005` | 建立 lint/typecheck/unit/integration/smoke CI | ROLE-PLATFORM | 2.0 PD | GOV-004 | W1–W2 | `IN_PROGRESS` | 本地 `make check`/MPS 已绿色；等待首次 GitHub Actions run |
| `GOV-006` | 建立实验产物、run ID 和证据存档规则 | ROLE-EXP | 0.5 PD | GOV-003 | 2026-07-15 | `DONE` | `docs/experiment_artifacts.md`, `artifacts/README.md` |

### 5.2 P0：科学假说与计算规格

| ID | 任务 | Owner | 估算 | 依赖 | 计划 | 状态 | 证据 |
|---|---|---|---:|---|---|---|---|
| `P0-01` | 建立 Evidence/Abstraction/Hypothesis registry | ROLE-NEURO | 2.0 PD | GOV-003 | W1 | `IN_PROGRESS` | DRAFT 模板已建；来源与评审待完成 |
| `P0-02` | 为 6 个 MVP 模块定义功能、状态、时间尺度与否证条件 | ROLE-ML, ROLE-NEURO | 2.0 PD | P0-01 | W1 | `IN_PROGRESS` | 六模块 DRAFT 已建；证据绑定待完成 |
| `P0-03` | 绘制模块连接图、信息流和动作副本时序 | ROLE-ML | 1.0 PD | P0-02 | W1 | `IN_PROGRESS` | DRAFT 计算图已建；接口/时序待评审 |
| `P0-04` | 冻结 BrainPacket/ModuleOutput/state/loss 契约 | ROLE-ML | 2.0 PD | P0-03, GOV-004 | W1 | `NOT_STARTED` | `src/neuromorphic/core/contracts.py` + tests |
| `P0-05` | 定义 module registry、生命周期和冻结/解冻规则 | ROLE-ML | 1.0 PD | P0-04 | W1–W2 | `NOT_STARTED` | `docs/module_lifecycle.md` |
| `P0-06` | 冻结三个任务、数据划分、预算和主指标 | ROLE-EXP | 2.0 PD | P0-02 | W1 | `NOT_STARTED` | `docs/experiment_protocols.md` |
| `P0-07` | 冻结单体/简单记忆/固定与随机路由基线；指定参数匹配主比较及训练计算、推理成本匹配敏感性比较 | ROLE-EXP, ROLE-ML | 2.0 PD | P0-06 | W2 | `NOT_STARTED` | `docs/baseline_spec.md` |
| `P0-08` | 冻结多种子、CI、效应量及五类收益的定量 Gate 阈值 | ROLE-EXP, ROLE-QA | 2.0 PD | P0-06 | W2 | `NOT_STARTED` | `docs/statistical_protocol.md` |
| `P0-09` | 定义 telemetry v1 模块语义；不含 atlas 字段依赖 | ROLE-PLATFORM, ROLE-ML | 1.0 PD | P0-04 | W2 | `NOT_STARTED` | `schemas/telemetry-v1.json` |
| `P0-10` | GATE-0 独立评审 | ROLE-QA | 0.5 PD | P0-01—P0-09 | W2 末 | `NOT_STARTED` | `reports/gates/GATE-0.md` |

### 5.3 P1：任务基线与训练骨架

| ID | 任务 | Owner | 估算 | 依赖 | 计划 | 状态 | 证据 |
|---|---|---|---:|---|---|---|---|
| `P1-01` | 实现 run manifest、seed、设备和数据版本记录 | ROLE-PLATFORM | 1.5 PD | GATE-0 | W3 | `NOT_STARTED` | run manifest schema/tests |
| `P1-02` | 实现 checkpoint 保存/恢复与 RNG 状态复现 | ROLE-PLATFORM, ROLE-ML | 2.0 PD | P1-01 | W3 | `NOT_STARTED` | resume equivalence test |
| `P1-03` | 实现 Associative Recall 与干扰变体 | ROLE-EXP | 1.5 PD | GATE-0 | W3 | `NOT_STARTED` | task tests + fixtures |
| `P1-04` | 实现 Delayed Rule Switch 与延迟外推变体 | ROLE-EXP | 1.5 PD | GATE-0 | W3 | `NOT_STARTED` | task tests + fixtures |
| `P1-05` | 实现 MiniGrid/小型图环境适配 | ROLE-EXP | 2.0 PD | GATE-0 | W3 | `NOT_STARTED` | deterministic environment smoke |
| `P1-06` | 实现随机、简单记忆和固定规则基线 | ROLE-ML | 2.0 PD | P1-03—P1-05 | W3 | `NOT_STARTED` | baseline configs/results |
| `P1-07` | 实现参数匹配主基线及训练计算、推理成本匹配的单体 RNN/Transformer 敏感性基线 | ROLE-ML | 3.5 PD | P1-03—P1-05 | W3–W4 | `NOT_STARTED` | baseline matching report |
| `P1-08` | 实现训练/验证/测试、早停、指标和 bootstrap CI | ROLE-EXP | 2.0 PD | P1-01, P1-06 | W4 | `NOT_STARTED` | evaluation tests |
| `P1-09` | 建立 NaN、梯度、mask、状态泄漏和序列边界检查 | ROLE-QA, ROLE-ML | 1.5 PD | P1-07, P1-08 | W4 | `NOT_STARTED` | failure-injection tests |
| `P1-10` | 跑通三个任务 1-seed smoke 与至少一个正式基线 | ROLE-EXP | 2.0 PD+计算 | P1-06—P1-09 | W4 | `NOT_STARTED` | run manifests + report |
| `P1-11` | GATE-1：基线与训练体系评审 | ROLE-QA | 0.5 PD | P1-01—P1-10 | W4 末 | `NOT_STARTED` | `reports/gates/GATE-1.md` |

### 5.4 P2：模块化类脑网络

| ID | 任务 | Owner | 估算 | 依赖 | 计划 | 状态 | 证据 |
|---|---|---|---:|---|---|---|---|
| `P2-01` | 实现 module registry、state reset 和组合接口 | ROLE-ML | 2.0 PD | GATE-1, P0-04 | W5 | `NOT_STARTED` | registry/state tests |
| `P2-02` | 实现统一训练 step、多损失记录和模块冻结 | ROLE-ML | 2.0 PD | P2-01 | W5 | `NOT_STARTED` | training integration tests |
| `P2-03` | 实现最小感觉编码器 | ROLE-ML | 1.5 PD | P2-01 | W5 | `NOT_STARTED` | shape/gradient tests |
| `P2-04` | 实现情景记忆：写入、模式分离、检索 | ROLE-ML | 3.0 PD | P2-03 | W5–W6 | `NOT_STARTED` | recall/interference tests |
| `P2-05` | 实现工作记忆：规则状态、门控、容量限制 | ROLE-ML | 3.0 PD | P2-03 | W5–W6 | `NOT_STARTED` | delayed-state tests |
| `P2-06` | 实现预测适配器与 `(s_t,a_t,s_(t+1))` 时序 | ROLE-ML | 3.0 PD | P2-03 | W6 | `NOT_STARTED` | transition/efference tests |
| `P2-07` | 实现动作选择器、候选冲突和策略输出 | ROLE-ML | 2.0 PD | P2-05, P2-06 | W6 | `NOT_STARTED` | policy/conflict tests |
| `P2-08` | 实现任务特定 `eligible_experts` 上的 top-k 路由、负载偏差监控和容量限制；必经模块不计入稀疏率 | ROLE-ML | 3.0 PD | P2-01 | W6 | `NOT_STARTED` | routing health tests |
| `P2-09` | 集成情景记忆 + 工作区任务闭环 | ROLE-ML | 2.0 PD | P2-04, P2-05 | W6 | `NOT_STARTED` | Associative/Rule integration |
| `P2-10` | 集成预测器 + 选择器 + 环境闭环 | ROLE-ML | 2.5 PD | P2-06, P2-07, P1-05 | W7 | `NOT_STARTED` | MiniGrid integration |
| `P2-11` | 集成稀疏路由和 ≥5 模块联合训练 | ROLE-ML | 3.0 PD | P2-08—P2-10 | W7 | `NOT_STARTED` | full-network tests |
| `P2-12` | 显式产出 GPU 归约 telemetry，consumer 可为空 | ROLE-PLATFORM, ROLE-ML | 2.0 PD | P0-09, P2-11 | W7 | `NOT_STARTED` | telemetry on/off equivalence |
| `P2-13` | 分阶段训练、冻结/解冻与辅助损失权重实验 | ROLE-EXP, ROLE-ML | 3.0 PD | P2-11 | W7–W8 | `NOT_STARTED` | training strategy report |
| `P2-14` | 梯度余弦、状态动力学和路由健康监控 | ROLE-EXP | 2.0 PD | P2-11 | W8 | `NOT_STARTED` | analysis report |
| `P2-15` | 完整网络 1-seed smoke、错误修复和 checkpoint 恢复 | ROLE-QA, ROLE-ML | 2.0 PD | P2-12—P2-14 | W8 | `NOT_STARTED` | smoke manifests; zero blockers |
| `P2-16` | GATE-2：模块闭环与正式实验就绪 | ROLE-QA | 0.5 PD | P2-01—P2-15 | W8 末 | `NOT_STARTED` | `reports/gates/GATE-2.md` |

### 5.5 P3：因果、泛化与网络 MVP

| ID | 任务 | Owner | 估算 | 依赖 | 计划 | 状态 | 证据 |
|---|---|---|---:|---|---|---|---|
| `P3-01` | 完整模型与公平基线各运行 ≥3 seeds | ROLE-EXP | 4.0 PD+计算 | GATE-2 | W9–W10 | `NOT_STARTED` | frozen run manifests |
| `P3-02` | 汇总性能、样本效率、参数、FLOPs、显存、wall-clock | ROLE-EXP | 1.5 PD | P3-01 | W10 | `NOT_STARTED` | efficiency report |
| `P3-03` | 情景记忆消融与干扰鲁棒性 | ROLE-EXP, ROLE-QA | 1.5 PD+计算 | P3-01 | W10 | `NOT_STARTED` | ≥15pp / CI 或失败报告 |
| `P3-04` | 工作区消融与延迟/规则外推 | ROLE-EXP, ROLE-QA | 1.5 PD+计算 | P3-01 | W10–W11 | `NOT_STARTED` | ≥10pp / CI 或失败报告 |
| `P3-05` | 预测器 probe 与闭环样本效率消融 | ROLE-EXP, ROLE-QA | 2.0 PD+计算 | P3-01 | W11 | `NOT_STARTED` | ≥10% / CI 或失败报告 |
| `P3-06` | 固定/随机/稀疏路由、直接 policy head 对照和 eligible-set 塌缩审计 | ROLE-EXP | 2.5 PD+计算 | P3-01 | W11 | `NOT_STARTED` | routing/selector report |
| `P3-07` | OOD 延迟/干扰/组合规则/环境转移，加固定预算 3 任务顺序学习筛查 | ROLE-EXP | 4.0 PD+计算 | P3-01 | W11–W12 | `NOT_STARTED` | generalization/forgetting report |
| `P3-08` | linear probe、CKA/RSA、动力学、跨 seed 分工及冻结随机/浅层编码器替换对照 | ROLE-EXP, ROLE-NEURO | 3.5 PD | P3-01 | W12 | `NOT_STARTED` | representation/control report |
| `P3-09` | telemetry on/off 推理、单训练步、整段训练等价性与独立开销测试 | ROLE-QA | 2.0 PD | P3-01 | W12 | `NOT_STARTED` | semantic/perf report |
| `P3-10` | 失败结果、敏感性和替代解释审查 | ROLE-QA, ROLE-NEURO | 1.5 PD | P3-02—P3-09 | W13 | `NOT_STARTED` | critical review |
| `P3-11` | fresh 环境复现全部关键结果 | ROLE-QA, ROLE-PLATFORM | 2.0 PD+计算 | P3-10 | W13 | `NOT_STARTED` | reproduction report |
| `P3-12` | GATE-NN-MVP 独立评审 | ROLE-QA, ROLE-PM | 1.0 PD | P3-01—P3-11 | W14 | `NOT_STARTED` | `reports/gates/GATE-NN-MVP.md` |

---

## 6. 可选 VIZ 任务台账

所有任务默认 `DEFERRED`；入口条件是 `P2-12 telemetry v1` 稳定且网络关键路径资源充足。

| ID | 任务 | Owner | 依赖 | 状态 | 证据 |
|---|---|---|---|---|---|
| `VIZ-01` | 冻结 atlas、空间、ROI、许可证 | ROLE-NEURO, ROLE-VIZ | P2-12 | `DEFERRED` | atlas ADR/manifest |
| `VIZ-02` | 方向/affine/分辨率和 ≥20 ROI 质心 QA | ROLE-VIZ | VIZ-01 | `DEFERRED` | coordinate QA |
| `VIZ-03` | ROI mesh、vertex_to_roi 和体积误差 QA | ROLE-VIZ | VIZ-02 | `DEFERRED` | GLB/assets report |
| `VIZ-04` | 冻结模块→ROI distribution 与 analogy_strength | ROLE-NEURO | VIZ-01, P2-12 | `DEFERRED` | mapping schema |
| `VIZ-05` | 非阻塞 replay/WebSocket 服务 | ROLE-PLATFORM | P2-12 | `DEFERRED` | disconnect/backpressure tests |
| `VIZ-06` | Three.js viewer、GPU palette、picking | ROLE-VIZ | VIZ-03, VIZ-05 | `DEFERRED` | viewer tests |
| `VIZ-07` | Activity/Learning 两视图、时间轴与回放 | ROLE-VIZ | VIZ-04—VIZ-06 | `DEFERRED` | UI/schema audit |
| `VIZ-08` | Model Causal Effect 视图、事件详情与免责声明 | ROLE-VIZ | VIZ-05—VIZ-07, P3-03—P3-05 | `DEFERRED` | causal-view e2e snapshots |
| `VIZ-09` | FPS、延迟、内存、断连与训练隔离测试 | ROLE-QA | VIZ-08 | `DEFERRED` | benchmark report |
| `VIZ-10` | 随机 ROI 映射与展示叙事对照 | ROLE-NEURO, ROLE-QA | VIZ-08 | `DEFERRED` | interpretation audit |
| `VIZ-11` | GATE-VIZ 独立评审 | ROLE-QA | VIZ-01—VIZ-10 | `DEFERRED` | `reports/gates/GATE-VIZ.md` |

---

## 7. 后续研究 Epic

| Epic | 内容 | 入口 | 退出证据 | 状态 |
|---|---|---|---|---|
| `P4-E01` | 生成解码器与 token/frame 追踪 | GATE-NN-MVP | 生成任务、成本、追踪报告 | `DEFERRED` |
| `P4-E02` | 世界模型和反事实 rollout | P4-E01 | 行为收益与额外 FLOPs 对照 | `DEFERRED` |
| `P4-E03` | replay 与连续学习 | P4-E02 | 7 任务平均遗忘相对降低 ≥20% | `DEFERRED` |
| `P4-E04` | 记忆来源、删除、过期和污染 | P4-E03 | 隐私/一致性测试 | `DEFERRED` |
| `P5-E01` | recurrent predictive coding | GATE-NN-MVP | 任务/拟合/泛化/成本报告 | `DEFERRED` |
| `P5-E02` | ANN/SNN 混合前端 | GATE-NN-MVP | 性能/延迟/能量代理对照 | `DEFERRED` |
| `P5-E03` | STDP/三因子学习与稳态规则 | GATE-NN-MVP | 稳定性和任务对照 | `DEFERRED` |
| `P5-E04` | 连接组软先验与神经数据相似性 | 数据许可完成 | 随机先验对照 | `DEFERRED` |

---

## 8. 阶段验收测试

### 8.1 GATE-0

| Test ID | 通过标准 | 状态 |
|---|---|---|
| `AT-P0-01` | 6 个模块均有证据、假说、输入/输出/状态/损失/消融/否证条件 | `NOT_RUN` |
| `AT-P0-02` | packet/state/loss 契约通过 valid/invalid/shape 测试 | `NOT_RUN` |
| `AT-P0-03` | 三任务协议包含划分、预算、seed、指标、基线和统计 | `NOT_RUN` |
| `AT-P0-04` | telemetry schema 不依赖 atlas、Web 或 viewer | `NOT_RUN` |
| `AT-P0-05` | 科学审查无一脑区一功能或生物等价性声称 | `NOT_RUN` |

### 8.2 GATE-1

| Test ID | 通过标准 | 状态 |
|---|---|---|
| `AT-P1-01` | 三任务共享同一数据/评估接口；参数匹配主基线、训练计算匹配和推理 FLOPs/延迟匹配敏感性基线均可运行 | `NOT_RUN` |
| `AT-P1-02` | 同 seed smoke 在容许误差内复现 | `NOT_RUN` |
| `AT-P1-03` | checkpoint 恢复后指标与连续运行一致 | `NOT_RUN` |
| `AT-P1-04` | 参数、FLOPs、训练步数和 wall-clock 均记录 | `NOT_RUN` |
| `AT-P1-05` | NaN/mask/状态泄漏故障注入均被测试捕获 | `NOT_RUN` |

### 8.3 GATE-2

| Test ID | 通过标准 | 状态 |
|---|---|---|
| `AT-P2-01` | ≥5 模块通过显式接口完成 forward/backward | `NOT_RUN` |
| `AT-P2-02` | `(s_t,a_t,s_(t+1))` 动作副本与误差时序正确 | `NOT_RUN` |
| `AT-P2-03` | 三任务 1-seed smoke 无 NaN、死锁、泄漏或不可控图增长 | `NOT_RUN` |
| `AT-P2-04` | 路由只统计任务特定 `eligible_experts`；必经模块排除；eligible set ≥4 时平均激活率 ≤60%，并相对预期任务分布审计负载偏差 | `NOT_RUN` |
| `AT-P2-05` | telemetry on/off：固定 checkpoint/input 推理输出 `allclose`；单训练步 loss/gradient/update `allclose`；同 seed 完整训练指标在预注册容差内；开销单独报告 | `NOT_RUN` |

### 8.4 GATE-NN-MVP

| Test ID | 通过标准 | 状态 |
|---|---|---|
| `AT-NN-01` | 完整模型和公平基线各 ≥3 seeds，95% CI 与效应量完整 | `NOT_RUN` |
| `AT-NN-02` | 记忆消融下降 ≥15 个百分点且 95% CI 不跨 0 | `NOT_RUN` |
| `AT-NN-03` | 工作区消融下降 ≥10 个百分点且 95% CI 不跨 0 | `NOT_RUN` |
| `AT-NN-04` | 预测器消融使 probe 或样本效率至少一项恶化 ≥10% | `NOT_RUN` |
| `AT-NN-05` | 相对参数匹配主基线，五类收益至少两项达标：任务分数 +≥5%；OOD 归一化分数 +≥5%；达标样本数 -≥15%；3 任务平均遗忘绝对 -≥5pp；活跃模块 FLOPs -≥20% 且分数非劣不超过 2%；bootstrap 95% CI 支持预期方向 | `NOT_RUN` |
| `AT-NN-06` | OOD 延迟/干扰/规则/环境转移及固定预算 3 任务顺序学习的遗忘、前向迁移和负迁移结果完整 | `NOT_RUN` |
| `AT-NN-07` | fresh 环境复现关键结果 | `NOT_RUN` |
| `AT-NN-08` | 失败、方差、成本、限制和替代解释完整报告 | `NOT_RUN` |

### 8.5 GATE-VIZ（可选）

| Test ID | 通过标准 | 状态 |
|---|---|---|
| `AT-VIZ-01` | viewer 只读取版本化 telemetry/冻结消融报告 | `DEFERRED` |
| `AT-VIZ-02` | viewer 关闭/断线不影响训练 | `DEFERRED` |
| `AT-VIZ-03` | 三视图数据源与语义分离 | `DEFERRED` |
| `AT-VIZ-04` | atlas/source/version/license/checksum/mapping 可追溯 | `DEFERRED` |
| `AT-VIZ-05` | 30 Hz、约 110–130 ROI 时 ≥55 FPS，p95 延迟 <150 ms | `DEFERRED` |

---

## 9. 项目快照

| 字段 | 当前值 |
|---|---|
| 快照日期 | 2026-07-15 |
| 总体状态 | `ACTIVE / GATE-0_PREPARATION` |
| 当前 Gate | GATE-0 准备中；尚未进入评审 |
| 已完成核心任务 | 5 / 55（规划 2，仓库治理 3） |
| 核心实施任务完成率 | 3 / 53 = 5.7%；模型实现仍为 0 |
| 可选 VIZ | 0 / 11，全部 `DEFERRED` |
| 开放风险 | 9 |
| 开放问题 | 0 |
| 下一 Gate | GATE-0，计划 2026-07-31 |
| 下一重点 | 证据来源评审、接口契约、任务与公平基线协议冻结 |

### 9.1 首批可执行任务

1. `GOV-003 → GOV-004 → GOV-005`；
2. `P0-01 → P0-02 → P0-03/P0-06`；
3. `P0-03 → P0-04 → P0-05/P0-09`；
4. `P0-06 → P0-07/P0-08`；
5. 全部证据完成后执行 `P0-10`。

---

## 10. 风险登记册

概率/影响 1–5；评分≥15 必须每周审查。

| ID | 风险 | 概率 | 影响 | 分数 | 触发器 | 缓解/应急 | Owner | 状态 |
|---|---|---:|---:|---:|---|---|---|---|
| `RISK-001` | 模块只有脑区名字，没有计算收益 | 3 | 5 | 15 | 消融 CI 跨 0 | 公平基线、预注册消融；Gate 失败并缩减模块 | ROLE-ML | `OPEN` |
| `RISK-002` | 任务存在捷径 | 3 | 5 | 15 | 简单基线接近上限 | 干扰、延迟外推、组合规则、shortcut audit | ROLE-EXP | `OPEN` |
| `RISK-003` | 多损失梯度冲突 | 4 | 4 | 16 | 某模块辅助损失下降但主任务恶化 | 分阶段训练、冻结/解冻、梯度余弦和权重扫描 | ROLE-ML | `OPEN` |
| `RISK-004` | 路由塌缩 | 3 | 4 | 12 | eligible experts 的使用分布持续偏离预注册任务先验，或长期退化为单一路径 | eligible-set 审计、负载约束、容量限制；回退固定路由热启动 | ROLE-ML | `OPEN` |
| `RISK-005` | 公平基线不足 | 3 | 5 | 15 | 参数、训练计算或推理成本仅匹配一项且未披露 | 参数匹配作主比较；训练计算和推理成本匹配分别做敏感性分析 | ROLE-EXP | `OPEN` |
| `RISK-006` | 灾难性遗忘/负迁移 | 4 | 4 | 16 | 3 任务顺序学习筛查出现显著遗忘或负迁移 | MVP 如实判定；P4 再比较 replay、正则和参数隔离 | ROLE-ML | `OPEN` |
| `RISK-007` | 复现失败 | 3 | 5 | 15 | checkpoint/seed 环境结果漂移 | run manifest、RNG 状态、锁定依赖、fresh review | ROLE-PLATFORM | `OPEN` |
| `RISK-008` | 展示层挤压网络主线 | 4 | 4 | 16 | GATE-2 前投入 atlas/viewer 优化 | VIZ 全部 DEFERRED；PM 停止非关键工作 | ROLE-PM | `OPEN` |
| `RISK-009` | 科学类比过度 | 2 | 5 | 10 | UI/文档出现生物等价性声称 | evidence registry、科学审查、禁止发布 | ROLE-NEURO | `OPEN` |

---

## 11. 问题、决策与范围变更

### 11.1 问题日志

| Issue ID | 日期 | 任务 | 描述 | Owner | 目标日 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| — | — | — | 当前无开放问题 | — | — | — | — |

阻塞关键路径 ≥2 个工作日需升级；预计 Gate 延迟 ≥3 日需建立 `CR-*`。

### 11.2 决策日志

| ADR | 日期 | 决策 | 状态 | 检查点 |
|---|---|---|---|---|
| `ADR-001` | 2026-07-15 | 网络设计、训练和验证为唯一主线；3D 为可选 observer | `ACCEPTED` | GATE-NN-MVP |
| `ADR-002` | 待 P0 | 模块契约与状态生命周期 | `PROPOSED` | GATE-0 |
| `ADR-003` | 待 P0 | 三任务和公平基线协议 | `PROPOSED` | GATE-0 |
| `ADR-004` | 待 P2 | 分阶段/联合训练与辅助损失策略 | `PROPOSED` | GATE-2 |
| `ADR-005` | 2026-07-15 | `brain` 使用 Python 3.12/PyTorch 2.12.1；本机 MPS、CI CPU；当前不开放授权 | `ACCEPTED` | GOV-004 |
| `ADR-V01` | 待 VIZ | atlas、ROI、许可证和展示映射 | `DEFERRED` | GATE-VIZ |

### 11.3 范围变更

| CR | 日期 | 变更 | 影响 | 决策 | 状态 |
|---|---|---|---|---|---|
| `CR-001` | 2026-07-15 | 将重点从“类脑网络 + 3D 联合 MVP”调整为“网络优先；3D 独立可选” | 网络周期调整为 14 周；VIZ 移出关键路径 | `APPROVED` | `CLOSED` |

---

## 12. 证据索引

| ID | 交付物 | 关联 | 路径 | 状态 |
|---|---|---|---|---|
| `EV-001` | 网络优先可行性研究 | GOV-001 | [可行性研究](brain-inspired-3d-model-feasibility.md) | `AVAILABLE` |
| `EV-002` | 网络优先实施台账 | GOV-002 | [实施台账](brain-inspired-3d-model-implementation-tracker.md) | `AVAILABLE` |
| `EV-G01` | 仓库治理与科学边界 | GOV-003 | [README](../../README.md)、[科学边界](../../docs/scientific-boundaries.md) | `AVAILABLE` |
| `EV-G02` | osx-arm64 环境清单与锁 | GOV-004 | [环境记录](../../reports/environment/bootstrap-notes.md) | `AVAILABLE` |
| `EV-G03` | 本地质量门禁 | GOV-005 | `make check`：9 passed/1 skipped；`make smoke-mps`：passed | `LOCAL_PASS` |
| `EV-G04` | 实验产物规则 | GOV-006 | [实验产物规范](../../docs/experiment_artifacts.md) | `AVAILABLE` |
| `EV-003` | GATE-0 | P0-10 | [GATE-0](../../reports/gates/GATE-0.md)（评审时记录 commit hash） | `PENDING` |
| `EV-004` | GATE-1 | P1-11 | [GATE-1](../../reports/gates/GATE-1.md)（评审时记录 commit hash） | `PENDING` |
| `EV-005` | GATE-2 | P2-16 | [GATE-2](../../reports/gates/GATE-2.md)（评审时记录 commit hash） | `PENDING` |
| `EV-006` | GATE-NN-MVP | P3-12 | [GATE-NN-MVP](../../reports/gates/GATE-NN-MVP.md)（评审时记录 commit hash） | `PENDING` |
| `EV-V01` | GATE-VIZ | VIZ-11 | [GATE-VIZ](../../reports/gates/GATE-VIZ.md)（评审时记录 commit hash） | `DEFERRED` |

---

## 13. 周进展日志

### 2026-W29：优先级重置

**状态**：`PLANNING_COMPLETE`

已完成：

- 将项目本体明确为类脑神经网络设计、训练和验证；
- 将 3D 移出网络关键路径和完成定义；
- 重建网络优先的需求、任务、Gate、风险和证据台账；
- 建立独立 `GATE-NN-MVP` 与可选 `GATE-VIZ`。

当时未开始：代码、任务、模型、实验、atlas 和 viewer 均尚未实施。

下一步：执行 GOV-003—GOV-005 与 P0-01—P0-09。

### 2026-W29：仓库初始化

**状态**：`ACTIVE / GATE-0_PREPARATION`

已完成：

- Git `main` 仓库、专有权利声明、README、贡献指南和科学边界；
- Conda `brain`：Python 3.12.13、PyTorch 2.12.1、Apple M4 MPS；
- `neuromorphic` 0.1.0 `src/` 包骨架、配置、环境锁与脱敏清单；
- P0 文档与 telemetry schema 草案，全部保持 `DRAFT`；
- 本地 Ruff、format、mypy、pytest、CPU/MPS smoke 全部通过；
- GitHub Actions 已配置，因尚无远程仓库，GOV-005 保持 `IN_PROGRESS`。

未开始：类脑模块、任务、训练、实验和可视化实现。

下一步：补齐 evidence registry 的论文级来源，冻结模块/状态/loss 契约与三任务协议。

### 周报模板

```markdown
### YYYY-Www

**总体状态**：ON_TRACK / AT_RISK / OFF_TRACK

已完成：
- TASK-ID：结果；证据。

进行中：
- TASK-ID：进度；预计完成日。

网络指标：
- task/baseline/seed：当前值 vs Gate；
- 参数/FLOPs/显存/吞吐：当前值；
- 路由/梯度/状态健康：当前值。

阻塞与风险：
- ISSUE/RISK-ID：处理计划。

下周：
- TASK-ID：可验收结果。

VIZ：DEFERRED / 进展（不得替代网络指标）。
```

---

## 14. Gate 评审模板

```markdown
# GATE-X Review

- 日期：
- commit / 环境：
- 评审人：
- 输入任务与冻结协议：

## 验收
| Test ID | 结果 | 证据 | 偏差 |
|---|---|---|---|

## 基线公平性
- 参数匹配主比较：
- 训练 token/环境步数与 wall-clock 匹配敏感性：
- 推理 FLOPs/延迟匹配敏感性：
- 无法同时匹配的维度及影响：

## 失败、风险与替代解释

-

## 结论
- [ ] PASSED
- [ ] FAILED
- [ ] CONDITIONAL（仅非关键整改）

## 签署
- ROLE-PM：
- ROLE-QA：
- ROLE-ML：
- ROLE-EXP：
- ROLE-NEURO：
```

---

## 15. Gate 完整性检查

- [ ] 所有核心任务状态已更新，`DONE` 均有证据；
- [ ] 所有 `BLOCKED` 关联 Issue；
- [ ] 数据、seed、预算、基线、环境和代码版本冻结；
- [ ] 参数、FLOPs、显存、wall-clock 和样本效率完整；
- [ ] 消融使用预注册阈值，失败结果未删除；
- [ ] OOD、干扰和替代解释已检查；
- [ ] checkpoint 与 fresh 环境复现通过；
- [ ] telemetry consumer 可关闭；
- [ ] VIZ 进展未被计入网络 Gate；
- [ ] 风险、ADR 和 CR 已更新；
- [ ] 下一阶段入口条件满足。

---

## 16. 文档变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1.0 | 2026-07-15 | 初始“网络 + 3D 联合 MVP”台账 |
| 0.2.0 | 2026-07-15 | 全面调整为网络优先；重排 14 周关键路径；3D 移至独立 VIZ 台账与 Gate |
| 0.2.1 | 2026-07-15 | 收紧公平基线、收益阈值、路由口径、持续学习筛查、遥测等价性和 VIZ 分段依赖 |
| 0.3.0 | 2026-07-15 | 初始化 Git、`brain` 环境、Python 包、P0 DRAFT、质量门禁与 CI；进入 GATE-0 准备阶段 |
