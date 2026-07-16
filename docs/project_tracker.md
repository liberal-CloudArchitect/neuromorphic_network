---
title: 项目实施追踪台账
status: ACTIVE
last_updated: 2026-07-16
---

# 项目实施追踪台账

状态口径：`DONE` 表示已有可复核证据；`IN_PROGRESS` 表示实现已落盘但尚未通过阶段 Gate；`NOT_STARTED` 表示尚未开始。P2 项目在 `GATE-2` 独立裁决前一律不计为完成。

## 阶段状态

| 阶段 | 状态 | Gate | 说明 |
|---|---|---|---|
| P0 科学假说与计算规格 | DONE | GATE-0 PASSED | 人工计算抽象，不作脑区等价声明 |
| P1 任务基线与训练骨架 | DONE | GATE-1 PASSED | 三任务、单体基线、恢复与统计已冻结 |
| P2 模块化类脑网络 | IN_PROGRESS | GATE-2 PENDING | 等待完整 MPS suite、独立 verifier 与远程 CPU CI |
| P3 多种子收益与消融 | NOT_STARTED | GATE-3 PENDING | 不纳入 P2 结论 |

## P2 工作项

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| P2-01 | 冻结 P2 实施规格与科学边界 | IN_PROGRESS | `docs/p2_implementation_spec.md` |
| P2-02 | 六模块 registry 与公共状态 | IN_PROGRESS | `ModuleRegistry`、`NetworkState` |
| P2-03 | TaskControl 与三任务 boundary adapters | IN_PROGRESS | 104 维防泄漏 control |
| P2-04 | 感觉编码模块 | IN_PROGRESS | shared residual encoder |
| P2-05 | 情景记忆模块 | IN_PROGRESS | read-before-write、16 slots |
| P2-06 | 工作记忆模块 | IN_PROGRESS | 4 slots、自有更新门 |
| P2-07 | 预测适配模块 | IN_PROGRESS | actual-action dynamic target |
| P2-08 | 动作选择模块 | IN_PROGRESS | 隔离 task heads、动作 mask |
| P2-09 | 稀疏路由模块 | IN_PROGRESS | raw/executed top-2 与容量改派 |
| P2-10 | 组合网络与 SmallGraph live loop | IN_PROGRESS | slice/forward/scatter、状态化 rollout |
| P2-11 | 四阶段共享锚点预训练 | IN_PROGRESS | 精确阶段预算与冻结集合 |
| P2-12 | telemetry 配对联合训练 | IN_PROGRESS | CPU 微型分支参数/指标差 0 |
| P2-13 | modular checkpoint-v2 | IN_PROGRESS | 预验证、RNG、sampler、TBPTT |
| P2-14 | 成本、路由与状态监控 | IN_PROGRESS | MAC、延迟、梯度、state dynamics |
| P2-15 | CPU/MPS 测试与 CLI/CI | IN_PROGRESS | CPU 微型 suite 已通过 |
| P2-16 | 独立 GATE-2 裁决与版本 0.3.0 | NOT_STARTED | 仅在全部硬条件通过后执行 |

## P2 验收项

| ID | 验收项 | 状态 |
|---|---|---|
| AT-P2-01 | 六模块契约、状态、梯度与真实稀疏调用 | PENDING |
| AT-P2-02 | checkpoint-v2 中断恢复与 P1 回归 | PENDING |
| AT-P2-03 | telemetry 三层数值等价 | PENDING |
| AT-P2-04 | CPU 微型 CI 与完整 MPS suite | PENDING |
| AT-P2-05 | 报告、checksum、科学边界与独立 Gate | PENDING |

## 架构决策

| ID | 决策 | 状态 |
|---|---|---|
| ADR-004 | P2 采用六个版本化人工模块、step 级 top-2 稀疏执行及 checkpoint-v2；3D 仅是后续展示层 | PROPOSED（待 GATE-2 接受） |
