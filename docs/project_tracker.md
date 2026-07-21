---
title: 项目实施追踪台账
status: ACTIVE
last_updated: 2026-07-21
---

# 项目实施追踪台账

状态口径：`DONE` 表示已有可复核证据；`IN_PROGRESS` 表示实现已落盘但尚未通过阶段 Gate；`NOT_STARTED` 表示尚未开始。P2 项目在 `GATE-2` 独立裁决前一律不计为完成。

## 阶段状态

| 阶段 | 状态 | Gate | 说明 |
|---|---|---|---|
| P0 科学假说与计算规格 | DONE | GATE-0 PASSED | 人工计算抽象，不作脑区等价声明 |
| P1 任务基线与训练骨架 | DONE | GATE-1 PASSED | 三任务、单体基线、恢复与统计已冻结 |
| P2 模块化类脑网络 | DONE | GATE-2 PASSED | 完整 MPS suite、独立 verifier 与远程 CPU CI 均通过 |
| P3 因果、泛化与网络 MVP | DONE | GATE-3 PASSED / GATE-NN-MVP FAILED | 科研矩阵完整；当前网络未取得 MVP 资格 |

## P2 工作项

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| P2-01 | 冻结 P2 实施规格与科学边界 | DONE | `docs/p2_implementation_spec.md` |
| P2-02 | 六模块 registry 与公共状态 | DONE | `ModuleRegistry`、`NetworkState` |
| P2-03 | TaskControl 与三任务 boundary adapters | DONE | 104 维防泄漏 control |
| P2-04 | 感觉编码模块 | DONE | shared residual encoder |
| P2-05 | 情景记忆模块 | DONE | read-before-write、16 slots |
| P2-06 | 工作记忆模块 | DONE | 4 slots、自有更新门 |
| P2-07 | 预测适配模块 | DONE | actual-action dynamic target |
| P2-08 | 动作选择模块 | DONE | 隔离 task heads、动作 mask |
| P2-09 | 稀疏路由模块 | DONE | raw/executed top-2 与容量改派 |
| P2-10 | 组合网络与 SmallGraph live loop | DONE | slice/forward/scatter、状态化 rollout |
| P2-11 | 四阶段共享锚点预训练 | DONE | 精确阶段预算与冻结集合 |
| P2-12 | telemetry 配对联合训练 | DONE | 完整分支参数最大差 `9.7312e-07`、指标差 0 |
| P2-13 | modular checkpoint-v2 | DONE | 预验证、RNG、sampler、TBPTT |
| P2-14 | 成本、路由与状态监控 | DONE | MAC、延迟、梯度、state dynamics |
| P2-15 | CPU/MPS 测试与 CLI/CI | DONE | 完整 MPS suite 与远程 CPU run `29494555468` 通过 |
| P2-16 | 独立 GATE-2 裁决与版本 0.3.0 | DONE | `reports/gates/GATE-2.md`、版本 `0.3.0` |

## P2 验收项

| ID | 验收项 | 状态 |
|---|---|---|
| AT-P2-01 | 六模块契约、状态、梯度与真实稀疏调用 | DONE |
| AT-P2-02 | checkpoint-v2 中断恢复与 P1 回归 | DONE |
| AT-P2-03 | telemetry 三层数值等价 | DONE |
| AT-P2-04 | CPU 微型 CI 与完整 MPS suite | DONE |
| AT-P2-05 | 报告、checksum、科学边界与独立 Gate | DONE |

## 架构决策

| ID | 决策 | 状态 |
|---|---|---|
| ADR-004 | P2 采用六个版本化人工模块、step 级 top-2 稀疏执行及 checkpoint-v2；3D 仅是后续展示层 | ACCEPTED（GATE-2） |
| ADR-005 | P3 拆分科研完整性 GATE-3 与科学收益 GATE-NN-MVP，并采用可恢复后台矩阵 | ACCEPTED（GATE-3） |

## P3 工作项

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| P3-01 | 冻结 protocol-v2、CR-002 与双 Gate | DONE | `docs/p3_implementation_spec.md`、`docs/change_requests/CR-002.md` |
| P3-02 | P3 数据、shared 双主基线和 Transformer-v2 | DONE | 单元/集成测试与 39-cell CPU 小矩阵 |
| P3-03 | 逐样本评估、AULC、严格配对统计 | DONE | strict pair fixture、Holm/AULC 测试 |
| P3-04 | checkpoint-v3、suite registry 与可恢复矩阵 | DONE | 81/81 cells、352 checksums、累计墙钟 49.16h |
| P3-05 | 因果干预、顺序学习、表征和成本分析 | DONE | 1,605,120 条逐样本记录与 10,000 次 paired bootstrap |
| P3-06 | network-mvp-v1 bundle 与推理接口 | DONE | Gate-controlled 接口已验证；科学 Gate 失败，未生成正式 bundle |
| P3-07 | CPU/MPS qualification、pilot 与后台管理脚本 | DONE | clean MPS 39/39、远程 CPU CI、12/12 pilot 与锁均通过 |
| P3-08 | 正式三 seed 后台矩阵 | DONE | `p3-full-dc6c259c-20260719T034230Z`，81/81 complete |
| P3-09 | GATE-3 独立评审 | DONE | `reports/gates/GATE-3.md`：PASSED |
| P3-10 | GATE-NN-MVP 独立评审 | DONE | `reports/gates/GATE-NN-MVP.md`：FAILED；0/2 收益类别，predictive 因果失败 |
