---
title: 项目实施追踪台账
status: ACTIVE
last_updated: 2026-07-21
---

# 项目实施追踪台账

状态口径：`DONE` 表示已有可复核证据；`IN_PROGRESS` 表示实现已落盘但尚未通过对应 Gate；`NOT_STARTED` 表示尚未开始。qualification 通过只关闭工程资格项，不关闭正式科学 Gate。

## 阶段状态

| 阶段 | 状态 | Gate | 说明 |
|---|---|---|---|
| P0 科学假说与计算规格 | DONE | GATE-0 PASSED | 人工计算抽象，不作脑区等价声明 |
| P1 任务基线与训练骨架 | DONE | GATE-1 PASSED | 三任务、单体基线、恢复与统计已冻结 |
| P2 模块化类脑网络 | DONE | GATE-2 PASSED | 完整 MPS suite、独立 verifier 与远程 CPU CI 均通过 |
| P3 因果、泛化与网络 MVP | DONE | GATE-3 PASSED / GATE-NN-MVP FAILED | 科研矩阵完整；当前网络未取得 MVP 资格 |
| P4 预测闭环与语义稀疏路由 | IN_PROGRESS | GATE-4-QUAL 待 clean-SHA 裁决 | 工程实现与 CPU 资格路径完成；正式科学矩阵未运行 |

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
| ADR-006 | P4 将预测器移入跨步必经闭环，状态专家采用语义保留 top-1 路由，并隔离新协议结果 | ACCEPTED（protocol freeze） |

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

## P4 工作项

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| P4-01 | 冻结 protocol-v1、CR-003、ADR-006 与缺失指标 | DONE | `docs/p4_implementation_spec.md`、`docs/change_requests/CR-003.md`、`docs/decisions/ADR-006.md` |
| P4-02 | 建立 P4 task versions 与独立 split seed 空间 | DONE | namespace `p4`、split/hash 隔离测试 |
| P4-03 | 实现 `predictive_adapter.v2` 跨步预测、stop-gradient 与有界反馈 | DONE | 状态/时序/梯度/泄漏单测 |
| P4-04 | 实现语义保留 top-1 路由及 dense/legacy 控制 | DONE | reservation、tie-break、zero-drop、真实调用测试 |
| P4-05 | 实现 `modular-brain-v2`、telemetry-v2 与成本统计 | DONE | 三任务前后向、数值等价、schema 与 MAC 测试 |
| P4-06 | 实现 checkpoint-v4 与 v1～v3 回归兼容 | DONE | pending forecast、31→32、损坏拒绝与恢复测试 |
| P4-07 | 实现 chance/OOD/AULC/forgetting 与严格配对统计 | DONE | SmallGraph DP、非法输入和 bootstrap fixture |
| P4-08 | 实现 qualification/pilot/mechanism/full registry | DONE | 8/4/24/81 cell 配置与完整性测试 |
| P4-09 | 实现后台 start/status/logs/resume/stop/verify 和阶段锁 | DONE | 控制器单测、clean SHA/CI/MPS/电源/磁盘预检 |
| P4-10 | CPU/MPS qualification 与独立 GATE-4-QUAL | IN_PROGRESS | CPU 8/8 已通过；等待 clean SHA 的 MPS 与远程 CI |
| P4-11 | 冻结 pilot 选择 | NOT_STARTED | 必须先通过 GATE-4-QUAL |
| P4-12 | 三 seed 24-cell 机制矩阵与 GATE-4-MECH | NOT_STARTED | 不允许用 qualification 替代 |
| P4-13 | 81-cell 正式矩阵与 GATE-4 | NOT_STARTED | 仅机制 Gate 通过后启动 |
| P4-14 | GATE-NN-MVP-v2 与可选 network-mvp-v2 bundle | NOT_STARTED | 只有科学 Gate 通过才生成 bundle |
| P4-15 | 版本升级 0.5.0 | NOT_STARTED | 仅 GATE-4 PASSED 后执行 |
