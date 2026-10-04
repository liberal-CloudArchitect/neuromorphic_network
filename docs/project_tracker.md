---
title: 项目实施追踪台账
status: ACTIVE
last_updated: 2026-10-04
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
| P4 预测闭环与语义稀疏路由 | DONE | GATE-4-MECH-CUDA FAILED | 24-cell CUDA mechanism 完整执行；预测因果与稀疏最坏 seed 非劣性被否证，禁止进入 full |
| P5 预测 surprise 与稳定语义路由 | IN_PROGRESS | qualification pending | v3 模块与 CPU qualification 已落盘，待 clean SHA MPS/CUDA/CI 闭环 |
| F 果蝇连接组探索支线 | IN_PROGRESS | 无（exploratory） | `CR-013` 提议；F0 完成；F1a 系列（v1–v4）完成（`docs/fly/F1a-report.md`）：局部关联规则防遗忘获支持；稀疏码仅在低噪声多峰类别下优于原型学习（确认实验通过）；真实接线无拓扑优势；F1b 预测误差门控可塑性获支持；F1c 环结构仅存在于真实接线但均匀参数下无吸引子；F1d 真实气味下采样偏好无判别收益；总结见 `docs/fly/F1-summary.md`；结果不计入任何 Gate |

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
| P4-10 | CPU/MPS/CUDA qualification 与独立工程 Gate | DONE | clean SHA `6fcc69a5` 的 CUDA qualification 8/8、checksum 有效且 CI 绿色；历史 MPS qualification 保留为敏感性证据 |
| P4-11 | 冻结 pilot 选择 | DONE | clean SHA `6fcc69a5` 的 CUDA pilot 4/4，按冻结规则选择 `preset-3` |
| P4-12 | 三 seed 24-cell 机制矩阵与 GATE-4-MECH | DONE | CUDA run 24/24、0 工程失败；`GATE-4-MECH-CUDA FAILED`，科学否证而非执行故障 |
| P4-13 | 81-cell 正式矩阵与 GATE-4 | NOT_STARTED | 仅机制 Gate 通过后启动 |
| P4-14 | GATE-NN-MVP-v2 与可选 network-mvp-v2 bundle | NOT_STARTED | 只有科学 Gate 通过才生成 bundle |
| P4-15 | 版本升级 0.5.0 | NOT_STARTED | 仅 GATE-4 PASSED 后执行 |
| P4-16 | 冻结 protocol-v2 的 mechanism 资源修订与阶段可观测性 | DONE | `CR-005`、48h 配置、evaluation/live-rollout heartbeat、cell 间 MPS cache 清理；不改变科学阈值 |
| P4-17 | 隔离 accelerator worker、持久化评估分区并建立 Windows/CUDA 工程验证路径 | DONE | `CR-006`；clean implementation SHA `d9a49b86` 的 MPS/CUDA qualification 均 8/8，GitHub Actions `32511375072` 绿色；详见 `reports/p4/execution-isolation.md` |
| P4-18 | 冻结并运行 Windows CUDA pilot 通道 | DONE | `CR-007`；同 SHA CI/qualification/pilot 闭环，选择 `preset-3` |
| P4-19 | 冻结并运行 Windows CUDA mechanism 通道 | DONE | `CR-008`；24/24 完成、checksum 有效、Gate 失败且未生成 mechanism lock |

## P5 工作项

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| P5-01 | 固化 CUDA mechanism 负结果并冻结新假说 | DONE | `GATE-4-MECH-CUDA`、`CR-009`、`p5_implementation_spec.md` |
| P5-02 | 实现 residual `predictive_adapter.v3`，移除直接感觉反馈 | DONE | persistence 初始等价、无感觉突变、时序/梯度单测 |
| P5-03 | 实现语义覆盖、有限 dual-route 与 straight-through `sparse_router.v3` | DONE | AR/DRS coverage、dual budget、zero-drop、scorer gradient 测试 |
| P5-04 | 实现 `ModularBrainNetworkV3` 与 P1–P4 兼容注册 | DONE | 三任务 forward/backward 与 v2 回归 |
| P5-05 | 建立 P5 task namespace 与独立 split seeds | DONE | P4/P5 内容哈希隔离测试 |
| P5-06 | CPU/MPS/CUDA qualification | IN_PROGRESS | dirty-SHA CPU qualification 已通过；待 clean SHA MPS/CUDA 与远程 CI |
| P5-07 | checkpoint-v6、telemetry-v3 与后台控制 | DONE | `CR-010`/`CR-012`；原子恢复、analysis curve、schema、on/off 等价、status/resume/verify 定向测试 |
| P5-08 | 冻结 validation-only pilot | DONE | 四候选、eligibility、AULC 排序与设备锁已冻结；正式 mechanism 仍未冻结 |
| P5-09 | P5 pilot、三 seed mechanism 与独立 Gate | IN_PROGRESS | pilot PASSED（preset-1）；`CR-011` 冻结 24-cell Gate，待 CPU/CUDA mechanism qualification 与正式运行 |
| P5-10 | 科研有效性修正与 protocol-v2 重建 | IN_PROGRESS | `CR-012`；fixed-budget analysis AULC、best checkpoint、可复算 summary、诊断损失诚实化与路由向量化已实现，待 clean SHA 全链重建 |

## F 支线工作项（exploratory，CR-013）

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| F0-01 | 开源果蝇数据集盘点：文件、大小、checksum、许可与访问方式 | DONE | `docs/fly/dataset_inventory.md`、`configs/fly/datasets.yaml` |
| F0-02 | 带 checksum 的显式获取与 signed-count 连接组容器 | DONE | `neuromorphic.fly`、`tests/unit/test_fly_*.py` |
| F0-03 | 用户批准并获取 T0-full 与 T1 数据集 | DONE | 14 个文件、1.16 GB，全部通过 size/checksum 校验 |
| F0-04 | 加载 FlyWire 并核对发布统计 | DONE | 官方 783 表聚合后与 Shiu v783 逐项一致；v630 为 127,400 个神经元 |
| F0-05 | PyTorch 全脑 LIF 与 Shiu 2024 定性复现 | DONE | 糖 GRN → MN9：逐神经元 rate r=0.9998；200 Hz 下 MN9 94.2 vs 93.3 Hz |
| F0-06 | CPU 确定性与 CPU/MPS 容差 | DONE | CPU 逐位一致；200 Hz 下 MPS 与 CPU 的 MN9 相差 0.3 Hz |
| F1a-01 | 冻结 MB 持续学习协议 | DONE | `docs/fly/F1a-mb-protocol.md` |
| F1a-02 | 右半球 PN→KC 回路提取与拓扑零模型 | DONE | 122 个 PN、2,423 个 KC、12,887 条边；保度重连与 ER 零模型 |
| F1a-03 | 10 seed 运行与预注册比较 | DONE | C1 支持；C2–C5 不支持或未确定；见 `docs/fly/F1a-report.md` |
| F1a-04 | 事后敏感性分析（描述性） | DONE | 稀疏 5% 在 9 种设置下均劣于 50% 与 NCM |
| F1a-05 | F1a-v2：KC 稳态归一化（预注册，seeds 10–19） | DONE | D1–D5 均不支持；码塌缩假说被否证 |
| F1a-06 | F1a-v3：多峰类别（预注册，seeds 20–29） | DONE | E3/E4 支持，E1/E2 不支持；事后发现噪声区依赖 |
| F1a-07 | F1a-v4：低噪声确认实验（预注册，seeds 30–39） | DONE | G1 +16.6pp、G2 +17.3pp、G3 +2.8pp 均支持；G4 不等价 |
| F1b | 多巴胺门控 surprise 学习（预注册，seeds 40–49） | DONE | H1 预测误差门控 +20.5pp 支持；H2 全局 surprise 不支持；H3 非劣；H4 不支持（`docs/fly/F1b-report.md`） |
| F1c | 中央复合体环形吸引子（预注册，seeds 50–59） | DONE | 结构：仅真实接线呈环（谱间隙）；动力学：均匀参数速率模型 J1–J3 全部不支持（`docs/fly/F1c-report.md`） |
| F1c-v2 | 训练细胞类型参数的环形吸引子（预注册，seeds 60–69） | INVALID | 三张图的训练都退化为全网静默，损失停在机会水平；不构成拓扑证据，需要新协议 |
| F1d | 真实气味统计（DoOR，预注册，seeds 70–79） | DONE | L1/L2 不支持（采样偏好无判别收益）；L3 real 略优于保度零模型 +0.7pp（`docs/fly/F1d-report.md`） |
| F1e | MBON→DAN 反馈回路结构检验（预注册） | DONE | O1 互惠反馈富集 7.3× 支持；O2 抑制偏好不显著（`docs/fly/F1e-report.md`） |
| F1c-v3 | 带资格检查的类型参数环形吸引子 | INVALID | 资格检查失败：阳性对照理想环在 BPTT 下也停在机会水平；正式 seeds 80–89 未使用。诊断：模型类可表达环形吸引子（网格搜索 100%），失败来自 BPTT 优化 |
| F1c-v4 | 无梯度等预算随机搜索类型参数（预注册，seeds 90–99） | DONE | P1–P3 全部支持：real 0.659，比 type_preserving 高 38pp、比 degree_preserving 高 24pp |
| F1f | 路径积分：PEN 单侧驱动（预注册，沿用 F1c-v4 参数） | DONE | R1–R3 不支持：只按记忆选出的参数不会自发产生左右相反的转动（`docs/fly/F1f-report.md`） |
| F1f-v2 | 记忆与转动联合等预算搜索（预注册，seeds 100–109） | DONE | 阳性对照通过（记忆 1.0、A 1.0）；real 找不到联合解，S1–S3 不支持；路径积分在本抽象中未复现 |
| F1g | 细化路径积分（LNO 角速度、ER 视觉、二值权重） | INVALID | 资格检查失败：理想环的亚阈值突触在二值阈值下被删除；结构检查已完成（左右 PEN 偏移方向相反，Delta7 抑制相反方向） |
| F1g-v2 | 细化路径积分（修正阳性对照，seeds 130–139） | DONE | 阳性对照通过；V1–V3 不支持：经真实 ER 去抑制提示，三张图都无法形成定位的 bump（记忆搜索最好约 0.33），A 不可解读。与“ER→EPG 映射靠可塑性习得”的生物学一致 |
| F1h | 动态 compartment 读出（真实 MBON→DAN 回路） | IN_PROGRESS | 协议已冻结（`d1d87c4`） |
| F1-SUM | F1 阶段总结与主线迁移建议 | DONE | `docs/fly/F1-summary.md` |
