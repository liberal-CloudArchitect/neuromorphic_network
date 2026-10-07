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
| F 果蝇连接组探索支线 | IN_PROGRESS | 无（exploratory） | `CR-013` 提议（状态待批准）；F0 完成；F1 主体完成（27 轮，见 `docs/fly/F1-summary.md` 与 CR-013 的 F1 附录）：蘑菇体局部关联读出防遗忘、多峰类别下稀疏扩展、预测误差门控、突触标签式巩固获支持；中央复合体真实拓扑支持朝向记忆（FlyWire 与 hemibrain，后者需按类型统一递质符号）、可塑输入映射、路径积分（hemibrain），PEN→EPG 偏移为因果成分；未获支持：单峰类别下稀疏码优势、全局 surprise 调制、嗅小球采样偏好、MBON→DAN 回路实现 RPE 或无标签巩固；结果不计入任何 Gate |
| T 果蝇原理迁回主网络 | IN_PROGRESS | 无（exploratory） | `CR-014` 提议（状态待批准）；`neuromorphic.transfer` 子包，不改动 P1–P5；路线 T1 预测误差门控工作记忆（无提示规则切换）→ T2 局部关联读出防遗忘 → T3 环形吸引子状态模块 → T4 标签式巩固 → T5 稀疏扩展编码；T1 完成（`docs/transfer/T1-report.md`）：surprise 门控使工作记忆恢复准确率 +10.9pp（H1）、优于同信息输入 +3.9pp（H2）与打乱信号（H3），10/10 个 seed 学会（plain 5/10）；相对 shared GRU 仍差 2.9pp（H4 不支持）；T1b 完成（`docs/transfer/T1b-report.md`）：候选写入来自读出增强后的表示时，门控工作记忆恢复 +7.6pp（H1）并与 shared GRU 非劣（H5、H6，差距缩小 96–99%），但不优于 GRU（H4）；无门控时写入改进不提高可靠性（H2 不支持），门控仍有独立贡献（H3）：写入结构决定上限，门控决定可靠性；T1c：多情境下 GRU 更好（恢复 −3.6pp）；T2 完成：局部关联读出使类增量遗忘 0.99 → 0.05、与回放打平；T1d：按读出注意力寻址的门控没有可靠地修复多情境缺陷（+1.2pp，不显著），对单情境无害；CR-015（T-Gate）草案待审阅 |

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
| F1h | 动态 compartment 读出（真实 MBON→DAN 回路，seeds 120–129） | DONE | W1/W2 按字面支持、W3 非劣；事后对照显示优势来自标定区间不同，参数匹配时真实回路不优于重连回路，在慢区间更差（`docs/fly/F1h-report.md`） |
| F1i | 可塑 ER→EPG 视觉映射与 LNO 驱动转动（seeds 140–149） | DONE | I1 学习映射比解剖映射记忆高 26.8pp、I2 比 type_preserving 高 26.4pp，均支持；I3/I4 转动不支持；I5 按字面支持但转动量为噪声级（`docs/fly/F1i-report.md`） |
| F1j | 正反馈 MBON→DAN 回路的功能（共用参数网格，seeds 150–159） | DONE | J1 特异巩固被否证（−0.038，方向相反）；J2/J3 共模漂移支持（新异气味偏向趋近 +0.098，符号取反则反向）；J4 不依赖真实接线（重连漂移更强）；J5 略低于门槛（`docs/fly/F1j-report.md`） |
| F1k | 记忆特异的 MBON→DAN 回路：KC 侧突触标签与兴奋性谷氨酸（seeds 170–179） | DONE | K1 标签带来特异巩固 +0.023、K2 消除漂移 +0.068（均支持，24/24 网格点）；K3 不依赖真实接线；K4/K5 兴奋性谷氨酸消除漂移但不巩固；K7 反转代价 +0.018 低于门槛（`docs/fly/F1k-report.md`） |
| F1l | hemibrain PB 小球相位：结构与路径积分（seeds 160–169） | DONE | S1/S2 按字面未通过（左右 PB 交错半格；PEN 偏移 ±65–70°，一致性 0.998；谱坐标压缩约 5 倍）；P3 INVALID；P1、P2、P4 不支持：路径积分未复现，FlyWire 上的记忆优势未在 hemibrain 重复；发现搜索偏爱静默解（`docs/fly/F1l-report.md`）；**勘误**：hemibrain 递质标签错误（见 F1n/F1o），P1、P2、P4 不作为否定证据；路径积分见 F1p/F1q |
| F1m | 活动下限修正搜索后，FlyWire 与 hemibrain 的 CX 记忆优势（seeds 180–189） | DONE | M1 FlyWire 突触数 +0.409、M2 二值 +0.279 支持；M3 hemibrain 突触数 +0.025、M4 边数匹配二值 +0.049 不支持 → 数据集特异；下限没有改变任何结论，F1c-v4 不是退化解造成的（`docs/fly/F1m-report.md`）；**勘误**：M3、M4 来自 hemibrain 递质标签错误，统一符号后成立（F1o） |
| F1n | 输入组成互换：FlyWire/hemibrain 记忆差异的原因（seeds 190–199） | DONE | N1–N3 不支持：差异不来自类型级组成；事后诊断发现 hemibrain bump 全部钉在同一位置，原因是逐 body 递质预测把 38/46 个 EPG、一半 PEN 标成谷氨酸（`docs/fly/F1n-report.md`） |
| F1o | 按细胞类型统一递质符号后重测 hemibrain 记忆（seeds 200–209） | DONE | O1 +0.628、O2 +0.309、O3 +0.536 全部支持：hemibrain real 0.961，记忆优势跨数据集成立；F1l/F1m 的 hemibrain 结果已加勘误（`docs/fly/F1o-report.md`） |
| F1p | hemibrain 路径积分（统一递质符号，seeds 210–219） | DONE | Q1 LNO 驱动 J +0.622、Q3 比 type_preserving 高 +0.618、Q4 PEN 驱动 J +0.628 支持：路径积分首次复现；Q2、Q5 偏移消融未通过（消融取整不干净，消融后方向反转 9/10）（`docs/fly/F1p-report.md`） |
| F1q | PEN→EPG 映射的因果检验（seeds 220–229） | DONE | R1 方向与结构一致 +0.578、R3/R6 打乱 PEN→EPG 取消转动（+0.573、+0.490，记忆仍 0.91）、R4/R7 镜像使方向反转（+0.362、+0.496）均支持；R2/R5 去偏移未通过，因标签坐标与 EB 交错半格、实际留下反向偏移（`docs/fly/F1q-report.md`） |
| F1r | 路径积分的积分精度（F1p 解，seeds 210–219） | DONE | T1/T3 持续性均值 0.44/0.41、T2 LNO 分级 0.39 未达门槛；T4 PEN 分级 +0.943 支持；每种驱动 4/10 个 seed 为恒速、速度分级的积分器（R² 0.97–1.00，1–3.6 rad/s），其余为一次性位移（`docs/fly/F1r-report.md`） |
| F1s | 以积分为目标的搜索（seeds 230–239） | DONE | S2 LNO real − type_preserving +0.414、S3 real − pen_shuffled +0.336 支持；S1 real 均值 0.478 低于 0.5；PEN 驱动 S4/S5 不支持，S6 因对照门 INVALID；LNO 下真实接线 6/10 个 seed 找到积分器，零模型 0/10（`docs/fly/F1s-report.md`） |
| F1-SUM | F1 阶段总结与主线迁移建议 | DONE | `docs/fly/F1-summary.md` |

## T 线工作项（exploratory，CR-014）

| ID | 工作项 | 状态 | 当前证据 |
|---|---|---|---|
| T1 | 预测误差门控的工作记忆写入（无提示规则切换，seeds 501–510） | DONE | H1 +10.9pp、H2 +3.9pp、H3 +9.3pp 支持；H4（相对 GRU，OOD 总体）−2.9pp 不支持（`docs/transfer/T1-report.md`） |
| T1b | 写入结构 × surprise 门控 2×2 + GRU（seeds 601–610） | DONE | H1 +7.6pp、H3 +11.9pp 支持；H2 不支持；H5、H6 与 GRU 非劣（−0.3pp、−0.05pp）；H4 不支持；探针显示读出增强写入使记忆直接存储更新后的规则（`docs/transfer/T1b-report.md`） |
| T1c | 多情境无提示切换：槽结构 vs GRU（seeds 701–710） | DONE | GRU 更好：恢复 −3.6pp、OOD −2.2pp（H1、H2 不支持，H4、H5 非劣不成立）；读出增强写入仍必要（H3 +6.9pp）；槽结构对未切换情境的干扰更小（−0.4pp 对 −1.3pp），但被切换情境的更新更差（`docs/transfer/T1c-report.md`） |
| T2 | 局部关联读出 + 固定随机编码的类增量防遗忘（seeds 801–810） | DONE | 相对 `gru_sgd` 最终准确率 +67pp（H1）、遗忘 0.99 → 0.05（H2）；学习规则 +35pp、表示冻结 +55pp、稀疏扩展 +16pp（H3–H5）；优于 EWC（H7）；与回放打平（−0.2pp，H6 不支持，H8 非劣下界 −2.3pp 未通过）（`docs/transfer/T2-report.md`） |
| T1d | 按内容寻址的 surprise 门控（多情境主检验 + 单情境安全检查，seeds 901–910） | DONE | H1 不支持：寻址比全局门控高 1.2pp（CI 含 0）；多情境仍比 GRU 低 2.5pp（H3、H4 非劣不成立）；单情境无害（H5、H6 非劣）；按预设规则 CR-015 保留全局门控（`docs/transfer/T1d-report.md`） |
