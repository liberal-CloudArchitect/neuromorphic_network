---
title: GATE-2 独立验证报告
status: PASSED
gate: GATE-2
verified_commit: 9499a33645e0b446e2039b790387b240f5758bda
artifact_run: p2-suite-s7-20260716T112848Z
verified_at: 2026-07-16
---

# GATE-2 独立验证报告

## 裁决

**GATE-2：PASSED。**

本地代码、完整 MPS suite、产物完整性、路由、成本、梯度覆盖、telemetry 等价性及科学边界均通过本次独立复核。提交 `9499a33645e0b446e2039b790387b240f5758bda` 对应的远程 GitHub Actions CPU CI 也已通过公开 GitHub API 独立核验为绿色，所有冻结硬条件均已闭合。

可以关闭 P2-01～P2-16 与 AT-P2-01～05、接受 ADR-004，并将项目升级为 `neuromorphic 0.3.0`。版本、台账和报告更新仍应作为独立 Lore Commit 提交并再次接受正常 CI，不得把本报告误解为 P3 科学收益结论。

## 验证对象

- Git：`main`，`HEAD == origin/main == 9499a33645e0b446e2039b790387b240f5758bda`；完整运行 manifest 记录 `dirty: false`。
- 完整运行：`artifacts/runs/p2-suite-s7-20260716T112848Z`。
- 环境：Python `3.12.13`、PyTorch `2.12.1`、Apple MPS。
- 预算：四个预训练阶段各 100 updates；telemetry off/on 各 600 updates，每任务各 200 updates；batch 64。

## 本地硬条件证据

### 1. Manifest、状态与校验和

- `manifest.json` 通过 `schemas/run-manifest-v1.json` Draft 2020-12 校验，`status=completed`、`failure=null`。
- manifest 的提交 SHA 与当前 `HEAD` 一致，且运行时工作树为 clean。
- manifest 登记的 15 个产物逐一重算 SHA-256，`15/15` 匹配；配置、训练日志、阶段 checkpoint、共享 checkpoint、两个分支 checkpoint、telemetry 和 summary 均在登记范围内。
- 本次 verifier 开始检查时 Git 工作树干净。其后出现的 P2 摘要与台账修改属于 Gate 汇总工作，不改变被验证的 clean SHA 或忽略目录中的原始运行产物。

### 2. 训练预算、有限性与下降

所有 loss 均为有限数，且每组末 10 次均值低于首 10 次均值：

| 训练段 | updates | 首 10 均值 | 末 10 均值 | 结果 |
|---|---:|---:|---:|---|
| sensory 预训练 | 100 | 1.840542 | 0.803510 | PASSED |
| episodic 预训练 | 100 | 1.445165 | 1.266271 | PASSED |
| working 预训练 | 100 | 0.578837 | 0.577190 | PASSED |
| predictive 预训练 | 100 | 2.637643 | 1.101747 | PASSED |
| off / Associative Recall | 200 | 1.169073 | 0.006384 | PASSED |
| off / Delayed Rule Switch | 200 | 0.581631 | 0.056401 | PASSED |
| off / SmallGraph-v1 | 200 | 0.133532 | 0.001589 | PASSED |
| on / Associative Recall | 200 | 1.169073 | 0.006384 | PASSED |
| on / Delayed Rule Switch | 200 | 0.581631 | 0.056401 | PASSED |
| on / SmallGraph-v1 | 200 | 0.133532 | 0.001589 | PASSED |

这些是 P2 工程健康条件，不构成相对单体基线的科学收益结论。

### 3. 稀疏路由与真实调用

- 两个 600-step 分支结果一致：`valid_tokens=479200`、`executed_assignments=958400`，即每个有效 token 精确执行 2 个 optional experts；`exact_top_k=true`、`capacity_drops=0`。
- raw shares 为 `[47.0101%, 34.8538%, 18.1361%]`，三个专家均超过 5% smoke 健康护栏。
- executed shares 为 `[41.2461%, 33.7768%, 24.9770%]`；`reroute_rate=23.6329%`，raw 与 executed 口径在报告中分离。
- 三个 optional experts 的真实 active calls 分别为 episodic `395303`、working `323717`、predictive `239380`，均大于 0。
- 集成测试 `test_unselected_expert_is_never_executed` 使用失败桩证明未选专家不会执行；路由/容量/padding/稳定 tie-break 测试通过。

### 4. MAC、延迟与观测覆盖

- optional active MAC：`22,489,618,944`；对应 dense optional MAC：`32,631,603,200`，前者严格小于后者，节省 `10,141,984,256` MAC。
- profiler 参数覆盖率 `99.3124%`，无 unsupported parameter 条目。
- 端到端估算 active MAC 为 `75,058,207,744`；MPS 延迟 10 次采样为 P50 `661.890 ms`、P95 `696.113 ms`。
- 只有三个 eligible experts，活跃率固定为 `2/3`；`eligible>=4 时 <=60%` 在 P2 正确标记为 N/A，不能据此声称端到端收益。

### 5. 六模块梯度与状态监控

- telemetry off/on 的联合训练 gradient coverage 对 sensory、episodic、working、predictive、selector 和 router 六模块均为 `true`。
- 预训练汇总中 router 为 `false`，符合四个锚点阶段冻结 router 的预注册设计；完整联合 suite 已覆盖 router 的有效 forward/backward。
- 报告包含 gradient cosine、每模块 state norm/change、raw/executed routing、熵、CV、容量与 reroute 统计。

### 6. Telemetry 三层等价与 schema

- 完整 600-step 分支：最大参数差 `9.731156751513481e-7`，最大任务指标差 `0.0`，满足 MPS 容差和主指标 `1e-4` 门槛。
- telemetry-off 产生 0 个事件；telemetry-on 产生 3600 个事件，恰为 `600 steps × 6 modules`。每个 `(step,module_id)` 恰有一条事件，event ID 唯一。
- 3600 条事件逐条通过 `schemas/telemetry-v1.json`；六个注册 module ID 均出现。
- 固定输入以及单训练 step 的 telemetry off/on 等价测试通过，覆盖输出、loss、state、gradient 和参数更新；完整分支覆盖最终参数与任务指标等价。
- telemetry wall-clock 开销为 `10.437 s`，相对 off 分支为 `1.1939%`。时间戳、event ID 和 wall-clock 未参与数值等价比较。

### 7. Checkpoint 与 P1 兼容

- 抽查的阶段、共享、latest 和 final 文件均为 `modular-checkpoint-v2`，保存 6 个模块 state、3 个 sampler、64 个 batch item 的 TBPTT counters、课程阶段/游标、冻结集合、optimizer groups、配置哈希及 RNG。
- checkpoint-v2 测试通过：配置兼容哈希、完整课程/RNG 恢复、加载前全量预验证、不兼容 ID/version/shape/config 拒绝、六模块与三 sampler 完整性。
- P1 checkpoint-v1 CPU 位级恢复与 MPS 容差恢复测试随当前全量测试通过；P1 公共配置和训练路径未被 P2 CLI discriminant 破坏。

### 8. 本地质量门禁与科学边界

- `make check`：Ruff、format、mypy、pytest 和 CPU environment smoke 全部返回 0；pytest 为 `151 passed, 1 skipped`，覆盖率 `89%`。
- `pre-commit run --all-files`：Ruff lint、Ruff format 和仓库卫生检查全部通过。
- 源码、环境、配置和依赖声明未发现 atlas、Three.js、Nilearn、WebSocket、viewer 或 3D 运行依赖；唯一命中是 telemetry schema 中明确声明“不含 atlas/web/viewer 依赖”的说明文字。
- `docs/scientific-boundaries.md` 与 `docs/p2_implementation_spec.md` 明确六模块是人工计算抽象，不声称脑区一一对应、生物等价或临床意义。

## 远程 CPU CI

| 条件 | 状态 | 说明 |
|---|---|---|
| 当前 SHA 的远程 GitHub Actions CPU CI 绿色 | **PASSED** | CI run [`29494555468`](https://github.com/liberal-CloudArchitect/neuromorphic_network/actions/runs/29494555468)，head SHA `9499a33645e0b446e2039b790387b240f5758bda`，`completed/success`。 |
| `quality` job | **PASSED** | Job [`87608327865`](https://github.com/liberal-CloudArchitect/neuromorphic_network/actions/runs/29494555468/job/87608327865)，Ubuntu latest，`completed/success`；所有 14 个 setup、quality、P1/P2 smoke 和 post 步骤均为 `success`。 |

公开 GitHub API 显示 workflow 于 `2026-07-16T11:28:33Z` 创建并开始，`2026-07-16T11:31:36Z` 更新为完成；quality job 于 `2026-07-16T11:28:36Z` 开始，`2026-07-16T11:31:35Z` 完成。关键步骤 `Test`、`CPU smoke test`、`P1 deterministic training smoke` 和 `P2 modular CPU micro smoke` 均为 `completed/success`。

## 限制与 P3 边界

- P2 只证明模块化闭环、真实稀疏执行、恢复、成本记账和 telemetry 语义等价；不证明类脑模块优于 GRU/Transformer，也不证明生物机制等价。
- 多种子收益、公平参数/计算基线、正式消融、因果扰动、统计置信区间和科学否证属于 P3，不得由本次 seed-7 smoke 外推。
- 3D/atlas/WebSocket/viewer 仍只是后续可选 telemetry 消费层，不得进入 forward、loss、路由、优化器或 Gate 科学结论。

## 阶段结论

GATE-2 的本地 MPS 与远程 CPU 两条证据链均已闭合。本报告接受 P2 的工程实现与冻结边界；项目可更新完成台账、接受 ADR-004 并升级至 `neuromorphic 0.3.0`，随后进入 P3 多种子收益、基线对照、消融与统计验证。
