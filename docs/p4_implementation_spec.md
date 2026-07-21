---
title: P4 跨步预测闭环与语义稀疏路由实施规格
status: ACCEPTED
phase: P4
protocol: p4-protocol-v1
target_release: 0.5.0
last_updated: 2026-07-21
---

# P4 跨步预测闭环与语义稀疏路由实施规格

P4 修订 P3 中未获得支持的预测因果路径和稀疏执行路径。确认对象是 `predictive_adapter.v2` 的跨步闭环、`episodic_memory.v1` / `working_memory.v1` 的 semantic top-1 路由，以及由此产生的 ID、OOD、AULC、顺序训练遗忘、预测质量和计算稀疏性。目标软件版本为 `0.5.0`，科学裁决入口为 `GATE-NN-MVP-v2`。

## 架构冻结

### 跨步预测闭环

`predictive_adapter.v2` 必须按以下顺序执行，禁止同一步标签或未来状态泄漏：

1. 第 `t` 步只消费第 `t-1` 步动作形成后提交的 forecast；仅当 `source_step + 1 == t` 且 episode 连续时有效。
2. forecast 与第 `t` 步 sensory representation 比较，形成 temporal loss 和误差反馈；反馈为 `0.25 * tanh(...)`，绝对值上界为 `0.25`。
3. 路由、记忆和 action selector 使用修正后的表示形成第 `t` 步动作。
4. 动作形成后，才以第 `t` 步 sensory representation 和实际所选动作提交第 `t+1` 步 forecast。
5. episode 结束、padding 或不连续 step 不得消费旧 forecast；terminal 必须清除 pending forecast。

`predictor-off`、`loss-zero`、`feedback-zero` 是不同干预：分别关闭整个预测器、只移除训练损失、只切断反馈。它们不得互相替代。

`predictive_adapter.v2` 的模块级 `consume`、`commit` 和 `forward` 只接受 `T=1`。多步序列必须由 `ModularBrainNetworkV2.forward_batch` 执行逐步 `consume → selector → commit`，防止直接模块调用把同一旧 forecast 错用于多个时间步。

### Semantic top-1 记忆路由

- required path 固定为 sensory encoder → `predictive_adapter.v2` → `sparse_router.v2` → action selector。
- optional experts 仅为 `episodic_memory.v1` 和 `working_memory.v1`；每个有效 token 精确执行一个 optional expert。
- Associative Recall 的 STORE/QUERY 语义 token 必须 100% reservation 到 episodic，reservation 优先于 learned score。
- 其余有效 token 在 episodic / working 之间执行 stable semantic top-1；未选中的 expert 不得执行。
- `capacity_drops` 必须为 `0`。dense-memory 仅作为冻结对照，不得改变 learned top-1 的定义。

## 数据与随机性

三个任务保持 `associative_recall.v1`、`delayed_rule_switch.v1`、`small_graph.v1`，但 P4 使用独立 namespace 和独立 split seeds，不复用 P1–P3 样本流。

| split | P4 seed | 用途 |
|---|---:|---|
| train | 11101 | 参数更新 |
| validation | 12201 | early stop 与 pilot 选择 |
| analysis | 15501 | AULC、预测质量与顺序训练分析 |
| test | 13301 | 冻结 checkpoint 的 ID 评估 |
| OOD | 14401 | 冻结 checkpoint 的 OOD 评估 |

正式训练 seeds 固定为 `[17, 29, 43]`；seed `7` 只用于 qualification 和 pilot，不进入正式 CI。正式样本量固定为 train 8192、validation 2048、analysis 512、test 2048、每个 OOD distribution 2048。

## 运行阶段与矩阵

| profile | cell 数 | seed / 设备 | 作用 |
|---|---:|---|---|
| qualification | 8 | `7` / CPU micro + MPS | 缩小数据和步数验证 8 种 mechanism 路径、恢复、产物和数值健康；`qualification_only=true` |
| pilot | 4 | `7` / MPS | 仅访问 train/validation，在四个冻结 preset 中选择一个；不产生正式统计 |
| mechanism | 24 | `[17,29,43]` / MPS | 每 seed 8 cell，裁决预测闭环机制；累计墙钟上限 24 小时 |
| full | 81 | `[17,29,43]` / MPS | 24 个 mechanism cell 加 57 个网络比较 cell；累计墙钟上限 72 小时 |

每个 seed 的 8 个 mechanism cell 固定为：full、retrained predictor-off、retrained loss-zero、retrained feedback-zero，以及 frozen-checkpoint acute-feedback-off、shuffle-forecast、dense-memory、legacy-capacity。故 mechanism 矩阵为 `8 × 3 = 24`。

full 在这 24 个 cell 之外，每 seed 增加 19 个 cell：2 个 shared 主基线、9 个 per-task 模型、3 个三任务顺序训练模型、episodic-off 与 working-reset 两个记忆因果 cell，以及 direct-head、frozen-random-encoder、shallow-encoder 三个结构控制。故新增 `19 × 3 = 57`，总数为 `24 + 57 = 81`。

pilot 的四个 preset 各执行 1,000 个三任务确定性 round-robin updates，分别为 learning rate `{1e-4, 3e-4}` × temporal loss weight `{0.05, 0.10}`，weight decay 固定 `1e-2`。候选必须满足 forecast coverage ≥`0.95`、forecast error 相对 persistence error 改善 ≥`5%` 且 feedback 非零；合格候选再按 validation macro AULC 降序、最终 loss 升序、preset ID 升序选择。pilot lock 必须绑定 clean SHA。

正式阶段只能按 qualification lock → pilot lock → 24-cell mechanism lock → 81-cell full 的顺序启动；不得绕过前置 lock 或在正式运行中重新选择 preset。

24-cell mechanism 全部完成后，runner 必须自动生成 `mechanism-report.json`，按冻结的 10,000 次配对分层 bootstrap、预测质量与稀疏非劣性阈值裁决。只有报告为 `PASSED` 时才生成同 SHA、带 checksum 的 `artifacts/p4/mechanism-lock.json`；矩阵完成但科学阈值失败时状态为 `mechanism_failed`，且 full 继续拒绝启动。

## 冻结指标与门槛

所有正式比较使用三个训练 seed、10,000 次 seed→stratum/sample 严格配对 bootstrap、RNG seed `20260715`、双侧 95% percentile CI，并在冻结 family 内执行 Holm 校正。收益比较必须同时满足效应门槛、CI 支持预期方向和 Holm-adjusted `p <= 0.05`。

| 类别 | 定义 | 通过门槛 |
|---|---|---|
| chance | 每个样本按任务 chance 标准化：`(score - chance) / (1 - chance)`；SmallGraph 使用该图、起点、终点和 horizon 下 uniform-valid-action live rollout 的精确到达概率 | chance 计算可复现；非法图、`chance >= 1` 或缺失 live rollout 直接使比较无效 |
| ID | test 上逐样本 chance-normalize，再对三个任务等权平均 | 相对 shared GRU 和 shared Transformer 均提高 ≥`5%` |
| OOD | 仅使用 SmallGraph `scale`、`topology`、`joint` 的 live rollout；每个 view 计算 `(OOD - chance) / (ID - chance)` | `ID > chance`，且相对两个主基线均提高 ≥`5%` |
| AULC | analysis split 上按固定最大预算归一化学习曲线积分 | 相对两个主基线均提高 ≥`15%` |
| forgetting | 每任务 `历史最佳 analysis score - 最终 score`；效应方向为 `baseline - modular-v2` | 相对两个主基线的绝对降低均 ≥`0.02` |
| 预测因果 | full 与相同 seed 随机初始化的 retrained predictor-off 比较 | macro AULC 相对提高 ≥`5%`，且每个任务最终分数下降不超过 `2pp` |
| 预测质量 | 有效连续 transition 上比较 forecast error 与 persistence error | 聚合相对误差降低 ≥`5%`，且至少两个任务的误差降低为正 |
| 稀疏执行 | learned semantic top-1 与 dense-memory 对照 | optional active MAC 减少 ≥`20%`；每任务分数损失 ≤`2pp`；`capacity_drops=0`；AR STORE/QUERY episodic reservation/execution=`100%` |

`GATE-NN-MVP-v2` 要求 ID、OOD、AULC、forgetting、预测因果、预测质量和稀疏执行按以上冻结定义完整报告。qualification、pilot 或 mechanism 通过只授权进入下一阶段，不等同于该科学 Gate 通过。

## 科学与历史边界

- 本阶段只检验人工网络中的计算机制，不构成真实脑区一一对应、生物等价、BOLD 或临床结论。
- `predictive_adapter.v2`、`sparse_router.v2` 和 P4 split namespace 是新版本路径；不得替换 P3 artifact 中的 v1 module ID、样本或统计定义。
- P3 的 `GATE-NN-MVP FAILED`、`p3-protocol-v2` 报告和 `0.4.0` 历史保持原样。P4 结果只能形成新的 `GATE-NN-MVP-v2` 裁决，不得回写、重算或重新解释 P3。
- `0.5.0` 只是 P4 的目标软件版本；只有 `GATE-NN-MVP-v2` 正式通过后，才能使用相应的 qualified 表述。

## 版本规则

`p4-protocol-v1` 冻结上述架构顺序、split seeds、训练 seeds、预算、preset、矩阵、比较 family、指标、阈值和统计方法。任何一项改变都必须建立新的 protocol version 和 change request，并保留本版本的运行与失败记录；不得对本协议或旧 Gate 进行结果驱动的回填。
