---
title: 六模块计算假说
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-15
---

# 六模块计算假说

本文冻结六模块的最小职责、数据语义、状态、损失、对照和否证规则。工程接口以 `core/contracts.py` 为准；本文中的脑科学术语仅是启发来源，不表示脑区一一映射。

## 注册表与公共约束

| 注册 ID | 必经/可选 | eligible-set | 最小职责 |
|---|---|---|---|
| `sensory_encoder.v1` | 必经 | 不适用 | 把任务输入转为统一时序表示 |
| `episodic_memory.v1` | 可选专家 | 三任务均可用 | episode 内快速绑定、读前写和线索检索 |
| `working_memory.v1` | 可选专家 | 三任务均可用 | 有限容量规则/目标状态与门控更新 |
| `predictive_adapter.v1` | 可选专家 | 三任务均可用 | 动作条件下一状态预测与辅助误差 |
| `sparse_router.v1` | 必经 | 不适用 | 每 step 从可选专家中默认选 top-2 |
| `action_selector.v1` | 必经 | 不适用 | 依据表示、专家输出与动作 mask 产生 logits |

- 必经模块不进入可选专家激活率的分子或分母；活跃计算同时报告 MAC 和 wall-clock。
- 所有模块只接受显式 device 上的 packet/state；不得硬编码 MPS、CUDA 或 CPU。
- 主任务损失与辅助损失分开命名、归约和记录。辅助损失必须是标量，不得隐式改变状态。
- telemetry 是有界、已归约、可关闭的旁路输出，不是 forward、loss、路由或动作决策的输入。
- 每个 episode 边界按 `reset_mask` 清空瞬时状态；训练默认每 32 个有效 step 截断并 detach 计算图。

## `sensory_encoder.v1`

- **Evidence / Abstraction**：依据 `EV-SENS-001/002`，抽象多层感觉选择性为共享可学习投影，不模拟具体感受野、皮层柱或脉冲编码。
- **输入**：任务原始 `inputs[B,T,D]`、`valid_mask[B,T]` 和可选目标上下文；padding 位置必须被 mask。
- **输出**：`representation[B,T,F]` 的 `BrainPacket`，保持时间轴、episode 与 device；不产生动作。
- **状态与时间尺度**：P1 基线无运行时状态；未来短时上下文必须版本化并在 episode 边界重置。
- **损失**：主任务梯度；可选 `sensory.reconstruction` 或 `sensory.contrastive` 标量辅助损失，默认权重 0。
- **假说**：共享编码在三任务上相对冻结随机和浅层容量匹配编码提高平均 OOD 分数至少 5%。
- **对照/否证**：冻结随机、浅层容量匹配、输入通道置乱；95% CI 跨 0 或收益由参数失配解释即否证。

## `episodic_memory.v1`

- **Evidence / Abstraction**：依据 `EV-EMEM-001/002`，抽象快速学习与模式分离为有限容量 key/value 记忆；不实现海马微回路或巩固。
- **输入**：编码 packet、episode-local state、查询标记和 `reset_mask`。
- **输出**：检索表示及可选检索置信度；写入发生在本 step 读取完成之后。
- **状态与时间尺度**：每个 batch item 独占 memory slots、占用 mask 与写指针；episode 边界完全清零，batch 间不得共享。
- **损失**：`episodic.retrieval`、可选 `episodic.separation`；均按有效查询归约为标量。
- **假说**：Associative Recall query accuracy 相对无记忆提高至少 15pp，并保持 OOD 干扰收益。
- **对照/否证**：禁读、禁写、写前读顺序反转、随机检索、简单 key/value memory；效应不足 15pp、CI 跨 0 或简单记忆同等即否证。

## `working_memory.v1`

- **Evidence / Abstraction**：依据 `EV-WMEM-001/002`，把动态分布式任务状态抽象为有限容量循环槽；不指定单一脑区或必须持续放电。
- **输入**：当前编码、可选 `goal_context`、旧工作状态和 `reset_mask`；更新门不得由选择器或未来标签提供。
- **输出**：当前工作表示、门值摘要和新状态。
- **状态与时间尺度**：规则/目标槽与更新门；跨延迟步保持、episode 边界清零；默认每 32 step detach。
- **损失**：主任务损失、可选 `working.state_consistency` 和 `working.gate_regularization`。
- **假说**：Delayed Rule Switch 相对状态清零/无工作记忆提高至少 10pp，并降低 OOD switch cost。
- **对照/否证**：每步清零、固定/随机门、目标置乱、容量匹配 GRU；效应不足 10pp、CI 跨 0 或单体容量解释全部收益即否证。

## `predictive_adapter.v1`

- **Evidence / Abstraction**：依据 `EV-PRED-001/002`，把层级或奖励预测原则抽象为动作条件下一状态预测；不等同于皮层预测编码或多巴胺信号。
- **输入**：当前 `s_t`、动作副本 `a_t` 和待匹配转移状态；环境返回前不得访问 `s_(t+1)`。
- **输出**：`predicted_next_state`；环境反馈后才计算误差。
- **状态与时间尺度**：最多保存未完成转移 `(s_t,a_t)`；转移完成即消费，episode 结束必须丢弃悬空项。
- **损失**：`predictive.next_state` 标量辅助损失。P0/P1 误差只参与优化，不反馈修改当前或下一工作状态。
- **假说**：SmallGraph-v1 next-state accuracy 提高，且达到目标成功率的样本量至少减少 10%。
- **对照/否证**：无预测器、动作/标签置乱、容量匹配辅助头；若仅辅助 loss 改善而行为/probe/样本效率均不变即否证。

## `action_selector.v1`

- **Evidence / Abstraction**：依据 `EV-ACT-001/002`，把竞争通道选择抽象为统一动作 logits；不模拟基底节直接、间接和超直接通路。
- **输入**：编码、因果上已可用的记忆专家输出、目标上下文、候选集合和有效动作 mask；当前转移预测不得反馈同一步动作。
- **输出**：`action_logits[B,T,A]`；无效动作在 softmax/loss 前被排除。等长最短路径用动作集合目标。
- **状态与时间尺度**：P1 无跨 step 持久状态；冲突统计只是 telemetry。
- **损失**：`action.cross_entropy` 或集合目标负对数似然；可选 `action.calibration`。
- **假说**：冲突条件下优于容量匹配直接 policy head，且 success rate 非劣界限为 2%。
- **对照/否证**：直接 head、候选置乱、mask 故障、冲突分层；直接 head 同等或更好即删除专门选择器。

## `sparse_router.v1`

- **Evidence / Abstraction**：依据 `EV-ROUT-001/002`，把任务依赖的信息流调节抽象为 step 级 top-k；不把专家或 gate 映射为丘脑/皮层脑区。
- **输入**：编码 packet、`task_id` 和本任务 `eligible_modules`；三个任务的可选专家均为情景记忆、工作记忆、预测适配。
- **输出**：每 step 的专家权重和选择 mask；默认 top-2，tie 使用注册 ID 稳定顺序以保证复现。
- **状态与时间尺度**：决策本身为 step-local；负载统计按 train/validation/test/OOD phase 分离。
- **损失**：`router.load_balance` 与 `router.communication_cost`，同主任务损失分开记录。
- **假说**：相对稠密路由在性能不劣超过 2% 时降低可选专家活跃 MAC，并产生可复现的任务条件路由重组。
- **对照/否证**：稠密、固定、均匀随机、任务标签置乱；负载塌缩、计算不降或性能越过非劣界即否证。

## 冻结结论

- 模块注册 ID、必经/可选角色、三任务 eligible-set、默认 top-2 和 step 粒度均在 P0 冻结。
- 记忆采用 `read-before-write`；工作记忆门由模块自身根据编码和目标产生；预测误差在 P0/P1 仅作辅助损失。
- 本文阈值是模块功能主张的预注册门槛；网络总体 Gate 使用 `statistical_protocol.md` 的五类收益与多重比较规则。
