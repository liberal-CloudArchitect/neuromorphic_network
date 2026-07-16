---
title: 实验协议
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-16
---

# P1 任务与实验协议

本文冻结 P1 的任务生成、划分、预算、seed、指标和停止规则。实现必须由 `(task_version, split_seed, sample_index)` 唯一决定样本；测试集和 OOD 不参与配置选择。

## 共同数据协议

| split | `split_seed` | smoke 样本 | 正式 Associative Recall 样本 | 用途 |
|---|---:|---:|---:|---|
| train | `1101` | 64 | 8,192 | 梯度更新 |
| validation | `2201` | 32 | 2,048 | 早停/选 checkpoint |
| test | `3301` | 32 | 2,048 | 一次性 ID 报告 |
| OOD | `4401` | 32 | 2,048 | 一次性分布外报告 |

- task version 分别为 `associative-recall-v1`、`delayed-rule-switch-v1`、`small-graph-v1`。
- 每个样本保存规范序列化后的内容 SHA-256；四个 split 的哈希集合必须两两不相交。
- smoke model seed 固定为 `7`，最多 200 optimizer steps；它只验证可运行性，不能支持科学结论。
- P1 正式 confirmatory run 只冻结 Associative Recall GRU，model seeds 为 `[17, 29, 43]`。训练 8,192 个确定性样本，每 seed 最多 5,000 optimizer steps。
- 默认 AdamW、batch size 64、每 100 steps 验证、patience 10 次验证、`min_delta=0.001`。最佳 validation 主指标 checkpoint 用于 test/OOD。
- Delayed Rule Switch 与 SmallGraph-v1 在 P1 完成三任务 seed-7 smoke；它们的多 seed confirmatory 比较在模块实现可用后另行预注册，不用 smoke 结果关闭科学假说。

## Associative Recall

**生成**：key 与 value 分别来自 32 个离散符号。每个样本先无放回采样 key/value 对并依次呈现，再插入与已用 key 不同的干扰 token，最后呈现一个已出现 key 的 query；target 是其绑定 value。ID 为 4–8 对、0–4 个干扰项；OOD 为 9–12 对、5–8 个干扰项。query 位置之前无监督 target，`loss_mask` 仅在 query response 位置为真。

**指标**：主指标 `query_accuracy`；次指标为按干扰数量分层的准确率与 `interference_drop = accuracy(no/low interference) - accuracy(high interference)`，越低越好。正式对比按相同 sample index 配对。

**有效性检查**：query key 必须恰好对应一个先前 pair；干扰不得重复 query key；padding 不计 loss；简单 key/value oracle 必须达到 100%。

## Delayed Rule Switch

**生成**：刺激为两位 `(x0,x1)`，四条规则分别输出 `x0`、`x1`、`xor(x0,x1)`、`xnor(x0,x1)`。episode 开始给出 rule cue，此后 cue 被 mask；经过 delay 后要求响应。ID delay 为 2–8，episode 至多在一个非响应位置切换一次并发出新 cue；OOD delay 为 9–16，切换位置来自训练未使用的归一化位置区间 `[0.70,0.90]`。

**指标**：主指标为有效响应位置的 `response_accuracy`；次指标 `switch_cost = accuracy(no-switch) - accuracy(first response after switch)`，越低越好，并报告按 delay 分层准确率。

**有效性检查**：四规则与刺激组合均覆盖；response 前的 target 不进入 loss；fixed-rule oracle 必须在无切换子集达到 100%，规则状态 oracle 在全体达到 100%。

## SmallGraph-v1

**生成**：确定性生成简单、无向、连通图；先建立随机生成树，再在不超过最大度 4 的前提下增加边。ID 为 6–10 个节点，OOD 为 11–16 个节点。每个样本采样不同的 start/goal，时间步输入包含当前节点、目标节点和当前节点按节点 ID 升序排列的最多 4 个邻居动作槽。执行合法动作后转移至对应邻居；无效槽被 mask。

**监督目标**：行为克隆目标是所有使最短路距离减少 1 的动作槽集合；loss 使用集合目标概率之和，不强制任意单一 tie-break。辅助 target 是执行动作后的节点，用于 `next_state_accuracy`。最大 rollout 长度为 `2 * node_count`，到达目标后 episode 结束。

**指标**：主指标 `success_rate`；次指标 `path_excess = executed_steps - shortest_path_length` 和 `next_state_accuracy`。最短路径 oracle 必须达到 100% success 且 path excess 为 0。

**有效性检查**：图必须连通、无自环/重边且度不超过 4；目标集合非空；非法动作永不进入集合目标；OOD 节点数和图种子空间不与 ID 重叠。

## 训练与评估层级

1. **Deterministic fixtures**：固定 sample index 验证内容、mask、oracle、split hash 和 OOD 边界。
2. **Forward/backward smoke**：GRU 与 Transformer 在三任务、CPU 与可用的 MPS 上各完成一个 batch；三任务 seed-7 训练 loss 下降且无 NaN。
3. **Baseline qualification**：验证弱基线、参数匹配和成本记录；失败 run 不删除。
4. **P1 confirmatory**：Associative Recall GRU 三个正式 seed，冻结配置后不再调参；生成每 seed 与汇总报告。
5. **后续模块实验**：按 evidence registry 的模块消融和总体五类收益门槛执行，不能复用 P1 smoke 作验证性结果。

## 网络总体判据

相对参数匹配主基线，以下五类收益中至少两类达到门槛，且 95% CI 支持预期方向：任务分数相对提高 ≥5%；OOD 归一化分数提高 ≥5%；达标样本数减少 ≥15%；三任务平均遗忘绝对降低 ≥5pp；可选专家活跃 MAC 减少 ≥20% 且任务分数不劣超过 2%。该判据属于后续完整模块模型，不是 P1 单体基线的通过标准。

P3 使用 `p3-protocol-v2`：经 `CR-002` 将“达标样本数减少 ≥15%”替换为 analysis split 上归一化 AULC 相对提高 ≥15%。shared GRU 与 shared Transformer 均为主基线；前四类收益必须相对两者同时达标。稀疏收益与 dense modular 对照。详细定义以 `docs/p3_implementation_spec.md` 为准。

## 排除、失败与偏差

- 配置/数据验证失败的 run 不启动训练；启动后的 NaN、梯度异常、资源超限和中断均计为失败 run并保留 manifest。
- 仅允许在看到 test/OOD 前依据 validation 应用已冻结早停；不得替换 seed 或延长个别模型预算。
- 对任务、split、seed、预算、主指标或规则的变更必须新建 task/protocol version，并记录 deviation；不得覆盖 v1 结果。
