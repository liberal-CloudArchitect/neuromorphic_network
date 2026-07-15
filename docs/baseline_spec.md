---
title: 公平基线规格
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-15
---

# 公平基线与成本匹配规格

本文冻结 P1 基线结构、弱基线、匹配容差和成本披露。所有模型使用同一任务样本、mask、split、seed、优化器规则、验证频率和停止条件。

## 单体神经基线

| ID | 冻结结构 | 序列约束 |
|---|---|---|
| `gru-monolithic-v1` | 统一输入投影；单层 GRU；hidden size 128；任务专用 linear head | 按 valid mask 处理 padding；episode 边界显式清 state |
| `transformer-monolithic-v1` | 统一输入投影；hidden size 128；2 层；4 heads；FFN 512；任务专用 linear head | 严格因果 attention mask；padding key mask；不读取未来 token |

输入编码只做任务 schema 到固定维度张量的无状态转换；它不得包含目标泄漏、oracle 特征或额外记忆。任务 head 的动作 mask 与集合目标处理对所有模型一致。

## 弱基线与 oracle

| 任务 | 弱基线 | 资格标准 |
|---|---|---|
| Associative Recall | 均匀随机 value；多数 value；简单精确 key/value memory | 随机约 `1/32`；精确 memory 在合法 fixture 为 100% |
| Delayed Rule Switch | 随机二分类；始终使用初始规则的 fixed-rule；显式保存最新 cue 的 rule-state oracle | fixed-rule 在无切换为 100%；rule oracle 全体为 100% |
| SmallGraph-v1 | 合法动作均匀随机；贪心节点 ID 启发式；BFS 最短路径 oracle | oracle 100% success、path excess 0、next-state 100% |

弱基线用于验证任务和解释下限，不替代公平单体主比较。任何 oracle 未达到资格标准时，任务实现视为错误，不得运行正式实验。

## 主比较：参数匹配

- 主比较工具接收目标模型可训练参数量 `P_target`，仅通过 hidden size/FFN size 的确定性搜索选择最接近的单体配置。
- 匹配误差定义为 `abs(P_baseline - P_target) / P_target`，必须 ≤5%；输入投影与所有任务 head 均计入可训练参数。
- 候选差距相同时选择较小模型；搜索空间、选中配置和参数明细写入 manifest。
- 若合法搜索空间内无法达到 ±5%，正式比较不得开始；不能靠冻结参数、添加未使用参数或排除 head 人为匹配。

## 敏感性比较

| 比较 | 必须匹配 | 允许差异与报告 |
|---|---|---|
| 训练计算匹配 | 相同训练样本/token、optimizer steps、validation 次数、调参 trials | 报告参数量、总估算 MAC、device 与 wall-clock |
| 推理成本匹配 | 相同 batch、序列长度、device、dtype；估算活跃 MAC 目标 ±5% | 报告总/活跃参数、P50/P95 latency、吞吐与 profiler 覆盖率 |

训练计算和推理成本比较是敏感性分析，不能替代参数匹配主结论。无法同时匹配的维度必须量化披露。

## MAC 与延迟计量

- P1 内建计数器只覆盖 Linear、GRU、Multihead Attention/Transformer FFN 和任务 head；不新增 FLOPs 第三方依赖。
- 每个支持算子记录估算 MAC、调用次数和输入 shape；`coverage = 已计数支持算子参数 / 全部可训练参数`，覆盖率随结果报告。
- 未支持算子必须列名和参数量，不得把未知成本记为零。正式成本结论要求 profiler coverage ≥95%。
- latency 在 10 次 warm-up 后测量 50 次；同步设备后报告 P50/P95、batch、sequence length、dtype、device 和 wall-clock。MPS/CPU 结果不可直接混合。
- 稀疏路由的活跃 MAC 只统计实际选中的可选专家；编码器、路由器和选择器始终计入端到端总 MAC，但不进入专家激活率。

## 公平性与审计

- 同一 model seed 使用同一数据顺序、split 和 validation checkpoint 规则；正式配置在 test/OOD 前冻结。
- 默认优化器为 AdamW、batch 64、最多 5,000 steps、每 100 steps 验证、patience 10、`min_delta=0.001`。
- 调参预算 P1 为零：使用冻结默认值。后续若调参，各模型 trials 数和搜索空间复杂度必须预注册并相同。
- 所有启动 run 均登记；NaN、超限、中断和资格失败不事后删除或替换 seed。
- 结果同时披露参数量、样本/token、optimizer steps、验证次数、估算 MAC、coverage、P50/P95 latency、wall-clock、峰值内存和失败 run。
