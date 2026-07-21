---
gate: GATE-NN-MVP
status: FAILED
reviewed: 2026-07-21
source_run: p3-full-dc6c259c-20260719T034230Z
protocol: p3-protocol-v2
---

# GATE-NN-MVP 科学收益裁决

## 结论

`GATE-NN-MVP FAILED`。正式实验完整，但冻结的三个模块因果条件和至少两个总体收益类别
均未满足。因此不生成 `network-mvp-v1` bundle，不允许使用“网络 MVP qualified”表述。

## 三模块因果 family

| 模块 | full−ablated | 95% CI | Holm p | 门槛 | 结论 |
|---|---:|---:|---:|---:|---|
| Episodic | 0.4212 query accuracy | [0.3257, 0.4958] | 0.000600 | ≥15pp | PASS |
| Working | 0.2083 response accuracy | [0.1440, 0.3180] | 0.000600 | ≥10pp | PASS |
| Predictive | −0.0006 analysis AULC | [−0.0025, 0.0009] | 0.5179 | 相对 ≥10% | FAIL |

情景记忆和工作记忆的人工机制获得任务内因果支持。Predictive loss 置零没有产生预注册的
样本效率下降，且急性 predictive-off 在三个任务、三个 seed 上行为差异精确为 0；因此
“三个模块均有因果贡献”的联合条件被否证。

## 网络收益类别

| 类别 | 对 GRU | 对 Transformer | 类别结论 |
|---|---|---|---|
| Task score | raw proxy +0.2030；primary 定义不完整 | raw proxy −0.0860 | INVALID/FAIL |
| OOD | proxy +0.2124；primary 定义不完整 | proxy −0.0395 | INVALID/FAIL |
| AULC | +0.1463，relative +21.9%，PASS | −0.1274，relative −13.5%，FAIL | FAIL |
| Forgetting | −0.0439，CI 跨 0 | +0.1939；阈值未冻结 | INVALID/FAIL |
| Sparse routing | optional MAC −37.6% | 最大任务下降 10.4pp | FAIL |

通过类别为 0/2。即使忽略 `DR-001`～`DR-003`，模块化模型仍在 Transformer AULC、
predictive causal 和稀疏非劣性三个已冻结硬条件上失败，结论不依赖偏差处理方式。

## 成本与路由

- 三个 optional expert 的聚合调用份额分别约为 episodic 30.1%、working 44.9%、predictive 25.0%，未见调用塌缩。
- optional active MAC 相对 dense 平均减少 37.6%，profiler coverage 最低 99.31%。
- Sparse MPS P50 latency 中位数为 140.68ms，dense 为 128.63ms；MAC 减少没有转化为延迟收益。
- Associative Recall 相对 dense 下降 10.4pp，超过 2pp 非劣界。
- Formal summary 没有 capacity-drop 字段，不能独立满足容量健康条件。

## 科学解释边界

本次结果支持两个特定人工状态机制在对应合成任务中的作用，但否证当前网络作为整体达到
预注册 MVP 的主张。结果不证明或否定真实大脑机制，也不能通过改 seed、事后换指标或
补写阈值来改判；架构迭代必须进入新阶段和新 protocol。
