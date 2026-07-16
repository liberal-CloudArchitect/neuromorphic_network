---
title: P2 模块化网络实施规格
status: ACCEPTED
phase: P2
gate: GATE-2
last_updated: 2026-07-16
---

# P2 模块化网络实施规格

本规格在 P2 完整 smoke 运行前冻结。P2 只验证人工模块的工程闭环、状态语义、真实稀疏执行、恢复与观测等价性，不证明生物等价或相对单体基线的科学收益。

## 固定架构

- 非脑 task boundary adapters 将 Associative Recall、Delayed Rule Switch 和 SmallGraph-v1 的输入投影到 `F=128`；adapter/head 参数和 MAC 计入端到端成本，但不计入 optional expert 稀疏率。
- `goal_context` 固定 104 维：事件控制 5、key one-hot 32、动作有效 mask 32、task one-hot 3、实际动作副本 one-hot 32。它只由当前可观察输入与 mask 构造，禁止 target 或未来状态。
- 情景记忆使用 16 个 episode-local slots，只写 Associative Recall 的合法 store event；distractor 不写。读取产生 pending write，动作形成后提交。
- 工作记忆使用 4 个 32 维 slots；预测器只在 SmallGraph-v1 上使用动态实际动作转移 target；其他任务的预测辅助损失为 0。
- 三个 optional experts 为 episodic、working、predictive；每个有效 step 执行两个。sensory、router、selector 始终必经。

## 容量约束路由

router 先按 score、batch index、注册 ID 稳定排序产生 `raw_top2_mask`。每个专家的执行容量为：

```text
ceil(1.25 * valid_tokens * 2 / 3)
```

超出 raw 候选容量时稳定改派下一候选，形成 `executed_mask`。`executed_mask` 必须为每个有效 token 精确选择两个专家且 drops 为 0；raw/executed 份额、熵、CV、reroute rate 和容量使用率分开报告。真实执行必须按专家 slice batch rows 后调用并 functional scatter，禁止稠密计算后乘零。

当前只有三个 eligible experts，因此活跃率固定为 `2/3`；`eligible>=4 时 <=60%` 条件在 P2 标记为 N/A。GATE-2 要求 optional active MAC 小于 dense optional MAC，但不据此声称端到端收益。

## 共享预训练

固定 seed 7，train 64、validation 32、batch 64；只使用 train/validation。四阶段各精确 100 optimizer updates，每 25 updates 验证，不早停，无 scheduler；阶段边界重建 AdamW parameter groups 并保存 checkpoint-v2。

| 阶段 | 数据与顺序 | 可训练部分 | 目标 |
|---|---|---|---|
| sensory | 三任务确定性轮转 | adapters、sensory core、selector core/heads | primary `1.0` |
| episodic | Associative Recall | episodic memory | primary `1.0`、retrieval `0.1`、separation `0.01` |
| working | Delayed Rule Switch | working memory | primary `1.0`、consistency `0.05`、gate `0.001` |
| predictive | SmallGraph-v1 | predictive adapter | actual-action next-state `1.0` |

optimizer 固定为 AdamW：learning rate `3e-4`、weight decay `1e-2`、gradient clip norm `1.0`。每阶段末 10 次平均 loss 必须低于首 10 次平均 loss。

## 联合 smoke 与 loss

共享 checkpoint 克隆为 telemetry off/on 两个分支。每个分支按 Associative Recall → Delayed Rule Switch → SmallGraph-v1 确定性轮转 600 updates，每任务精确 200 updates；train 64，validation/test/OOD 各 32，batch 64。每任务每 25 次更新验证、每 50 次更新 checkpoint，不早停。

联合 loss 权重：primary `1.0`；episodic retrieval `0.1`、separation `0.01`；working consistency `0.05`、gate `0.001`；SmallGraph predictive next-state `0.1`；router load-balance `0.01`、communication-cost `0.001`。

每任务末 10 次平均 loss 必须低于首 10 次。P2 记录任务指标但不设性能收益门槛。

## 等价、恢复与报告

- telemetry 在模型 packet/state/loss 完成后从 detached 归约量构建；off 不归约、不搬到 CPU、不消耗 RNG。
- 固定 checkpoint 推理以及单训练 step 的 packet/logits/state/loss/gradient/update 按设备容差比较；完整两个 600-step 分支的每任务主指标绝对差不得超过 `1e-4`。event ID、时间戳和 wall-clock 不参与数值等价比较。
- checkpoint-v2 保存模块 state、课程阶段/游标、三任务 sampler、TBPTT 计数、冻结集合、optimizer groups、配置哈希和 RNG；兼容哈希排除 run/output/resume/telemetry 字段。
- gradient cosine、state norm/change、raw/executed route、容量、active/dense MAC、latency、peak memory、wall-clock 和 telemetry 开销全部报告。
- raw 每个专家在完整 joint smoke 中至少获得 5% 的聚合选择份额，仅作为工程健康护栏，不作为功能分工或科学否证标准。
