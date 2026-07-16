---
title: 统计分析协议
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-16
---

# 统计分析与复现协议

本文冻结 P1 及后续模块实验的重复单位、置信区间、多重比较、失败 run 和数值容差。smoke、探索性和 confirmatory 结果必须分开标记。

## 分析单位与 seed

- 独立训练 seed 是模型比较的主要重复单位；同一 seed 的模型使用相同数据 sample index 和评估样本，构成配对。
- P1 Associative Recall GRU 正式 seeds 固定为 `[17, 29, 43]`；seed `7` 只用于三任务 smoke，不进入 confirmatory CI。
- episode/sample 是 seed 内观测，不得伪装为独立训练重复。token 只用于计算 mask 后的指标，不作为独立重复。
- bootstrap RNG seed 固定为 `20260715`；重采样次数固定为 10,000，报告双侧 percentile 95% CI。

## 分层配对 bootstrap

比较模型 A 与 B 时，每次重采样执行：

1. 有放回抽取训练 seed，A/B 保持同 seed 配对；
2. 在被抽中的 seed 内，按预注册任务 strata 有放回抽取相同 sample index，A/B 保持样本配对；
3. 计算各 seed 指标差，再对 seed 等权平均；
4. 取 2.5% 和 97.5% percentile 作为区间。

strata 冻结为：Associative Recall 的 pair-count × interference-count bucket；Delayed Rule Switch 的 delay bucket × switch/no-switch；SmallGraph-v1 的 node-count × shortest-path-length bucket。空 strata 不补样本，实际 strata 数量随报告披露。

单模型绝对指标也使用相同步骤但不做 A/B 差值。只有三个训练 seed 时 CI 精度有限，必须同时报告所有 seed，不用窄区间措辞夸大确定性。

## 主指标与方向

| 范围 | 主指标 | 预期方向 / 最小效应 |
|---|---|---|
| Associative Recall | query accuracy | 情景记忆消融差 ≥15pp；总体任务收益相对 ≥5% |
| Delayed Rule Switch | response accuracy | 工作记忆消融差 ≥10pp；switch cost 更低 |
| SmallGraph-v1 | success rate | 预测器达标样本数减少 ≥10%；总体样本效率门槛 ≥15% |
| 稀疏路由 | 主任务分数与活跃 MAC 联合 | 分数非劣界限 2%，活跃 MAC 减少 ≥20% |
| 网络总体 | 五类预注册收益 | 至少两类达到门槛且对应 95% CI 支持预期方向 |

`pp` 表示绝对百分点；“相对提高”定义为 `(model - baseline) / abs(baseline)`。基线为零时只报告绝对差，不计算相对值。

### P3 protocol-v2 补充

- 经 `CR-002`，样本效率主指标改为 analysis split 上的归一化学习曲线 AULC；总体门槛为相对提高 ≥15%，predictive retrained ablation 门槛为相对恶化 ≥10%。
- analysis split seed 固定为 `5501`，正式大小为 512；不参与调参、早停或 checkpoint 选择。
- P3 confirmatory 主基线为 shared GRU 与 shared Transformer。task score、OOD、AULC、forgetting 分别与两种基线比较，共八个主比较；一个收益类别必须同时通过两种基线的阈值、CI 和 Holm 校正。
- P3 因果 family 固定为 episodic、working、predictive 三项 retrained contrasts。
- paired bootstrap 必须先按 model/variant、seed、task、distribution、sample index 和 stratum 严格对齐；不允许把两个模型的样本独立重采样。

## 多重比较

- confirmatory family 由同一 Gate 中所有“模型优于主基线”的主指标比较组成；在查看结果前固定 family。
- 对每个比较由配对 bootstrap 计算双侧 p 值：`2 * min(P(delta <= 0), P(delta >= 0))`，最小值为 `1/10000`。
- 使用 Holm step-down 方法控制 family-wise α=0.05；同时报告原始 p、Holm 调整 p、效应量和 CI。
- 模块专用阈值、OOD 次指标和成本敏感性均报告，但除非预注册为主比较，不可替换失败主指标。

## 失败、缺失与停止

- 配置或 fixture 验证失败在训练前标记 `INVALID`；启动后 NaN、梯度异常、资源超限或中断标记 `FAILED`，均保留 manifest。
- confirmatory seed 不得替换。若任一比较模型缺少配对 seed，该 seed 不进入效应估计，同时报告缺失原因；成功 seed 少于 3 时结论为 `INCONCLUSIVE`，不能通过科学 Gate。
- validation 早停固定为每 100 steps、patience 10、`min_delta=0.001`，最多 5,000 steps。test/OOD 只评估所选 checkpoint 一次。
- 不允许根据 test/OOD、显著性或 CI 延长预算、增加 seed 或修改 strata；任何变化建立新 protocol version 和 deviation record。

## 数值复现容差

- CPU 同 checkpoint、数据顺序和恢复游标的连续训练与恢复训练要求模型/优化器状态及后续指标逐位一致。
- MPS checkpoint 恢复后的张量使用 `allclose(rtol=1e-5, atol=1e-6)`，主指标绝对差 ≤`1e-4`。CUDA 在对应硬件上另行冻结，不能从 MPS 推测。
- telemetry on/off：固定 checkpoint/input 的 packet 与 logits 使用同设备上述 `allclose`；单训练步 loss、所有 gradient 和 optimizer update 使用相同容差；整段训练主指标差 ≤`1e-4`。
- 跨 CPU/MPS 只要求指标和误差在单独报告的容差内，不默认逐位一致；device、PyTorch 版本和确定性设置必须记录。

## 报告字段

每份正式报告必须包含全部 seed、均值/中位数、绝对与相对效应、10,000 次 bootstrap 95% CI、原始与 Holm 调整 p、失败/缺失 run、参数量、MAC coverage、P50/P95 latency、wall-clock 和峰值内存。CI 跨 0、效应低于门槛、匹配失败或成功 seed 少于 3 时分别写为“不支持”“工程收益不足”“比较无效”或“不确定”，不得写成通过。
