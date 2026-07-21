# P3 正式实验与科学裁决

- Run：`p3-full-dc6c259c-20260719T034230Z`
- 训练提交：`dc6c259c559eb3af6e4cc74c905cc2dfadf3690a`
- 设备：`mps`；seeds：`[17, 29, 43]`
- 实际 suite 墙钟：49.16 小时（上限 72 小时）
- `GATE-3`：**PASSED**
- `GATE-NN-MVP`：**FAILED**

## 完整性

81/81 cells 完成；352 个登记产物 checksum 通过；
共复核 1,605,120 条逐样本记录。没有失败、缺 seed 或资源截断。

## 网络收益 family

| 类别 | 基线 | 效应 | 95% CI | Holm p | 结论 |
|---|---|---:|---:|---:|---|
| task_score | gru | 0.2030 | [0.1970, 0.2092] | 0.0016 | INVALID/FAIL |
| task_score | transformer | -0.0860 | [-0.0911, -0.0808] | 0.0016 | INVALID/FAIL |
| ood | gru | 0.2124 | [0.2023, 0.2269] | 0.0016 | INVALID/FAIL |
| ood | transformer | -0.0395 | [-0.0504, -0.0310] | 0.0016 | INVALID/FAIL |
| aulc | gru | 0.1463 | [0.1354, 0.1517] | 0.0016 | PASS |
| aulc | transformer | -0.1274 | [-0.1324, -0.1213] | 0.0016 | FAIL |
| forgetting | gru | -0.0439 | [-0.1584, 0.0932] | 0.5135 | INVALID/FAIL |
| forgetting | transformer | 0.1939 | [0.1402, 0.2762] | 0.0016 | INVALID/FAIL |

通过类别数：**0/2**。

## 模块因果 family

| 模块 | 指标 | full−ablated | 95% CI | Holm p | 结论 |
|---|---|---:|---:|---:|---|
| episodic | query_accuracy | 0.4212 | [0.3257, 0.4958] | 0.0005999 | PASS |
| working | response_accuracy | 0.2083 | [0.1440, 0.3180] | 0.0005999 | PASS |
| predictive | analysis_aulc | -0.0006 | [-0.0025, 0.0009] | 0.5179 | FAIL |

情景记忆与工作记忆获得支持；预测适配的重新训练消融未达到 10% AULC 门槛，因此三模块联合因果条件失败。

## 稀疏执行

optional MAC 平均减少 **37.6%**，
但最大任务分数下降 **10.4pp**，超过 2pp 非劣界；结论为 **FAIL**。
profiler coverage 最低为 99.31%。所有三个专家均有真实调用，但 formal summary 未记录 capacity-drop 数。

## 结论与边界

`GATE-3` 只说明预注册矩阵、恢复、产物和诚实统计完成。`GATE-NN-MVP` 失败，
因此不生成 `network-mvp-v1` 正式 bundle，也不得使用“网络 MVP qualified”表述。
这些结果只适用于人工计算模型，不构成脑区等价或生物学结论。

协议缺口按 `DR-001`～`DR-003` 原样保留；不得用事后定义的 chance、阈值或替代指标反转本次 Gate。
