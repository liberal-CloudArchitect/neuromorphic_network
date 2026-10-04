---
title: F1m 活动下限与跨数据集记忆优势结果
status: EXPLORATORY
protocol: docs/fly/F1m-activity-floor-protocol.md
change_request: CR-013
date: 2026-10-05
---

# F1m 结果：修正搜索后，FlyWire 上的记忆优势稳健重复；hemibrain 上仍不成立

本报告属于探索层，不构成 Gate 证据。协议在正式运行之前已冻结并提交（`7f7174e`）；冻结前只在 seed 991 上运行过理想环对照。第 3 节中的成对差值为描述性分析，不属于 Holm family。

## 1. 资格检查（seed 990）

理想环（突触数 × 5）按带下限的规则评估：突触数权重 1.000，二值权重 1.000，均通过。

## 2. 预注册结果（seeds 180–189；运行耗时 5,176 s）

| arm | real | type_preserving | degree_preserving | 有效候选比例（real） | 延迟期 EPG 平均活动（real） |
|---|---:|---:|---:|---:|---:|
| `flywire/counts` | **0.635** | 0.226 | 0.370 | 0.55 | 0.26 |
| `flywire/binary`（≥ 5） | **0.641** | 0.362 | 0.415 | 0.51 | 0.28 |
| `hemibrain/counts` | 0.334 | 0.309 | 0.290 | 0.15 | 0.13 |
| `hemibrain/binary`（≥ 17，边数匹配） | 0.532 | 0.484 | 0.509 | 0.16 | 0.12 |
| `hemibrain/binary5`（≥ 5，描述性） | 0.329 | 0.346 | 0.265 | 0.14 | 0.14 |
| `hemibrain/pb_binary`（≥ 17，PB 相位，描述性） | 0.541 | 0.440 | 0.502 | 0.16 | 0.11 |

记忆准确率的机会水平为 0.125。

| ID | arm | real − type_preserving | 95% CI | Holm p | 裁决 |
|---|---|---:|---|---:|---|
| M1 | `flywire/counts` | **+0.409** | [+0.319, +0.485] | 0.0004 | **SUPPORTED** |
| M2 | `flywire/binary` | **+0.279** | [+0.162, +0.389] | 0.0004 | **SUPPORTED** |
| M3 | `hemibrain/counts` | +0.025 | [−0.092, +0.130] | 0.66 | NOT_SUPPORTED |
| M4 | `hemibrain/binary`（≥ 17） | +0.049 | [−0.039, +0.135] | 0.54 | NOT_SUPPORTED |

按预先写定的解释规则：**FlyWire 在两种权重下都通过，hemibrain 在两种权重下都不通过，判定为数据集特异。**

## 3. 描述性结果

### 3.1 活动下限的作用

| arm | real − type_preserving（带下限） | 同一批候选、旧规则（无下限） |
|---|---:|---:|
| `flywire/counts` | +0.409 | +0.380 [+0.305, +0.434] |
| `flywire/binary` | +0.279 | +0.279 [+0.150, +0.399] |
| `hemibrain/counts` | +0.025 | +0.014 [−0.107, +0.129] |
| `hemibrain/binary` | +0.049 | +0.054 [−0.037, +0.145] |

- 旧规则选中静默或均匀饱和候选的次数，FlyWire real 为 0/10（突触数）与 1/10（二值）；type_preserving 为 4/10 与 1/10；`hemibrain/binary5` 的 real 为 4/10，degree_preserving 为 5/10。
- 下限主要修正的是零模型与 hemibrain 的退化解，**没有改变任何一项结论**。F1c-v4 在 FlyWire 上的结果不是退化解造成的：在新的 seeds、新的候选集和两种权重下都稳健重复，效应甚至更大（+0.41 对 +0.38）。
- 带下限的规则只有一次在评估时失效（`flywire/counts/degree_preserving` 的一个 seed，记为机会水平）。

### 3.2 hemibrain 上的其他比较

| 比较 | 均值 Δ | 95% CI |
|---|---:|---|
| `hemibrain/pb_binary`：real − type_preserving | +0.101 | [+0.031, +0.169] |
| `hemibrain/pb_binary`：real − degree_preserving | +0.039 | [−0.078, +0.141] |
| `hemibrain/binary`：real − degree_preserving | +0.024 | [−0.074, +0.121] |
| `hemibrain/binary5`：real − type_preserving | −0.017 | [−0.165, +0.134] |
| `flywire/counts`：real − degree_preserving | +0.265 | [+0.083, +0.426] |
| `flywire/binary`：real − degree_preserving | +0.226 | [+0.149, +0.300] |

- **边数匹配提高了 hemibrain 上所有图的记忆**（二值 ≥ 17 时 real 为 0.53，≥ 5 时为 0.33），但零模型提高得一样多，真实拓扑的优势仍然没有出现。F1l 中 hemibrain 记忆低的一部分原因是二值阈值造成的密度差异，但这不能解释“没有拓扑优势”。
- 用 PB 相位定义提示时，real 比 type_preserving 高约 10pp（CI 不含 0），但不高于 degree_preserving。这是唯一一个在 hemibrain 上偏向真实拓扑的迹象，属于描述性结果，强度也远小于 FlyWire。
- hemibrain 上的有效候选只有 9–16%（FlyWire real 为 51–55%），延迟期活动也低一半。同一个类型参数分布，在 hemibrain 上更少落在“有活动、有空间结构”的区间。

## 4. 结论

1. **修正后的搜索确认了 F1c-v4：** FlyWire 上，真实 CX 拓扑的记忆优势在两种权重下都稳健（+41pp 与 +28pp，相对 degree_preserving 也有 +23–27pp），并且不是退化解造成的。
2. **这一优势在 hemibrain 上不成立**（M3、M4），换用边数匹配的阈值或 PB 坐标也只出现很弱的迹象。按预注册的规则，该原理维持降级：**“在 FlyWire 的这一抽象中成立，跨数据集未重复”**。
3. 可能的原因（均未检验）：两个数据集的突触检测与重建方法不同，导致权重分布的形状不同（hemibrain 连接更密、更均匀，突触数中位数 8 对 3）；同一套类型参数分布对两个数据集并不同样合适（hemibrain 的有效候选少得多）；或者 FlyWire 的特定接线细节恰好有利于这个抽象中的吸引子。要分清这些原因，需要按数据集分别调整参数分布，或者比较两个数据集中相互匹配的单个神经元的连接。
4. 方法学：活动下限必须同时限制平均活动与空间结构，否则静默与均匀饱和两类退化解都会混入（设计检查中，饱和解在搜索噪声下得分 1.0，换一个噪声就跌到 0.125）。今后所有搜索类协议都应沿用这条规则。
