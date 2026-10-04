---
title: F1n 输入组成互换结果，以及事后找到的真正原因
status: EXPLORATORY
protocol: docs/fly/F1n-composition-swap-protocol.md
change_request: CR-013
date: 2026-10-05
---

# F1n 结果：差异不来自类型级组成；事后诊断发现，原因是 hemibrain 的递质预测把大多数 EPG/PEN 标成了抑制性

本报告属于探索层，不构成 Gate 证据。协议在正式运行之前已冻结并提交（`5c36c9b`）。第 3 节为事后分析，未经预注册。

## 1. 预注册结果（seeds 190–199；突触数权重；带活动下限；运行耗时 2,089 s）

资格检查通过（理想环 1.000）。组成匹配的最大误差为 1.1×10⁻¹⁶。

| arm | real | type_preserving | 优势 | bump 一致性（real / tp） | 锚定误差，弧度（real） |
|---|---:|---:|---:|---:|---:|
| FlyWire 接线 + 自己的组成 | 0.650 | 0.274 | +0.376 | 0.65 / 0.13 | 0.79 |
| FlyWire 接线 + hemibrain 组成 | 0.726 | 0.264 | +0.463 | 0.57 / 0.13 | 0.83 |
| hemibrain 接线 + 自己的组成 | 0.308 | 0.216 | +0.091 | 0.63 / 0.10 | 1.53 |
| hemibrain 接线 + FlyWire 组成 | 0.344 | 0.256 | +0.088 | 0.60 / 0.21 | 1.55 |

| ID | 比较 | 均值 Δ | 95% CI | Holm p | 裁决 |
|---|---|---:|---|---:|---|
| N1 | hemibrain 接线 + FlyWire 组成：real − type_preserving | +0.088 | [+0.011, +0.163] | 0.07 | NOT_SUPPORTED |
| N2 | FlyWire：优势（自己的）− 优势（hemibrain 组成） | −0.086 | [−0.196, +0.015] | 0.20 | NOT_SUPPORTED |
| N3 | hemibrain：优势（FlyWire 组成）− 优势（自己的） | −0.004 | [−0.119, +0.111] | 0.96 | NOT_SUPPORTED |

按预先写定的规则（N1、N2、N3 都不通过）：**差异来自神经元级的图，与类型级输入组成无关。** FlyWire 的优势在 hemibrain 的组成下不减反增；hemibrain 的优势在两种组成下都约为 0.09。

## 2. 关键的描述性线索

- hemibrain 的 real 图选中的状态**是** bump（一致性 0.63，与 FlyWire 的 0.65 相当；零模型只有 0.10–0.21），但锚定误差接近机会水平（1.53 弧度，FlyWire 为 0.79）。也就是说，hemibrain 形成了 bump，但 bump 的位置不由提示决定。

## 3. 事后诊断（未经预注册）

### 3.1 bump 被钉在一个位置上

对 F1n 中 real 图（自己的组成）选中的参数逐个提示重新模拟（评估噪声）：

| | FlyWire | hemibrain |
|---|---:|---:|
| 延迟期解码角度与提示角度的环形对齐（0–1） | 0.64 | **0.05** |
| 落点的有效位置数（最多 8） | 3.2 | **1.08**（10 个 seed 中 9 个正好是 1.0） |
| EPG 兴奋性输入强度的变异系数（逐神经元 / 逐提示 bin） | 0.40 / 0.30 | **0.87 / 0.77** |

在 hemibrain 上，无论提示在哪里，bump 最后都落在同一个位置。“记忆”得分 0.31 来自残余的模式差异，而不是 bump 的位置。

### 3.2 原因：递质符号

按 PB 小球列出原始突触数时，各小球 EPG 收到的 PEN、EPG、Delta7 输入都比较均匀（PEN→EPG 每个 EPG 356–632 个突触）。不均匀来自**符号**：hemibrain 的逐 body 递质预测（`hemibrain-v1.2-body-mean-neurotransmitters.feather` 中的 `predicted_nt`）把大多数兴奋性柱状神经元标成了谷氨酸，本项目把谷氨酸按抑制处理：

| 类型 | hemibrain 中按谷氨酸/GABA 计为抑制的数量 | FlyWire 中计为抑制的数量 | FlyWire 的实验注释（`known_nt`） |
|---|---:|---:|---|
| EPG | **38 / 46** | 0 / 47 | acetylcholine |
| EPGt | 4 / 4 | 0 / 4 | — |
| PEN_a | **10 / 20** | 0 / 20 | acetylcholine，glutamate-negative |
| PEN_b | **12 / 22** | 0 / 22 | — |
| PEG | **15 / 18** | 0 / 20 | — |
| Delta7 | 42 / 42 | 41 / 42 | glutamate |

EPG、PEN 在实验上是胆碱能的（Wolff et al. 2024 EASI-FISH；Turner-Evans et al. 2020 单细胞测序），FlyWire 的预测与之一致。hemibrain 的这些标签让环上一部分 EPG/PEN 变成抑制性的，环的兴奋性回路于是严重不均匀，bump 被钉在兴奋最强的那一段。

### 3.3 影响范围

- F1l 的 hemibrain 动力学（路径积分与记忆）以及 F1m、F1n 中所有 hemibrain arm，都使用了这些错误的符号。F1l 的结构检查（S1–S4）只用突触数，不受影响。
- 因此，“真实拓扑的记忆优势在 hemibrain 上不成立”这一结论（F1l 的 P1，F1m 的 M3、M4）**不能**再解读为数据集之间的接线差异；它很可能是递质标签造成的伪差异。
- 下一步（F1o）：给两个数据集都按细胞类型统一符号（取 FlyWire 中该类型的多数预测，与实验注释一致），预注册后重新检验 hemibrain 上的记忆优势。

## 4. 方法学教训

- 递质预测必须先按类型检查一致性，再用来定符号。同一类型内部符号不一致，在 FlyWire 上几乎没有（Delta7 有 1 个例外），在 hemibrain 的 CX 柱状神经元上却是多数。
- “数据集之间结果不同”时，应先排除注释层面的差异（递质、类型、侧别），再去解释接线。
