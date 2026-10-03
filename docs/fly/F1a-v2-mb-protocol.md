---
title: F1a-v2 蘑菇体归一化实验协议
status: FROZEN-BEFORE-RUN
change_request: CR-013
supersedes_question_of: docs/fly/F1a-report.md
date: 2026-10-04
---

# F1a-v2：KC 稳态归一化能否让稀疏码起作用

## 1. 动机与诊断

F1a 中，5% 稀疏 KC 码不如 50% 码，也不如直接对 PN 输入做 NCM。

先排除一个解释：按样本整体缩放输入，不会改变 top-k 的选择，所以“除以总活动”式的标量归一化不可能是原因。

我在标定 seed 999 上做了诊断，只看码的结构统计，没有看任何模型准确率：

| 编码 | 曾被激活的 KC | 最常胜出的前 5% KC 所占激活份额 | 类间平均余弦 |
|---|---:|---:|---:|
| 突触数，原始输入 | 2021 / 2423 | 0.20 | 0.38 |
| 每个 KC 的输入权重归一化为和 1 | 2423 / 2423 | 0.13 | 0.25 |
| 输入均值中心化 | 2231 / 2423 | 0.13 | 0.30 |

假说：输入权重总和大的 KC 对任何气味都容易胜出，导致码塌缩、类间重叠高。KC 的稳态可塑性会补偿输入强度差异（Apostolopoulou & Lin 2020, PNAS），可以抽象为“每个 KC 的输入权重归一化”。

## 2. 设计

- 回路、模型、训练、评估与统计方法同 F1a 协议。σ 保持 1.4 不变。
- 新 seeds：10–19，与 F1a 的 seeds 不重叠。
- 归一化（作用于 KC 驱动之前）：
  - `none`：同 F1a；
  - `kc_norm`：W 按行归一化，即每个 KC 的输入权重和为 1。**这是主条件**，依据第 1 节的码结构诊断事先选定；
  - `center`：每个样本的 PN 向量减去自身均值；
  - `kc_norm+center`：两者都用。
- 任务：
  - 主任务：同 F1a（family_mixing = 0）；
  - 次任务：相似气味，10 个家族，family_mixing = 0.6，家族跨任务分布（类 c 属于家族 c % 10）。

## 3. 预注册比较（Holm family）

| ID | 任务 | 比较 | 通过条件 |
|---|---|---|---|
| D1 | 主 | `mb_assoc(real,5%,kc_norm)` − `mb_assoc(real,5%,none)` | Δ ≥ 10pp，CI 下界 > 0，Holm p ≤ 0.05 |
| D2 | 主 | `mb_assoc(real,5%,kc_norm)` − `pn_ncm` | Δ ≥ 2pp，CI 下界 > 0，Holm p ≤ 0.05 |
| D3 | 主 | `mb_assoc(real,5%,kc_norm)` − `mb_assoc(real,50%,kc_norm)` | Δ ≥ 2pp，CI 下界 > 0，Holm p ≤ 0.05 |
| D4 | 主 | `mb_assoc(real,5%,kc_norm)` − `mb_assoc(degree_preserving,5%,kc_norm)` | 95% CI 落在 ±1pp 之内判“等价”，否则“未确定” |
| D5 | 次 | `mb_assoc(real,5%,kc_norm)` − `pn_ncm` | Δ ≥ 2pp，CI 下界 > 0，Holm p ≤ 0.05 |

描述性报告：`center`、`kc_norm+center`，稀疏度 1% 与 20%，`er` 与 `real_binary` 拓扑，以及两个任务上的全部 F1a 基线。

## 4. 解释边界

- D1 只检验“输入强度不均导致码塌缩”这一工程机制，不证明 KC 稳态可塑性在果蝇中以这种方式实现。
- 若 D2 与 D5 都不通过，就说明在合成气味上，MB 式表示不优于简单原型学习；F1a 中“学习规则才是关键”的结论保持不变。
