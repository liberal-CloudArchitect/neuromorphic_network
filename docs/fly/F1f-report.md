---
title: F1f 路径积分结果
status: EXPLORATORY
protocol: docs/fly/F1f-path-integration-protocol.md
change_request: CR-013
date: 2026-10-04
---

# F1f 结果：只按“记忆”选出的参数，不会自发产生左右相反的转动

本报告属于探索层，不构成 Gate 证据。协议在实现和运行之前已冻结并提交（`31715ec`）。

## 1. 结果（seeds 90–99；沿用 F1c-v4 的类型参数；评估噪声 seed 为 100 + seed）

| 图 | 反对称指数 A（均值） | 平均转动量 \|Δ\|（弧度） |
|---|---:|---:|
| real | −0.33 | 0.39 |
| type_preserving | −0.33 | 0.11 |
| degree_preserving | −0.51 | 0.70 |

| ID | 比较 | 均值 Δ | 95% CI | Holm p | 裁决 |
|---|---|---:|---|---:|---|
| R1 | real A − 0 | −0.33 | [−0.65, +0.01] | 0.18 | NOT_SUPPORTED |
| R2 | real − type_preserving | 0.00 | [−0.45, +0.43] | 0.97 | NOT_SUPPORTED |
| R3 | real − degree_preserving | +0.18 | [−0.30, +0.64] | 0.92 | NOT_SUPPORTED |

## 2. 解释

- real 的 bump 在单侧 PEN 驱动下确实会移动（约 0.39 弧度），但左右驱动多数推向**同一方向**（A < 0）。这更像是“被驱动 PEN 的投射目标把 bump 拉过去”，而不是左右对称的转动。
- F1c-v4 只为“记忆”选择参数，选出的 PEN 增益很低（中位数约 1），PEN 在动力学中几乎不起作用。因此，“为记忆选出的参数会自发带来路径积分”这一点被否证。这并不否证真实 PEN→EPG 偏移投射能够支持转动，后者需要把转动也纳入参数选择目标。
- 后续：F1f-v2 用联合目标（记忆 + 转动反对称性）做等预算搜索，并加入一个带左右偏移 PEN 的理想环作为阳性对照。
