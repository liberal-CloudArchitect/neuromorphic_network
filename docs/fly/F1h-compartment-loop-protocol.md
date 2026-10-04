---
title: F1h 动态 compartment 读出协议（真实 MBON→DAN 回路计算教学信号）
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-04
---

# F1h：用真实 MBON→DAN 反馈计算各 compartment 的教学信号，与理想 r − v 比较

## 1. 动机与结构依据（FlyWire 783，右半球；只看结构）

- F1b：理想的预测误差门控（DAN = r − v）在无提示反转下优于纯关联学习 20.5pp。
- F1e：MBON→DAN 反馈中有 35% 落在互惠对上（7.3×）。
- MBON 效价按教导它的 DAN 家族确定（Aso 2014 框架）：PAM 教导的 21 个 MBON 记为“回避”（s = −1），PPL1 教导的 21 个记为“趋近”（s = +1），6 个未被教导的记为 s = 0。教导家族纯度的中位数为 0.98。
- 互惠反馈的符号：回避 MBON → 自己的 PAM DAN 以抑制为主（560 对 53 个突触）；趋近 MBON → 自己的 PPL1 DAN 以兴奋为主（257 对 104）。
- 读入：2,423 个嗅觉 KC 对 44 个 MBON 共有 22,304 对连接。

## 2. 任务

与 F1b 相同：20 种气味，σ = 0.4，真实 PN→KC 二值接线加 5% k-WTA；stable、volatile、noisy 三种日程各 2,000 个试次（每 200 个试次反转一半气味；noisy 的奖惩可靠性为 0.8）。

## 3. 读出架构（所有变体共用）

- C[m, k]：KC→MBON 二值连接（≥ 1 个突触）。可塑权重 W 初始化为 C。
- MBON 输出 o_m = W_m·z / k_active；基线 o⁰_m = C_m·z / k_active。
- 决策值 v = 2 · Σ_m s_m (o_m − o⁰_m) ÷ Σ_m \|s_m\| o⁰_m；v > 0 判为“接近”，判对标准同 F1b。
- 教学信号：d_m = Σ_d T̂[d, m]·DAN_d，其中 T̂ 是 DAN→MBON 突触数按 MBON 归一化（每个 MBON 的教导 DAN 权重之和为 1）。
- 可塑性（只作用于已有连接，即只有抑制性的多巴胺依赖压低，加上恢复项）：W ← clip(W − η·d·zᵀ + ρ·(C − W), 0, 2C)。
- 奖惩输入 ρ_d(r)：PAM 为 g·max(r, 0)，PPL1 为 g·max(−r, 0)，其余 DAN 为 0。

## 4. DAN 活动的四种变体

| ID | DAN_d |
|---|---|
| `none` | ρ_d(r)，没有反馈（纯关联学习） |
| `ideal` | PAM：g·relu(r − v)；PPL1：g·relu(v − r)；其余 0（对立的理想 RPE） |
| `loop_real` | relu(ρ_d(r) + g_fb · Σ_m F̂[m, d]·σ_m·(o_m − o⁰_m))，其中 F̂ 是真实 MBON→DAN 突触按 DAN 归一化（每个 DAN 的 MBON 输入之和为 1），σ_m 为 MBON 递质符号；g = 1 |
| `loop_rewired` | 同 `loop_real`，但 F 换成保度重连的结果（`rewire_feedback`，rng = 7000 + seed） |

## 5. 超参数标定（每个变体网格大小相同：36 组）

- 共同网格：η ∈ {0.03, 0.1, 0.3, 1.0}，ρ ∈ {0, 0.003, 0.03}。
- 第三维：`none` 与 `ideal` 取 g ∈ {0.3, 1, 3}；两个 `loop_*` 取 g_fb ∈ {1, 3, 10}。
- 只在 seeds 900–904 上标定，目标为三种日程的平均准确率；正式运行用 seeds 120–129。

## 6. 预注册比较（Holm family 只含 W1、W2；W3 为非劣性检验）

| ID | 日程 | 比较 | 通过条件 |
|---|---|---|---|
| W1 | volatile | `loop_real` − `none` | Δ ≥ 5pp，CI 下界 > 0，Holm p ≤ 0.05 |
| W2 | volatile | `loop_real` − `loop_rewired` | Δ ≥ 3pp，CI 下界 > 0，Holm p ≤ 0.05 |
| W3 | volatile | `loop_real` − `ideal` | 非劣：95% CI 下界 > −5pp |

描述性报告：stable 与 noisy 日程、反转后的恢复准确率，以及各变体选中的超参数。

## 7. 解释边界

- MBON 效价按教导家族推断，递质符号来自预测，可塑性只有压低加恢复；这些都是抽象假设。
- 若 W1 与 W3 都通过，说明真实反馈回路能近似实现 compartment 级的预测误差门控；若 W1 不通过，说明在这些假设下，真实回路的反馈符号不构成预测误差。
