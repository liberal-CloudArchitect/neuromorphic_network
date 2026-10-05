---
title: F1p 用统一递质符号在 hemibrain 上重做路径积分
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-05
---

# F1p：递质符号统一后，hemibrain CX 能否在 LNO 或 PEN 驱动下左右相反地转动 bump

## 1. 动机

- F1l 在 hemibrain 上检验路径积分时，用的是逐 body 递质预测：38/46 个 EPG、10/20 个 PEN_a、12/22 个 PEN_b、15/18 个 PEG 被当作抑制性（F1n、F1o）。按细胞类型统一符号后，hemibrain 的记忆为 0.961，比 type_preserving 高 63pp（F1o）。
- 结构基础早已确认（F1l，只用突触数，不受符号影响）：PB 坐标下 PEN 偏移为 PEN_a 左 +69.5°、右 −66.2°，PEN_b 左 +65.7°、右 −65.1°，一致性 0.998–0.999。偏移方向相反、一致性 ≥ 0.9，所以偏移消融图有效（F1l 中 P3 失效只是因为 S2 的上限设为 67.5°）。

## 2. 模型与方法

- hemibrain 核心（EPG、EPGt、PEN_a、PEN_b、PEG、Delta7），**按细胞类型统一符号**（`consensus_signs`，取 FlyWire 多数：只有 Delta7 为抑制）；**突触数权重**（F1o 中记忆最好），`relu_tanh`，12 个类型参数。
- LNO 输入：hemibrain LNO 家族（GLNO、LNO1、LNO2、LNOa，左右各 6）；每个神经元取其类型在 FlyWire 中的多数符号（四类都是抑制，与实验注释一致）。
- 提示：每个 PB 相位上的 EPG 得到直接输入（8 个提示）；解码用 PB 相位做群体向量。时序与 F1l 相同。
- **两种速度驱动**（同一组候选、同一次记忆模拟）：
  - `lno`：某一侧全部 LNO 家族神经元活动为 a，经真实 LNO→核心连接输入（F1l 的方式）；
  - `pen`：直接给某一侧（PB 侧别）全部 PEN_a 与 PEN_b 输入 a（检验 PEN→EPG 偏移本身能否转动 bump）。
- 搜索：每个 seed 1,000 组候选（rng = 13000 + seed）：12 个类型参数（分布同 F1c-v4），log₁₀ g_cue ~ U(−1, 1)，log₁₀ a ~ U(log₁₀ 0.05, log₁₀ 5)。搜索噪声 950（记忆）/951（转动），评估噪声 200 + seed / 201 + seed。
- **活动下限**（F1m）：延迟期读出窗口与速度驱动读出窗口中，EPG 平均活动 ≥ 0.05 且空间标准差 ≥ 0.05。延迟期不满足的候选不参与选择；评估时记忆不满足记为 0.125，转动不满足则转动得分记为 0。
- **门控转动得分**：R = A × min(1, 平均 |位移| / 0.2 弧度)；联合得分 J = min(记忆, R)。每种驱动各自按 J 选择联合最优参数并评估。
- 图：real；`type_preserving` 与 `degree_preserving`（rng = 10000 + seed）；`offset_ablated`（`ablate_offsets`：每条 PEN→EPG 边的目标移回该组平均偏移，落到最近的 PB 相位上）。

## 3. 阳性对照与资格检查（seed 990）

理想转动环（全部突触数 × 5）加理想 LNO，两种驱动分别判定：联合最优参数下记忆 ≥ 0.5 且 J ≥ 0.5 才算该驱动有效；同时报告去掉偏移的理想环。设计检查（已披露）：冻结前只在 seed 991 上运行过对照：完整理想环在两种驱动下都是记忆 1.000、J +1.000（位移 0.74–0.75 弧度）；去掉偏移后 J −0.005。没有在 hemibrain 数据上运行动力学。

## 4. 预注册比较（seeds 210–219；Holm family 为 Q1–Q5 中有效的部分）

| ID | 驱动 | 比较 | 通过条件 |
|---|---|---|---|
| Q1 | `lno` | real J − 0 | 均值 ≥ 0.30，CI 下界 > 0，Holm p ≤ 0.05 |
| Q2 | `lno` | real J − `offset_ablated` J | Δ ≥ 0.20，同上 |
| Q3 | `lno` | real J − `type_preserving` J | Δ ≥ 0.20，同上 |
| Q4 | `pen` | real J − 0 | 均值 ≥ 0.30，同上 |
| Q5 | `pen` | real J − `offset_ablated` J | Δ ≥ 0.20，同上 |

描述性报告：

- `degree_preserving`；记忆最优参数下的记忆；未门控的 A、平均位移、有效候选比例。
- **转动方向是否与结构偏移一致**：左侧 PEN 的偏移为正、右侧为负。`pen` 驱动左侧时，预期 bump 正向移动（d_left > 0，d_right < 0）；`lno` 抑制左侧 PEN、使右侧占优时，预期方向相反（d_left < 0，d_right > 0）。

解释规则：Q1 或 Q4 通过，且相应的偏移消融比较（Q2 或 Q5）也通过，才判定“真实接线在该驱动下实现了依赖 PEN 偏移的路径积分”。

## 5. 解释边界

- 速度被抽象为单侧恒定输入，不涉及积分的定量精度（角速度与 bump 转速的线性关系）。
- 类型参数统一、PEN 没有依赖速度的增益调制；这些都是抽象假设。
