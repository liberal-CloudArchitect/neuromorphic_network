---
title: F1q PEN→EPG 映射的干净因果检验（hemibrain 路径积分）
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-05
---

# F1q：只改 PEN→EPG 这一块连接，转动会消失、保留还是反转

## 1. 动机

F1p 中，hemibrain CX（按类型统一递质符号）在 LNO 或 PEN 单侧驱动下左右相反地转动 bump（联合得分约 0.62），远高于 type_preserving。但偏移消融检验（Q2、Q5）没有通过：消融按 45° 取整，不干净（PEN_a 左侧被过度校正，其余组残留约 ±20°）；消融后的图仍然转动，方向在 9/10 个 seed 中反转；而 J 只看反对称、不看方向。

## 2. 三种只作用于 PEN→EPG 块的操作（其余连接完全不变，突触总数不变）

hemibrain 的 PB 相位是 45° 的整格，PEN 自身的相位来自它的 PB 小球标签，因此下面两种重映射都没有取整误差（冻结前已在 hemibrain 上核对：只看结构，未运行动力学）：

| 图 | 操作（`neuromorphic.fly.central_complex`） | 结构核对：PEN 偏移（PEN_a 左/右，PEN_b 左/右） |
|---|---|---|
| real | 无 | +69.5 / −66.2 / +65.7 / −65.1°（一致性 1.00） |
| `offset_zeroed` | `remap_offsets(mode="zero")`：每条 PEN→EPG 边移到与该 PEN 自身相位相同的 EPG 上 | 全部 0.0° |
| `offset_mirrored` | `remap_offsets(mode="mirror")`：目标相位变为 2 × 自身相位 − 原目标相位 | −69.5 / +66.2 / −65.7 / +65.1° |
| `pen_shuffled` | `shuffle_block`：只在 PEN→EPG 边之间交换目标（保留度数与每条边的突触数；rng = 15000 + seed） | 一致性降到 0.18–0.48 |
| type_preserving | 参照（rng = 10000 + seed） | — |

重映射后，落到同一对神经元上的边合并，突触总数不变。

## 3. 模型与方法（与 F1p 完全相同）

- 模型、搜索、下限、两种速度驱动都与 F1p 相同（`fly_f1p_pi_signs.Rotating`、`search_and_evaluate`）：hemibrain 核心按类型统一符号，突触数权重；LNO 输入按类型统一符号；1,000 组候选（rng = 13000 + seed）；搜索噪声 950/951，评估噪声 200 + seed / 201 + seed；延迟期与速度读出窗口的活动下限；门控转动得分 R = A × min(1, |位移| / 0.2)，J = min(记忆, R)。
- **选择不看方向**：联合最优参数仍按 J 选择；方向只在评估时测量，这样搜索不会偏向任何方向。
- **方向指标**：带符号位移 s = (d_left − d_right) / 2（正值表示左侧驱动让 bump 向相位增大的方向转）。
  - `dir`：J × sign(s)（J ≤ 0 时记为 0）。对 `lno` 驱动，预期方向为正：LNO→PEN 是严格对侧的，左侧 LNO 抑制右侧 PB 的 PEN，左侧 PEN（偏移为正）占优。
  - `mirror`：每个 seed 上，镜像图的 J × sign(s)，再乘以 −sign(real 的 J × sign(s))。正值表示镜像图朝 real 的**反方向**转；real 没有转动时记为 0。

## 4. 阳性对照与资格检查（seed 990）

理想转动环（全部突触数 × 5，理想 LNO），施加同样的三种操作中的 `zero` 与 `mirror`，两种驱动分别判定：

- `intact`：记忆 ≥ 0.5 且 J ≥ 0.5；
- `zeroed`：完整环有效，且去掉偏移后的 J 至少比完整环低 0.3；
- `mirrored`：完整环有效，镜像环 J ≥ 0.5，且转动方向与完整环相反。

每项比较只在对应的对照门通过时有效。

**设计检查（已披露）**：冻结前只在 seed 991 上运行过对照，三道门在两种驱动下都通过（完整环 J +1.000；去掉偏移后 −0.005；镜像后 +1.000/+0.999，方向相反）。同时发现：在理想环上，直接 PEN 驱动时 bump 朝**未驱动一侧**的偏移方向转（带符号位移 −0.75，与线性推理相反，可能是被驱动一侧 PEN 饱和、调制被抹平所致）。因此 PEN 驱动下的绝对方向取决于工作区间，**不对它预注册固定方向**；只对 `lno` 检验绝对方向，镜像检验则用相对方向（与 real 相反）。初稿中对 PEN 驱动的固定方向假设已据此删除。

## 5. 预注册比较（seeds 220–229；Holm family 为 R1–R7 中有效的部分）

| ID | 驱动 | 比较 | 通过条件 |
|---|---|---|---|
| R1 | `lno` | real 的 `dir` − 0（方向与结构偏移一致） | 均值 ≥ 0.30，CI 下界 > 0，Holm p ≤ 0.05 |
| R2 | `lno` | real J − `offset_zeroed` J | Δ ≥ 0.20，同上 |
| R3 | `lno` | real J − `pen_shuffled` J | Δ ≥ 0.20，同上 |
| R4 | `lno` | `mirror` − 0（镜像后反向转动） | 均值 ≥ 0.30，同上 |
| R5 | `pen` | real J − `offset_zeroed` J | Δ ≥ 0.20，同上 |
| R6 | `pen` | real J − `pen_shuffled` J | Δ ≥ 0.20，同上 |
| R7 | `pen` | `mirror` − 0 | 均值 ≥ 0.30，同上 |

解释规则：

- R2、R3（或 R5、R6）通过：转动依赖 PEN→EPG 的拓扑与偏移。
- R4（或 R7）通过：偏移的符号决定转动方向。
- 两者同时成立：PEN→EPG 偏移是路径积分的因果成分（在本抽象中）。

描述性报告：各图记忆最优参数下的记忆（预期 `offset_zeroed` 与 `pen_shuffled` 仍能保持一定记忆）；未门控的 A；方向为正的 seed 数；PEN 驱动下的绝对方向。

## 6. 解释边界

- 操作只改变 PEN→EPG 块；PEN 的输入（EPG→PEN、LNO→PEN）与其他连接不变。
- `offset_zeroed` 把一个 PEN 的输出集中到同一相位，会改变局部兴奋的分布，不只是“偏移变为 0”；`pen_shuffled` 用来补充这一点。
