---
title: F1m 加活动下限后，重测 FlyWire 与 hemibrain 上的 CX 记忆优势
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-04
---

# F1m：修正搜索后，真实 CX 拓扑的记忆优势在两个数据集上是否都成立

## 1. 动机

- F1c-v4（FlyWire，突触数权重）：real 0.659，比 type_preserving 高 38pp。F1l（hemibrain，二值权重，PB 坐标）：只高 7.4pp（CI 跨 0）。
- F1l 发现搜索会偏爱**退化解**：在 1,000 组候选的带噪得分里取最大值时，静默网络会被选中（real 的联合最优参数 6/10 是静默的）。F1m 的设计检查（seed 991，理想环，突触数权重）又发现了第二类退化解：**完全饱和**的网络（延迟期平均活动 0.99999994，空间标准差 6×10⁻⁸）在搜索噪声下得分 1.0，换一个评估噪声就跌到机会水平 0.125。
- 两个数据集之间还有一个尚未控制的差异：**突触数尺度不同**。两者的核心组成几乎相同（FlyWire 155 个神经元，hemibrain 152 个），但每条连接的突触数中位数，FlyWire 是 3，hemibrain 是 8。≥ 5 的二值阈值在 FlyWire 保留 2,716 条边，在 hemibrain 保留 6,184 条（密 2.3 倍）。按边数匹配的 hemibrain 阈值为 17（保留 2,655 条）。

## 2. 方法（与 F1c-v4 相同，只加下限、加数据集与权重形式）

- 任务：F1c-v4 的记忆任务。`simulate_memory`，200 ms 基线 → 300 ms 提示（幅度 1）→ 1,000 ms 延迟，读延迟期最后 200 ms 的 EPG；`relu_tanh`；模板解码，8 个提示，机会水平 0.125。
- 坐标：每个数据集用自己 real 图的**谱环角度**定义 8 个提示 bin（两个数据集用同一种坐标；hemibrain 上谱角度与 PB 相位的对齐为 0.968）。零模型沿用 real 的提示 bin。
- 搜索：每个 seed 1,000 组类型参数候选（rng = 11000 + seed，分布同 F1c-v4），同一 seed 内所有数据集、权重形式与图共用同一候选集；搜索噪声 950，每个提示 5 个试次；评估噪声 300 + seed，每个提示 10 个试次。
- **活动下限（有效性）**：延迟期读出窗口中，EPG 平均活动 ≥ 0.05，且 EPG 间空间标准差（逐试次计算后取平均）≥ 0.05。
  - 搜索：只在有效候选中取得分最高者；没有有效候选时，记忆记为机会水平 0.125。
  - 评估：选中的候选在评估噪声下也必须有效，否则记忆记为 0.125。
  - 同时对同一批候选应用 F1c-v4 的旧规则（不设下限，取最大值），用同一评估噪声报告，作为描述性对照。
- 图：real、`type_preserving`、`degree_preserving`（rng = 10000 + seed，同 F1c-v4）。

## 3. 条件（arm）

| arm | 数据集 | 权重 | 坐标 | 角色 |
|---|---|---|---|---|
| `flywire/counts` | FlyWire | 突触数（按中位输入和归一化） | 谱 | 主要（F1c-v4 的设置） |
| `flywire/binary` | FlyWire | 二值，≥ 5 | 谱 | 主要 |
| `hemibrain/counts` | hemibrain | 突触数 | 谱 | 主要 |
| `hemibrain/binary` | hemibrain | 二值，≥ 17（与 FlyWire 的边数匹配） | 谱 | 主要 |
| `hemibrain/binary5` | hemibrain | 二值，≥ 5（F1l 的阈值） | 谱 | 描述性 |
| `hemibrain/pb_binary` | hemibrain | 二值，≥ 17 | PB 相位 | 描述性 |

## 4. 阳性对照与资格检查（seed 990）

理想环（F1c-v3/v4 的 `ideal_ring_circuit`，全部突触数 × 5，使二值阈值不删除任何连接），在突触数与二值（≥ 5）两种权重下，按带下限的规则评估，记忆都 ≥ 0.5 才算通过；不通过则整个协议 INVALID。设计检查（已披露）：冻结前只在 seed 991 上运行过阳性对照：带下限时两种权重下都是 1.000；不设下限时，突触数权重选中了一个完全饱和的候选，评估为 0.125。没有在任何真实数据上运行。

## 5. 预注册比较（seeds 180–189；Holm family 为 M1–M4）

| ID | arm | 比较 | 通过条件 |
|---|---|---|---|
| M1 | `flywire/counts` | real − type_preserving | Δ ≥ 0.10，CI 下界 > 0，Holm p ≤ 0.05 |
| M2 | `flywire/binary` | real − type_preserving | 同上 |
| M3 | `hemibrain/counts` | real − type_preserving | 同上 |
| M4 | `hemibrain/binary`（≥ 17） | real − type_preserving | 同上 |

解释规则（预先写定）：

- 至少有一种权重形式在两个数据集上都通过（M1 与 M3，或 M2 与 M4）：判定真实拓扑的记忆优势**跨数据集成立**，恢复 F1 总结中的原理地位。
- FlyWire 通过而 hemibrain 在两种权重下都不通过：判定为**数据集特异**，原理维持降级。
- 两个数据集都不通过：判定 F1c-v4 的结果至少部分来自搜索的退化解。

描述性报告：`degree_preserving`；`hemibrain/binary5` 与 `hemibrain/pb_binary`；各 arm 有效候选的比例；旧规则（无下限）下的记忆与选中解的活动水平，以及旧规则选中退化解的次数。

## 6. 解释边界

- 下限阈值（0.05、0.05）是预先设定的启发式值，用于排除静默与均匀饱和两类退化状态，不代表生理上的活动范围。
- 密度匹配只匹配边数，不匹配权重分布的形状；两个数据集的重建与突触检测方法不同（FlyWire 自动检测、hemibrain 的 T-bar 检测）。
- 结论仍然只适用于“统一类型参数 + 速率模型”的抽象。
