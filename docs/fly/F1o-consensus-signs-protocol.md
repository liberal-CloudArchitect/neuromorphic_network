---
title: F1o 按细胞类型统一递质符号后，重测 hemibrain 上的 CX 记忆优势
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-05
---

# F1o：hemibrain 上记忆优势的缺失，是否来自错误的递质符号

## 1. 动机

F1n 的事后诊断（`F1n-report.md` §3）发现：hemibrain 逐 body 的递质预测把大多数兴奋性柱状神经元标成了谷氨酸，本项目按抑制处理：EPG 38/46、PEN_a 10/20、PEN_b 12/22、PEG 15/18、EPGt 4/4。FlyWire 的预测里这些类型全部是乙酰胆碱，FlyWire 的实验注释（`known_nt`）也是如此（EPG、PEN_a 为乙酰胆碱，PEN_a 为 glutamate-negative；Delta7 为谷氨酸）。在 hemibrain 上，bump 无论提示在哪里都落在同一个位置（落点有效位置数 1.08，提示对齐 0.05），而 FlyWire 上分别为 3.2 与 0.64。

## 2. 方法（F1m 的搜索与活动下限，原样沿用）

- **统一符号**（`consensus_signs`）：每个神经元取其细胞类型在 FlyWire 中的多数符号。结果：EPG、EPGt、PEN_a、PEN_b、PEG 全部为兴奋，Delta7 全部为抑制。冻结前已核对 hemibrain 与 FlyWire 上的符号分配，只看结构，未运行动力学。
- 任务、搜索、下限、噪声与 F1m 相同：`fly_f1m_floor.search_and_evaluate`；1,000 组候选（rng = 11000 + seed）；搜索噪声 950；评估噪声 300 + seed；活动下限为延迟期 EPG 平均活动 ≥ 0.05 且空间标准差 ≥ 0.05。
- 图：real、`type_preserving`、`degree_preserving`（rng = 10000 + seed）。重连只移动边，符号随神经元保留。

| arm | 数据 | 符号 | 权重 | 坐标 | 角色 |
|---|---|---|---|---|---|
| `hemibrain_consensus/counts` | hemibrain | 按类型统一 | 突触数 | 谱 | 主要 |
| `hemibrain_consensus/binary` | hemibrain | 按类型统一 | 二值 ≥ 17（边数匹配） | 谱 | 主要 |
| `hemibrain_consensus/pb_binary` | hemibrain | 按类型统一 | 二值 ≥ 17 | PB 相位 | 描述性 |
| `hemibrain_predicted/counts` | hemibrain | 原始预测 | 突触数 | 谱 | 对照（同一批 seed） |
| `flywire_consensus/counts` | FlyWire | 按类型统一（只改变 1 个 Delta7） | 突触数 | 谱 | 参照 |

- 描述性（只对 real）：bump 落点与提示的环形对齐，以及落点的有效位置数（对选中参数逐提示重新模拟，用评估噪声）。

## 3. 阳性对照（seed 990）

理想环（突触数 × 5），按带下限的规则评估，记忆 ≥ 0.5 才算通过。同样的代码与种子在 F1n 的设计检查（seed 991）中得到 1.000。

## 4. 预注册比较（seeds 200–209；Holm family 为 O1–O3）

| ID | 比较 | 通过条件 |
|---|---|---|
| O1 | `hemibrain_consensus/counts`：real − type_preserving | Δ ≥ 0.10，CI 下界 > 0，Holm p ≤ 0.05 |
| O2 | `hemibrain_consensus/binary`：real − type_preserving | 同上 |
| O3 | hemibrain 突触数：优势（统一符号）− 优势（原始预测） | 同上 |

解释规则（预先写定）：

- O3 通过，且 O1 或 O2 至少一项通过：hemibrain 上记忆优势的缺失来自递质标签；按类型统一符号后，真实拓扑的记忆优势在两个数据集上都成立，F1 总结中的原理恢复为“跨数据集成立（要求按类型统一递质符号）”。
- O3 通过但 O1、O2 都不通过：符号是部分原因，仍有其他差异。
- O3 不通过：递质标签不是主要原因，需要继续排查。

## 5. 解释边界

- 按类型统一符号是一个假设：它忽略了同一类型内部真实存在的递质差异（目前没有证据表明 CX 柱状神经元存在这种差异）。
- FlyWire 的预测被用作类型符号的来源，因为它与实验注释一致；这并不意味着 FlyWire 的逐神经元预测在所有类型上都正确。
