---
title: F1a 蘑菇体持续学习实验协议
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-04
---

# F1a：蘑菇体（MB）式稀疏扩展 + 关联读出的类增量持续学习

本协议在任何比较结果产生之前冻结，属于探索层，不构成 Gate 证据。若运行后修改，必须另存新版本并说明原因。

## 1. 回路来源（FlyWire 783，右半球）

- PN：`cell_class=ALPN` 且 `cell_sub_class=uniglomerular`，并且对右侧 KC 有突触输入；共 122 个，覆盖 55 个嗅小球。
- KC：`cell_class=Kenyon_Cell`、`side=right`，并且至少有 1 个 PN 输入；共 2,423 个。
- `W_pk[kc, pn]`：Zenodo `proofread_connections_783` 中 PN→KC 突触数在所有 neuropil 上的总和。

## 2. 任务

- 合成“气味”：每个类别的原型是 55 维嗅小球激活向量。每个嗅小球以 0.3 的概率激活，激活值服从 LogNormal(0, 0.5)。
- 样本：`g = relu(prototype · (1 + σ·ε₁) + σ·0.5·ε₂)`，ε 服从标准正态；同一嗅小球的 PN 共享 g，各 PN 再加 0.05 倍的独立噪声。
- 规模：100 个类别，每类训练 50 个、测试 50 个样本；分为 10 个任务，每个任务 10 类；单输出头的类增量设置，预测在已见过的全部类别中进行。
- 噪声 σ 的标定：只用 seed 999，只用 `pn_ncm` 的联合训练，选 σ 使测试准确率最接近 0.70，然后冻结。标定不使用任何其他模型。

## 3. 模型

| ID | 表示 | 读出与学习 |
|---|---|---|
| `mb_assoc` | KC = top-k(W_pk·x) 的二值码，k = 5%（APL 式 k-WTA） | 关联规则：只增强“活跃 KC → 正确类”的突触（`W[c] += z`），其余权重冻结 |
| `mb_perceptron` | 同上 | 感知机规则：预测错误时同时增强正确类、削弱错误类（作为部分冻结的消融） |
| `mb_sgd` | 同上 | 对全部读出权重做 softmax 交叉熵，Adam 优化 |
| `pn_ncm` | PN 输入 x | 类均值原型（nearest class mean，按余弦相似度） |
| `pn_sgd` | PN 输入 x | 线性 softmax，Adam 优化 |
| `mlp_sgd` | PN 输入 x | 一层隐藏层 MLP，H = 1,090，参数量与 `mb_assoc` 读出约相等，Adam 优化 |

- SGD 类模型统一使用 Adam，lr 1e-3，batch 32，每个任务 5 个 epoch。不调参，学习率网格只作为敏感性分析。
- 变体只作用于 `mb_assoc`：
  - 稀疏度 k ∈ {1%, 5%, 20%, 50%}；
  - 拓扑 ∈ {`real`, `real_binary`, `degree_preserving`, `er`}：
    - `degree_preserving`：二部图双边交换，交换次数为 10 × 边数；保留每个 KC 的入度、每个 PN 的出度，权重随边一起移动。
    - `er`：保持边数，均匀随机放置，权重从真实权重中置乱抽取。

## 4. 指标与统计

- 主指标：全部 10 个任务学完后，全部 100 类测试集上的最终平均准确率。
- 次指标：平均遗忘，即各任务学完时的准确率减去最终准确率，再对前 9 个任务取平均。
- 重复：seeds 0–9。每个 seed 同时决定原型、样本、零模型重连与模型初始化；同一 seed 下各条件配对比较。
- 统计：按 seed 配对差值，对 seed 做 10,000 次 bootstrap，取 95% percentile CI；双侧 p 值后在主比较 family 内做 Holm 校正。

## 5. 预注册主比较（Holm family）

| ID | 比较 | 预期 | 通过条件 |
|---|---|---|---|
| C1 | `mb_assoc(real,5%)` − `pn_sgd` | 正 | Δ ≥ 10pp，CI 下界 > 0，Holm p ≤ 0.05 |
| C2 | `mb_assoc(real,5%)` − `mb_sgd(real,5%)` | 正（部分冻结的作用） | Δ ≥ 10pp，CI 下界 > 0，Holm p ≤ 0.05 |
| C3 | `mb_assoc(real,5%)` − `pn_ncm` | 不确定（扩展 + 稀疏是否超出原型学习） | Δ ≥ 2pp，CI 下界 > 0，Holm p ≤ 0.05 才算支持 |
| C4 | `mb_assoc(real,5%)` − `mb_assoc(degree_preserving,5%)` | ≈0（合成气味下拓扑不应起作用） | 95% CI 落在 ±1pp 之内才判“等价”；否则报告为“未确定” |
| C5 | `mb_assoc(real,5%)` − `mb_assoc(real,50%)` | 正（稀疏的作用） | Δ ≥ 5pp，CI 下界 > 0，Holm p ≤ 0.05 |

`mlp_sgd`、`mb_perceptron`、k = 1%/20%、`real_binary` 与 `er` 仅作描述性报告。

## 6. 解释边界

- 合成气味没有真实的嗅小球相关结构，因此 C4 只能检验“在无结构输入下拓扑是否起作用”，不能回答真实输入统计下的问题；后者留给使用 DoOR 真实气味谱的 F1b，需另行批准下载。
- “部分冻结防遗忘”是算法层面的结论，不代表果蝇的实际学习机制。
