---
title: T1 预测误差门控的工作记忆更新（无提示规则切换）
status: FROZEN-BEFORE-RUN
change_request: CR-014
date: 2026-10-05
---

# T1：把“预测误差门控写入”迁到主网络的工作记忆上，能否加快无提示规则切换后的恢复

## 1. 依据

- F1b（`docs/fly/F1b-report.md`）：预测误差门控的可塑性（DAN = r − v）在无提示反转下比纯关联学习高 20.5pp；全局 surprise 调制没有额外收益。
- 主线：DRS 的规则切换有显式 cue，工作记忆不需要从错误中发现切换；P4 中预测的因果作用被否证，P5 改为用 surprise 调制路由。F1 的提示是：**预测误差应当门控写入**。

## 2. 任务：`uncued-rule-switch-v1`（`neuromorphic.transfer.uncued_switch`）

- 输入 11 维：事件 one-hot（cue、blank、query、feedback）、规则 one-hot（只在开头的 cue 中出现）、x0、x1、反馈位。
- 每个 episode 以 cue 开头，给出四条规则之一（x0、x1、xor、xnor）。每个 trial 为：0–2 个 blank → query (x0, x1)（监督作答）→ feedback（显示正确答案）。
- ID：12 个 trial；75% 的 episode 在第 4–8 个 trial 静默切换一次规则（换成另一条规则），**没有任何 cue**。OOD：20 个 trial，2–3 次静默切换（间隔 ≥ 3 个 trial），blank 0–4 个。
- split seeds（新）：train 41101、validation 42201、test 43301、OOD 44401、analysis 45501；train 8,192、validation 1,024、test 2,048、OOD 2,048 个样本。
- **理想观察者 oracle**：保留与反馈一致的规则集合，出现矛盾时从与该 trial 一致的规则重新开始，作答为集合内规则的平均。validation 上：切换前 1.000，切换当 trial 0.42，恢复期（切换后第 1–4 个 trial）0.80，总体 0.92。

## 3. 模型（参数量在 ±6% 内匹配）

所有模型都是：adapter (11→64) → 核心 → `action_selector.v1` 的整合核心与二分类头（LayerNorm → Linear → GELU → Linear）。

| 模型 | 核心 | 参数量 |
|---|---|---:|
| `plain` | `sensory_encoder.v1` + 与 `working_memory.v1` 等式完全相同的工作记忆（4 个槽，单元测试逐元素核对） | 28,554 |
| `gate` | 同上；surprise 只进入更新门：门 logit 加上 s·w（w 为每个槽可学习的权重，初值 2.0） | 28,554 |
| `input` | 同上；同样的 surprise 经可学习投影加到表示上（信息相同，但不专门门控） | 28,554 |
| `shuffled` | 同 `gate`，但 surprise 取自 batch 中另一个 episode | 28,554 |
| `gru` | 主线 shared-GRU 基线（隐层 64） | 30,146 |

**surprise** s_t = |反馈位 − p|，只在 feedback 步计算；p 是网络自己在前一个 query 上对“答 1”的概率（detach，严格因果）。

## 4. 训练与选择（所有模型相同）

- AdamW，weight decay 1e-2，batch 64，梯度裁剪 1.0，**5,000 次更新**；每 250 次更新在 validation 上评估，保留 validation 准确率最高的检查点。
- 学习率网格 {1e-3, 3e-4}；每个（模型，seed）取 validation 准确率较高者。
- model seeds：501–510（10 个）。test 与 OOD 只在最终选定的检查点上各评估一次。

## 5. 指标

- **恢复准确率**（主指标）：切换后第 1–4 个 trial 的 query 准确率（切换当 trial 无法预知，不计入）；
- 总体准确率、切换前准确率、切换当 trial 准确率；OOD 的总体与恢复准确率。

## 6. 预注册比较（Holm family 为 H1–H4）

| ID | 指标 | 比较 | 通过条件 |
|---|---|---|---|
| H1 | test 恢复准确率 | `gate` − `plain` | Δ ≥ 0.05，95% CI 下界 > 0，Holm p ≤ 0.05 |
| H2 | test 恢复准确率 | `gate` − `input`（门控优于同信息输入） | Δ ≥ 0.03，同上 |
| H3 | test 恢复准确率 | `gate` − `shuffled`（surprise 的特异性） | Δ ≥ 0.05，同上 |
| H4 | OOD 总体准确率 | `gate` − `gru`（相对主基线） | Δ ≥ 0.03，同上 |

统计：每个 seed 的配对差，10,000 次百分位 bootstrap（RNG 20260715，`neuromorphic.evaluation.statistics`）。

## 7. 冻结前的设计检查（已披露）

只在 seed 999 上、只用 train/validation 运行过两次（`--pilot`），没有访问 test 与 OOD：

1. 初版用线性输出头时，四个工作记忆变体连切换前的规则都保持不好（validation 0.76–0.82），而 GRU 达到 0.92。原因是线性头无法把记忆中的规则与当前 query 非线性地结合（例如 xor），这与主网络不符（主网络的 `action_selector.v1` 有 LayerNorm → Linear → GELU 的整合核心）。因此所有模型（包括 GRU）都改用这一整合头。
2. 改后 validation：`plain` 0.889（恢复 0.793），`gate` 0.878（0.767），`input` 0.883（0.751），`shuffled` 0.806（0.544），`gru` 0.919（0.839）。工作记忆变体在 3,000 次更新时仍在上升，因此预算改为 5,000 次。单个 seed 的这些数值不构成结论。

## 8. 解释边界

- 这是 T track 的探索性检验，不构成任何 Gate 证据，也不修改 P5 的模块或协议。
- 工作记忆沿用 `working_memory.v1` 的结构（候选只来自当前表示）；若 H1–H3 不通过，可能是因为这一结构本身难以跨步结合 query 与反馈，而不只是 surprise 门控无效。报告中会区分这两种解释。
