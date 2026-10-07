---
title: T1d 按内容寻址的 surprise 门控（多情境与单情境）
status: FROZEN-BEFORE-RUN
change_request: CR-014
date: 2026-10-07
---

# T1d：让 surprise 只打开刚刚读过的槽，能否修复多情境下的更新缺陷

## 1. 依据

- T1c（`docs/transfer/T1c-report.md`）：三个情境交错、单情境静默切换时，读出增强 + surprise 门控的工作记忆（`enr_gate`）比 shared GRU 恢复低 3.6pp、OOD 低 2.2pp。它对未切换情境的干扰更小（−0.4pp 对 −1.3pp），但被切换情境的新规则在状态中读不出来（0.37 对 0.52）。
- 事后推测（T1c 第 4 节）：surprise 以每槽一个权重加到**所有**槽的门 logit 上，g_t = σ(W_g r_t + b + s_t·w)，只决定“写不写”，不决定“写哪个槽”。多情境中一次意外的反馈只关于一个情境，全局门控迫使模型在“改写其他槽”与“压低对 surprise 的响应”之间权衡。
- CR-015（T-Gate 草案）的问题 0：接入完整网络之前，先检验按内容寻址的门控，再决定 `working_memory.v2` 的门控方程。用户已决定先做 T1d。

## 2. 门控方程

工作记忆其余部分与 T1b/T1c 的 `enr_gate` 完全相同（4 个槽 × 32 维，读出 a_t = softmax(M_{t−1} q(r_t)/√32)，候选 c_t = tanh(W_c (r_t + P·read_t))）。只改 surprise 偏置落在哪些槽上：

| 模型 | 门 logit | 说明 |
|---|---|---|
| `enr_gate` | W_g r_t + b + s_t·w | T1b/T1c 的全局门控（对照） |
| `enr_addr` | W_g r_t + b + s_t·w ⊙ (S·a_t) | surprise 按本步读出注意力分配，S = 4 为槽数 |
| `gru` | — | shared GRU 主基线 |

- 注意力均匀（a_t = 1/S）时两者完全相同，所以 `enr_gate` 就是“均匀寻址”的对照；总的 surprise 预算（Σ_i S·a_t,i = S）相同，参数完全相同（多情境 28,746，单情境 28,554；GRU 30,338 / 30,146）。
- 读出注意力由本步表示 r_t 产生：在多情境任务中 feedback 行带有情境标记，所以注意力可以指向存放该情境规则的槽，“读哪个槽就写哪个槽”。
- 单元测试核对：注意力均匀时 `enr_addr` 与 `enr_gate` 逐元素相同；注意力集中于某槽时，surprise 只主要打开该槽；网络严格因果；参数量不变。

## 3. 两部分

| 部分 | 任务 | 预算与选择 | 作用 |
|---|---|---|---|
| A `multi` | T1c 的 `multi-context-switch-v1`（split seeds 51101/52201/53301/54401/55501，每个 episode 2 次切换） | 10,000 次更新（同 T1c） | 主检验：寻址能否修复多情境缺陷 |
| B `single` | T1/T1b 的 `uncued-rule-switch-v1`（split seeds 41101/42201/43301/44401/45501） | 5,000 次更新（同 T1b） | 安全检查：寻址是否损害单情境 |

两部分都与原协议相同：AdamW，weight decay 1e-2，batch 64，梯度裁剪 1.0；每 250 次更新在 validation 上评估并保留最佳检查点；学习率网格 {1e-3, 3e-4}，每个（部分，模型，seed）取 validation 准确率较高者。**model seeds 901–910**（新）。评估函数（`measure`、`rule_probe`、`oracle_levels`、`describe`）直接从 T1c 与 T1b 的脚本导入，未作任何改动。脚本：`scripts/transfer_t1d_addressed.py`；结果写入 `artifacts/transfer/t1d_report.json`（git-ignored）。

## 4. 指标

与 T1c（部分 A）和 T1b（部分 B）相同：主指标为 test 恢复准确率（切换后第 1–4 次出现），另有总体、切换前、OOD 总体与恢复；部分 A 另有干扰窗口准确率与多情境规则探针。

## 5. 预注册比较

Holm family 为 H1、H2（superiority）。H3–H6 为非劣性检验（界值 0.02）。

| ID | 部分 | 指标 | 比较 | 通过条件 |
|---|---|---|---|---|
| H1 | A | test 恢复 | `enr_addr` − `enr_gate`（寻址的作用） | Δ ≥ 0.03，95% CI 下界 > 0，Holm p ≤ 0.05 |
| H2 | A | test 恢复 | `enr_addr` − `gru`（superiority） | Δ ≥ 0.02，同上 |
| H3 | A | test 恢复 | `enr_addr` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |
| H4 | A | OOD 总体 | `enr_addr` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |
| H5 | B | test 恢复 | `enr_addr` − `enr_gate`（寻址不损害单情境） | 95% CI 下界 > −0.02 |
| H6 | B | OOD 总体 | `enr_addr` − `gru`（单情境与 GRU 持平） | 95% CI 下界 > −0.02 |

统计：每个 seed 的配对差，10,000 次百分位 bootstrap（RNG 20260715），`neuromorphic.transfer.statistics.paired_contrasts`。

### 5.1 总体裁决与对 CR-015 的建议

| 结果 | 裁决 |
|---|---|
| H1 SUPPORTED，且 H3、H4 都 NON_INFERIOR | 按内容寻址**修复**了多情境缺陷 |
| H1 SUPPORTED，但 H3 或 H4 不成立 | **部分改善** |
| H1 NOT_SUPPORTED | 就这种寻址而言**无效**；T1c 的事后推测不被支持 |

单情境安全：H5、H6 都成立。若“修复”且“安全”，建议 CR-015 的 `working_memory.v2` 采用按读出注意力寻址的门控；若“修复”但不“安全”，在报告中给出权衡，由用户决定；其他情况保留 T1b 的全局门控，并在 CR-015 中把多情境列为已知缺陷。

## 6. 描述性分析（预注册，不做裁决）

1. 部分 A 的干扰（干扰窗口准确率 − 切换前准确率）与多情境规则探针（`enr_addr` − `gru`、`enr_addr` − `enr_gate` 在 held_post 与 addressed_post 上的差）。
2. **寻址诊断**（部分 A，`enr_gate` 与 `enr_addr`）：analysis 前 1,024 个 episode 的 feedback 步上，读出注意力的平均峰值，以及“每个情境在同一 episode 内是否总是读同一个槽”的一致性。一致性只有与峰值一起看才有意义（近乎均匀的注意力也可能有稳定的 argmax）。
3. 学会率（test 恢复 ≥ 0.70 的 seed 数）。

## 7. 冻结前的设计检查（已披露）

第 5 节的比较与阈值在设计检查之前已写入脚本。之后只在非正式 seed 996 上运行过一次（`--pilot`，两部分 × 三个模型 × 两个学习率），只用 train/validation，没有访问 test、OOD 与 analysis；探针与寻址诊断在设计检查中改用 validation。

- 目的：确认 `enr_addr` 在两种任务的原预算下能正常训练，并估计耗时。
- 结果：所有单元都正常训练，**设计未做任何改动**。单个 seed 的 validation 值（总体 / 恢复）：部分 A：`enr_gate` 0.867 / 0.660，`enr_addr` 0.885 / 0.761，`gru` 0.884 / 0.765；部分 B：`enr_gate` 0.921 / 0.829，`enr_addr` 0.920 / 0.830，`gru` 0.921 / 0.839。寻址诊断：两种工作记忆在 feedback 步的读出注意力都已高度集中且按情境一致（峰值约 0.98，一致性约 1.0），所以两者的差别只在 surprise 偏置落在哪些槽上。单个 seed 的这些数值不构成结论。
- 耗时：部分 A 的工作记忆作业约 1,225 s，部分 B 约 515 s，GRU 约 460 s / 190 s；正式运行 120 个作业、9 个进程，预计约 2.5 小时。

## 8. 解释边界

- T track 的探索性检验，不构成 Gate 证据；不修改 P1–P5。`surprise_wm` 只新增 `gate_address`（默认 `global`，保持 T1–T1c 的行为）与仅供分析的注意力记录。
- 只检验一种寻址方式（按本步读出注意力）；结论不外推到其他寻址方式（例如按更新门自身的分布寻址）。
- 部分 B 只是安全检查，不重新检验 T1b 的结论。
