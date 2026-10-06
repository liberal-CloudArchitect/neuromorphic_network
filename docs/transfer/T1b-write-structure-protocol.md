---
title: T1b 工作记忆的写入结构 × surprise 门控（无提示规则切换）
status: FROZEN-BEFORE-RUN
change_request: CR-014
date: 2026-10-06
---

# T1b：主网络的工作记忆不如 GRU，是不是因为写入结构？

## 1. 依据

- T1（`docs/transfer/T1-report.md`）：surprise 门控让工作记忆的恢复准确率从 0.669 提高到 0.778，10/10 个 seed 都学会了；但 shared GRU 仍更好（恢复 0.836，OOD 总体 −2.9pp）。即使在 plain 也学会的 seed 上，工作记忆的恢复也只有约 0.79–0.80，而 GRU 为 0.83–0.84，所以差距不只是“学不学得会”，还有上限。
- 推测的原因：`working_memory.v1` 的候选写入 c_t = tanh(W_c r_t) 只依赖当前步的表示 r_t。在 feedback 步，r_t 只含反馈位，候选无法与上一步 query 的 (x0, x1) 或记忆中的旧规则结合；“新规则 = f(旧规则, x0, x1, 反馈)”无法在写入时算出并存回记忆，只能把零散的证据分别写入，再由读出端临时拼凑。GRU 的候选 n_t = tanh(W x_t + r ⊙ U h_{t−1}) 本来就结合了上一步的状态。
- T1b 检验两件事：（1）让候选写入看到本步读出（读—改—写），能否缩小或消除与 GRU 的差距；（2）写入改进之后，surprise 门控是否仍有独立贡献。

## 2. 任务与数据（与 T1 相同）

- 任务 `uncued-rule-switch-v1`（`neuromorphic.transfer.uncued_switch`），split seeds、样本量与 T1 完全相同：train 41101（8,192）、validation 42201（1,024）、test 43301（2,048）、OOD 44401（2,048）；analysis 45501（2,048）只用于第 7 节的描述性探针。
- 任务代码只新增两项读取功能：逐步的当前规则标签（`auxiliary_targets["rule"]`）和理想观察者在每一步之后的一致规则集合（`consistent_rules`，与 `oracle` 的更新完全相同）。随机数的使用顺序没有变化：五个 split 各抽查 293 个样本（序号 0, 7, 14, …），其输入、目标与 content hash 与 T1 提交（`21968c3`）逐一相同。
- 理想观察者水平（T1 报告）：test 恢复 0.803，OOD 总体 0.836。

## 3. 模型：写入来源 × surprise 门控（2×2）+ GRU

所有模型的外部结构与 T1 相同：adapter (11→64) → 核心 → `action_selector.v1` 的整合核心与二分类头。工作记忆核心（4 个槽、每槽 32 维）的单步方程：

- 读出：a_t = softmax(M_{t−1} q(r_t) / √32)，read_t = a_t M_{t−1}；送往选择头的是 e_t = r_t + P·read_t（四个单元相同）。
- 更新门：g_t = σ(W_g r_t + b_g [+ s_t·w])，方括号只在“有门控”的单元中出现。
- 候选：**current** 为 c_t = tanh(W_c r_t + b_c)（即 `working_memory.v1`）；**enriched** 为 c_t = tanh(W_c e_t + b_c)。
- 写入：M_t = M_{t−1} + g_t ⊙ (c_t − M_{t−1})。

| 模型 | 写入来源 | surprise 门控 | 与 T1 的关系 | 参数量 |
|---|---|---|---|---:|
| `cur_plain` | current | 无 | 等同 T1 `plain` | 28,554 |
| `cur_gate` | current | 有 | 等同 T1 `gate` | 28,554 |
| `enr_plain` | enriched | 无 | 新 | 28,554 |
| `enr_gate` | enriched | 有 | 新 | 28,554 |
| `gru` | — | — | 主线 shared-GRU 基线（隐层 64） | 30,146（+5.6%） |

- 写入来源只改变候选的输入，层与参数完全相同（`write_source`，单元测试核对：current 的候选与记忆内容无关，enriched 的候选随记忆内容变化）。
- 更新门在四个单元中都只看 r_t，门的时机只由 surprise 因素改变。这样两个因素是正交的：写入来源改变“写什么”，surprise 门控改变“何时写”。
- enriched 用的是本步从 M_{t−1} 读出的内容，严格因果（单元测试核对：改动未来输入不影响过去的输出）。
- surprise s_t = |反馈位 − p|，p 为网络自己在前一个 query 上答 1 的概率（detach），只在 feedback 步非零；门控权重 w 每槽一个、初值 2.0。均与 T1 相同。

## 4. 训练与选择（与 T1 相同）

- AdamW，weight decay 1e-2，batch 64，梯度裁剪 1.0，5,000 次更新；每 250 次更新在 validation 上评估，保留 validation 准确率最高的检查点。
- 学习率网格 {1e-3, 3e-4}；每个（模型，seed）取 validation 准确率较高者。
- **model seeds 601–610**（新；T1 用的是 501–510）。同一 seed 下各模型的训练样本顺序相同（配对设计）。
- test、OOD 与 analysis 只在最终选定的检查点上各评估一次。
- 脚本：`scripts/transfer_t1b_write.py`；结果写入 `artifacts/transfer/t1b_report.json`（git-ignored）。

## 5. 指标

- **恢复准确率**（主指标）：切换后第 1–4 个 trial 的 query 准确率；
- 总体、切换前、切换当 trial 的准确率；OOD 的总体与恢复准确率。

## 6. 预注册比较

Holm family 为 H1–H4（superiority）。H5、H6 为非劣性检验，界值 0.02 沿用主线的 2pp 惯例；二者要同时成立才算“追上 GRU”（交并检验），因此不做多重校正。

| ID | 指标 | 比较 | 通过条件 |
|---|---|---|---|
| H1 | test 恢复准确率 | `enr_gate` − `cur_gate`（有门控时，写入结构的作用） | Δ ≥ 0.03，95% CI 下界 > 0，Holm p ≤ 0.05 |
| H2 | test 恢复准确率 | `enr_plain` − `cur_plain`（无门控时，写入结构的作用） | Δ ≥ 0.05，同上 |
| H3 | test 恢复准确率 | `enr_gate` − `enr_plain`（写入改进后，门控是否仍有贡献） | Δ ≥ 0.03，同上 |
| H4 | OOD 总体准确率 | `enr_gate` − `gru`（superiority） | Δ ≥ 0.02，同上 |
| H5 | test 恢复准确率 | `enr_gate` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |
| H6 | OOD 总体准确率 | `enr_gate` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |

- 阈值来源：H1 的 0.03 约为 T1 中 `gate` 与 GRU 恢复差距（0.058）的一半；H2 沿用 T1 H1 的 0.05（plain 的 seed 间双峰使方差较大）；H3 沿用 T1 H2 的 0.03。H4 由 T1 的 0.03 改为 0.02，使 superiority 与 non-inferiority 用同一尺度（±2pp），并且 T1 中 GRU 的 OOD 总体（0.841）已高于理想观察者（0.836），3pp 的 superiority 几乎不可能达到、检验没有信息量。
- 统计：每个 seed 的配对差，10,000 次百分位 bootstrap（RNG 20260715），`neuromorphic.transfer.statistics.paired_contrasts`。

### 6.1 对“写入结构是不是原因”的总体裁决

| 结果 | 裁决 |
|---|---|
| H1 SUPPORTED，且 H5、H6 都 NON_INFERIOR | 写入结构是工作记忆不如 GRU 的**主要原因** |
| H1 SUPPORTED，但 H5 或 H6 不成立 | 写入结构是**部分原因**；报告第 7 节的差距缩小比例 |
| H1 NOT_SUPPORTED，且 H5、H6 不同时成立 | 就这种写入改进而言，写入结构**不是原因** |
| H1 NOT_SUPPORTED，但 H5、H6 都成立 | **无法归因**（在新 seed 上 `cur_gate` 与 GRU 的差距可能未复现，报告给出其描述性差值） |

H3 的解释：H3 SUPPORTED 表示写入改进后 surprise 门控仍有独立贡献；H3 不支持且 `enr_plain` 的学会率不低于 `cur_gate`，表示门控在 T1 中的作用主要是弥补写入结构的不足。

## 7. 描述性分析（预注册，不做裁决）

1. 交互作用：每个 seed 的 (`enr_gate` − `cur_gate`) − (`enr_plain` − `cur_plain`)，test 恢复与 OOD 总体，均值与 bootstrap 95% CI。
2. 差距缩小比例：(`enr_gate` − `cur_gate`) / (`gru` − `cur_gate`)，test 恢复与 OOD 总体，按 seed 均值计算。
3. 学会率：test 恢复 ≥ 0.70 的 seed 数（T1 中 plain 的 seed 分为约 0.54 与 ≥ 0.70 两群）；选中 lr = 1e-3 的 seed 数。
4. **规则可解码性探针**（直接检验“写入时是否结合了 query 与反馈”）：在 analysis split 上取每个 feedback 步之后的记忆状态（工作记忆为 4×32 个槽值展平成 128 维，GRU 为 64 维隐状态），标签为该 trial 的当前规则（4 类）。前 1,024 个 episode 拟合、后 1,024 个评分；特征标准化，多项 logistic 回归，L2 系数 1e-3，L-BFGS 200 次迭代。按切换前、切换当 trial、恢复期（切换后第 1–4 个 trial）分组报告。参考水平为“在理想观察者的一致规则集合中均匀猜一个”的命中率（不是严格上限：探针可以利用切换先验）。预期：若写入改进按假设起作用，恢复期的探针准确率 `enr_*` 高于 `cur_*`。报告 `enr_gate` − `cur_gate`、`enr_plain` − `cur_plain`、`enr_gate` − `gru` 的均值与 CI。

## 8. 冻结前的设计检查（已披露）

第 6 节的阈值与比较在设计检查之前已写入脚本。之后只在非正式 seed 996 上运行过一次（`--pilot`，五个模型 × 两个学习率），只用 train/validation，没有访问 test、OOD 与 analysis；探针在设计检查中改用 validation（前 512 个 episode 拟合、后 512 个评分）只为确认代码可运行。

- 目的：确认 enriched 写入在 T1 的预算与学习率网格下能正常训练（无发散、无 NaN），并估计耗时。
- 结果：所有单元都正常训练，**设计未做任何改动**。单个 seed 的 validation 值（总体 / 恢复）：`cur_plain` 0.889 / 0.789，`cur_gate` 0.873 / 0.753，`enr_plain` 0.807 / 0.560（停在未学会的一群），`enr_gate` 0.921 / 0.829，`gru` 0.921 / 0.839；恢复期的探针准确率 `cur_*` 0.35–0.50、`enr_plain` 0.07–0.20、`enr_gate` 0.78、`gru` 0.78（参考水平 0.73）。单个 seed 的这些数值不构成结论。
- 耗时：工作记忆每个作业约 405 s，GRU 约 100 s；正式运行 100 个作业、8 个进程，预计约 75 分钟。

## 9. 解释边界

- 这是 T track 的探索性检验，不构成任何 Gate 证据，也不修改 P1–P5 的模块、协议或锁；`write_source` 只加在 `neuromorphic.transfer` 的工作记忆副本上，主线的 `working_memory.v1` 不变。
- enriched 只是“跨步结合”的一种实现（借用已有的读出通路做读—改—写）；更新门仍只看 r_t，候选仍是单层。若写入结构的作用不被支持，并不排除“门也读记忆”或更深的候选网络有效。
- GRU 比工作记忆多 5.6% 的参数；比较按主线惯例视为参数量匹配。
- 新 seed 上的 `cur_plain`、`cur_gate` 是 T1 `plain`、`gate` 的重复，不重新检验 T1 的结论；与 T1 数值的对照只作描述。
