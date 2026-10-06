---
title: T1c 多情境无提示规则切换：可寻址的槽是否优于 GRU 的单一状态
status: FROZEN-BEFORE-RUN
change_request: CR-014
date: 2026-10-06
---

# T1c：需要同时保持多条规则时，工作记忆的槽结构是否带来优于 GRU 的收益

## 1. 依据

- T1b（`docs/transfer/T1b-report.md`）：读出增强写入 + surprise 门控的工作记忆（`enr_gate`）在单情境无提示切换任务上与 shared GRU 持平（非劣，但不优）；探针显示两者的状态都直接存着更新后的规则。任务只需要保持一个变量，槽结构没有用武之地。
- 工作记忆与 GRU 的结构差别在于：它有 4 个槽，按内容寻址读出，每个槽有自己的更新门，原则上可以把几个独立变量分别存放，只更新其中一个而不扰动其他；GRU 只有一个 64 维状态，每次更新都经过同一组门，需要自己学会划分子空间。
- T1c 检验这种结构差别在“同时保持多个独立变量、只更新被反馈涉及的那一个”的情形下是否转化为优势。

## 2. 任务：`multi-context-switch-v1`（`neuromorphic.transfer.multi_context`）

- 输入 14 维：事件 one-hot（cue、blank、query、feedback）、情境 one-hot（3 个）、规则 one-hot（只在 cue 中出现）、x0、x1、反馈位。
- 每个 episode 以 3 个 cue 开头（随机顺序），分别给出 3 个情境各自的规则（四条规则之一，彼此独立，可以相同）。之后每个 trial 随机抽一个情境（独立均匀）：0–2 个 blank → query（情境, x0, x1）（监督作答）→ feedback（情境, 正确答案）。feedback 带有情境标记，所以更新门可以只依据当前表示选择要写的槽。
- ID：18 个 trial；每个 episode 静默切换 2 次规则（第 5–15 个 trial，间隔 ≥ 3），切换的是该 trial 的情境（该 trial 即新规则下的第一个 query），其他情境不变，**没有任何 cue**；同一情境可能切换两次。OOD：30 个 trial，2–3 次切换（第 4–27 个 trial，间隔 ≥ 3），blank 0–4 个。
- split seeds（新）：train 51101、validation 52201、test 53301、OOD 54401、analysis 55501；样本量与 T1 相同（8,192 / 1,024 / 2,048 / 2,048 / 2,048）。平均长度 ID 约 57 步、OOD 约 123 步（T1 为约 37 步）。
- **理想观察者 oracle**：对每个情境分别运行 T1 的观察者（保留与该情境全部反馈一致的规则集合，出现矛盾时从该 trial 一致的规则重新开始），作答为集合内规则的平均。validation 上：切换前 1.000，切换当 trial 0.433，恢复期 0.752，总体 0.887；OOD（前 512 个样本）：恢复期 0.778，总体 0.901。每个 ID episode 平均有 3.6 个恢复期 query。

## 3. 模型（参数量在 ±6% 内匹配）

| 模型 | 核心 | 参数量 |
|---|---|---:|
| `cur_gate` | T1 的门控工作记忆（候选来自当前表示 + surprise 门控） | 28,746 |
| `enr_gate` | T1b 的读出增强写入 + surprise 门控 | 28,746 |
| `gru` | 主线 shared-GRU 基线（隐层 64） | 30,338（+5.5%） |

外部结构、surprise 定义（只在 feedback 步，|反馈位 − 网络自己在前一个 query 上答 1 的概率|，detach）与门控初值均与 T1/T1b 相同；工作记忆仍是 4 个槽、每槽 32 维。

## 4. 训练与选择（除预算外与 T1/T1b 相同）

- AdamW，weight decay 1e-2，batch 64，梯度裁剪 1.0，**10,000 次更新**（T1/T1b 为 5,000，见第 8 节）；每 250 次更新在 validation 上评估，保留 validation 准确率最高的检查点。
- 学习率网格 {1e-3, 3e-4}；每个（模型，seed）取 validation 准确率较高者。
- **model seeds 701–710**（新）。同一 seed 下各模型的训练样本顺序相同。
- test、OOD 与 analysis 只在最终选定的检查点上各评估一次。
- 脚本：`scripts/transfer_t1c_contexts.py`；结果写入 `artifacts/transfer/t1c_report.json`（git-ignored）。作业按“长作业优先、逐个分发”调度，只影响耗时。

## 5. 指标

- **恢复准确率**（主指标）：被切换情境在切换后第 1–4 次出现时的 query 准确率（按该情境自己的出现次数计）；
- 总体、切换前（该情境尚未切换的 query）、切换当 trial 的准确率；OOD 的总体与恢复准确率；
- 干扰窗口准确率（描述性）：另一情境切换后 1–3 个 trial 内、自身未切换的情境的 query 准确率。

## 6. 预注册比较

Holm family 为 H1–H3（superiority）。H4、H5 为非劣性检验（界值 0.02，主线惯例），二者同时成立才算“与 GRU 持平”（交并检验），不做多重校正。

| ID | 指标 | 比较 | 通过条件 |
|---|---|---|---|
| H1 | test 恢复准确率 | `enr_gate` − `gru`（superiority） | Δ ≥ 0.02，95% CI 下界 > 0，Holm p ≤ 0.05 |
| H2 | OOD 总体准确率 | `enr_gate` − `gru`（superiority） | Δ ≥ 0.02，同上 |
| H3 | test 恢复准确率 | `enr_gate` − `cur_gate`（T1b 的写入改进在多情境下是否仍成立） | Δ ≥ 0.03，同上 |
| H4 | test 恢复准确率 | `enr_gate` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |
| H5 | OOD 总体准确率 | `enr_gate` − `gru`（non-inferiority） | 95% CI 下界 > −0.02 |

阈值与 T1b 相同：superiority 与 non-inferiority 都用 ±2pp；H3 沿用 T1b H1 的 0.03。统计：每个 seed 的配对差，10,000 次百分位 bootstrap（RNG 20260715），`neuromorphic.transfer.statistics.paired_contrasts`。

### 6.1 总体裁决

| 结果 | 裁决 |
|---|---|
| H1 或 H2 SUPPORTED | 多情境下，槽结构的工作记忆**优于** GRU |
| H1、H2 都不支持，H4、H5 都 NON_INFERIOR | 持平延伸到多情境，**槽结构在此没有带来优势** |
| H1、H2 都不支持，H4 或 H5 不成立 | 多情境下 **GRU 更好**（在报告中指出差在恢复还是 OOD） |

## 7. 描述性分析（预注册，不做裁决）

1. 干扰：每个模型的（干扰窗口准确率 − 切换前准确率），test，均值与 bootstrap 95% CI。
2. **多情境规则探针**：analysis split 上取每个 feedback 步之后的状态（工作记忆 128 维槽值，GRU 64 维隐状态）；对每个情境 k 各拟合一个线性探针读出该情境的当前规则（前 1,024 个 episode 拟合、后 1,024 个评分；特征标准化，多项 logistic 回归，L2 1e-3，L-BFGS 200 次迭代）。按“该 feedback 是否关于情境 k”（addressed / held）与“情境 k 处于切换前，或已看到切换后第 1–4 次反馈”（pre / post）分组。参考水平为理想观察者该情境的一致规则集合内均匀猜一个的命中率。held_post 最直接地反映“更新了一个变量之后，其他时刻能否把它与别的变量分开保持”。报告 `enr_gate` − `gru`、`enr_gate` − `cur_gate` 在 held_post 与 addressed_post 上的均值与 CI。
3. 学会率（test 恢复 ≥ 0.70 的 seed 数）与选中 lr = 1e-3 的 seed 数。

## 8. 冻结前的设计检查（已披露）

第 6 节的阈值与比较在设计检查之前已写入脚本。之后只在非正式 seed 991 上运行过两次（`--pilot`，三个模型 × 两个学习率），只用 train/validation，没有访问 test、OOD 与 analysis；探针在设计检查中改用 validation。

1. **初版**（ID 中 75% 的 episode 切换 1 次，第 6–12 个 trial；5,000 次更新）。所有模型的恢复准确率都远低于理想观察者（validation 总体 / 恢复：`cur_gate` 0.939 / 0.651，`enr_gate` 0.940 / 0.652，`gru` 0.923 / 0.543；观察者恢复 0.777），并且 GRU 在 5,000 次更新时仍在上升（lr = 3e-4 时上升明显）。原因是切换信号被三个情境稀释：每个 episode 平均只有约 1.8 个恢复期 query。这样的比较会主要反映学习速度，而不是结构能学到什么。
2. **改动**（只为去掉预算限制，不针对任何模型）：ID 改为每个 episode 切换 2 次（第 5–15 个 trial，间隔 ≥ 3），预算改为 10,000 次更新。改后（同一 seed）：`cur_gate` 0.863 / 0.644，`enr_gate` 0.868 / 0.671，`gru` 0.882 / 0.759（观察者恢复 0.752）；所有模型的 validation 准确率都在预算内进入平台（选中检查点分别在第 9,250、10,000、5,750 次更新），预算不再是限制。单个 seed 的这些数值不构成结论；工作记忆的门控、写入与 T1b 完全相同，没有因为这次检查而改动。
3. 耗时：工作记忆每个作业约 900 s，GRU 约 390 s；正式运行 60 个作业、8 个进程，预计约 1.5 小时。
4. 第一次检查的 JSON 被第二次检查覆盖，上述第一次检查的数值来自运行时的记录。

## 9. 解释边界

- 这是 T track 的探索性检验，不构成任何 Gate 证据；不修改 P1–P5。`T1Network` 只新增了 `feedback_bit` 参数（默认值保持 T1/T1b 的行为）。
- 只检验 3 个情境、4 个槽这一种配置；槽数多于情境数，因此不是容量压力测试。结论不外推到情境数接近或超过槽数的情形。
- 没有“单槽”消融：若 H1 或 H2 成立，优势来自槽结构还是工作记忆的其他部分（如编码器、选择头的组合方式）需要另行检验。
- GRU 比工作记忆多 5.5% 的参数；按主线惯例视为参数量匹配。
