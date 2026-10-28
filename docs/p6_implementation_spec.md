---
title: P6（T-Gate）读出增强写入 + 预测误差门控工作记忆的实施规格
status: ACCEPTED
phase: P6
protocol: p6-protocol-v1
change_request: CR-015
last_updated: 2026-10-29
---

# P6（T-Gate）实施规格

**本规格随 CR-015 于 2026-10-29 批准并冻结**，所有原先待定的项目都已由用户决定（见 CR-015“用户决定”一节）。实现与运行必须严格遵守本文。发现错误时写勘误或新的 CR，不得直接编辑本文；正式 seed 的结果不得用来回头修改阈值、seed 或 preset。

## 1. 假说

1. 工作记忆的候选写入来自读出增强后的表示，使一次写入能把当前观察与已存内容结合（T1b：写入结构决定上限）。
2. 预测误差门控工作记忆的写入，使网络可靠地学会从结果中发现无提示的变化（T1：门控决定可靠性）。
3. 二者接入完整模块化网络后，在无提示切换任务上相对 `working_memory.v1` 有因果收益，并且不损害 AR、DRS、SmallGraph。

## 2. 冻结架构

- 新模块 `working_memory.v2`（state `working-state-v2`）：读出同 v1；c_t = tanh(W_c (r_t + P·read_t) + b)；g_t = σ(W_g r_t + b + s_t·w)，w ∈ R^4 初值 2.0；s_t 为 `predictive_adapter.v3` 的 detached surprise（与路由所用相同），在 padding 与 episode 边界为 0。
- 组合器 `ModularBrainNetworkV4`：V3 的图，`working_memory.v1` → `working_memory.v2`；其余模块、路由预算（dual-route ≤ 25%、drops = 0、AR episodic 与 DRS working 语义覆盖 100%）不变。
- 新增语义覆盖：uncued-rule-switch 的所有有效 step 强制执行 working（与 DRS 相同），使反馈步一定进入工作记忆。
- 单元测试必须逐元素核对：v2 在 s_t ≡ 0 且候选来源切回 r_t 时与 v1 一致；v2 的单步方程与 T1b `enr_gate` 一致。

## 3. 数据与随机性

| 任务 | 版本 | 说明 |
|---|---|---|
| Associative Recall | `associative-recall-p6-v1` | P6 namespace，生成规则同 P5 |
| Delayed Rule Switch | `delayed-rule-switch-p6-v1` | 有显式切换 cue（检验非退化） |
| SmallGraph | `small-graph-p6-v1` | 同 P5 |
| **Uncued Rule Switch** | `uncued-rule-switch-p6-v1` | 主线接口改写的 T1 任务：cue 给出初始规则，每个 trial 为 delay → query（监督）→ feedback（正确答案），规则静默切换；ID 12 个 trial、OOD 20 个 trial 2–3 次切换（同 T1） |

| split | P6 seed | 用途 |
|---|---:|---|
| train | 31101 | 参数更新 |
| validation | 32201 | early stop 与 pilot 选择 |
| analysis | 35501 | AULC 与顺序训练分析 |
| test | 33301 | 冻结 checkpoint 的 ID 评估 |
| OOD | 34401 | 冻结 checkpoint 的 OOD 评估 |

正式训练 seeds `[17, 29, 43, 59, 71]`；seed `7` 只用于 qualification 与 pilot。样本量同 P5：train 8192、validation 2048、analysis 512、test 2048、OOD 2048。

## 4. 运行阶段

| 阶段 | 内容 | 通过条件 |
|---|---|---|
| qualification | seed 7，缩小的数据与步数；四任务 forward/backward 有限；v2 的门控与候选都获得非零梯度；语义覆盖、dual 预算、drops = 0；uncued-rule-switch 反馈步上 forecast error 与结果 surprise 的 Spearman 相关 ≥ 0.5；报告反馈步与非反馈步的 s_t 分布（门权重初值 2.0 按 s ∈ [0, 1] 标定） | 全部满足才生成 qualification lock |
| pilot | seed 7，只访问 train/validation；候选为 P5 的 `preset-0`–`preset-3` 加 `preset-4`（lr 1e-3，temporal 0.10，semantic 0.01，即 `preset-1` 只改学习率），沿用 P5 的 eligibility 规则，按 validation macro AULC 选一个 | 生成绑定 clean SHA 的 pilot lock |
| mechanism | 每 seed 6 个 cell（4 个重新训练、2 个推理干预），五个 seed 共 30 个（见第 5 节） | 第 6 节 mechanism 门槛全部通过才生成 mechanism lock |
| full | mechanism 的 full cell 加主基线 shared GRU 与 shared Transformer（四任务共享训练） | 第 6 节 full 门槛 |

阶段只能按 qualification lock → pilot lock → mechanism lock → full 的顺序启动；不得在正式运行中重新选择 preset、seed 或阈值。

所有重新训练的 cell 与基线使用同一训练设置：batch 64，每任务至多 5,000 次更新，每 100 次更新做一次 validation，patience 30（3,000 次更新没有改善才早停），保留 validation 最优的 `best.pt`。

## 5. mechanism 矩阵（每 seed）

| cell | 类型 | 干预 |
|---|---|---|
| `full` | 重新训练 | V4（v2：读出增强写入 + 门控） |
| `wm-v1` | 重新训练 | V3（v1：当前表示写入、无门控），即现有主网络 |
| `gate-off` | 重新训练 | v2 去掉 s_t·w（只有读出增强写入） |
| `enrich-off` | 重新训练 | v2 的候选改回来自 r_t（只有门控） |
| `shuffle-surprise` | 同 seed full checkpoint | 推理时 s_t 取自 batch 中另一个 episode |
| `surprise-zero` | 同 seed full checkpoint | 推理时工作记忆门控的 s_t 置 0（路由仍用原 surprise） |

## 6. 门槛

统计沿用主线：10,000 次 seed→stratum/sample 严格配对分层 bootstrap，RNG 20260715，95% percentile CI，Holm p ≤ 0.05。uncued-rule-switch 的 strata 为“切换 / 不切换 × delay bucket”。

**mechanism（Holm family M1–M4）**

| ID | 指标 | 比较 | 门槛 |
|---|---|---|---|
| M1 | uncued-rule-switch 恢复准确率（切换后第 1–4 个 trial） | `full` − `wm-v1` | ≥ 5pp |
| M2 | 同上 | `full` − `gate-off` | ≥ 3pp |
| M3 | 同上 | `full` − `enrich-off` | ≥ 3pp |
| M4 | 同上 | `full` − `shuffle-surprise` | ≥ 3pp |
| M5 | AR、DRS、SmallGraph 各自的最终分数 | `full` − `wm-v1` | 非劣：每个任务与 seed 的下降都不超过 2pp |

`surprise-zero` 只作描述（区分“门控在推理时起作用”与“门控只在训练中起作用”）。

**full（主检验）**

| ID | 指标 | 比较 | 门槛 |
|---|---|---|---|
| F1 | uncued-rule-switch 恢复准确率 | `full` 对 shared GRU、shared Transformer | 非劣，界值 2pp |

**full（次要检验，不决定 Gate）**

| ID | 指标 | 比较 | 门槛 | Holm family |
|---|---|---|---|---|
| F2 | 四任务 chance-normalized ID macro | `full` 对 shared GRU、shared Transformer | 非劣，界值 2pp | S1（F2、F3） |
| F3 | 四任务 OOD | 同上 | 非劣，界值 2pp | S1（F2、F3） |
| F4 | uncued-rule-switch 恢复准确率 | 同上 | superiority ≥ 3pp | S2（F4） |

`GATE-6-WM` 通过 = M1–M5 全部通过且 F1 非劣。F2–F4 照常计算并报告，不论结果如何都不改变 Gate 裁决。`GATE-6-WM` 只裁决 `working_memory.v2` 的改动：通过不等于网络 MVP，不构成 `GATE-NN-MVP-v2` 证据。qualification、pilot 或 mechanism 通过只授权进入下一阶段，不等同于 Gate 通过。

## 7. 预算与设备

- 设备：Windows CUDA 通道（与 P4/P5 相同）。
- mechanism：五个 seed × 4 个重新训练 cell = 20 次训练，另加 10 个推理干预 cell，均覆盖四个任务。参考：P4 的 24-cell CUDA mechanism 实际用时 11.2 小时；按此粗估约 12–13 小时，patience 30 与 `preset-4` 可能使其更长。墙钟上限 48 小时。
- full：增加 2 个基线 × 5 seeds；墙钟上限 24 小时。

## 8. 科学边界

- 只检验人工网络中的计算机制，不对应真实脑区，不构成生物等价结论。
- T track 的结果只作为动机；本协议的裁决只来自 P6 namespace 的正式运行。
- T1c 与 T1d 已显示：带 surprise 门控的工作记忆在多情境任务上不如 GRU（恢复 −3.6pp / −2.5pp），按读出注意力寻址的门控也没有可靠地修复这一点。因此第 2 节保留全局门控，并把多情境列为已知缺陷；M1–M5 或 F1 失败时按协议提交负结果，F2–F4 的结果无论正负都如实报告，不在同一协议内换阈值或 seed。
