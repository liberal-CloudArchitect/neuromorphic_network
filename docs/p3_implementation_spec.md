---
title: P3 因果、泛化与网络 MVP 实施规格
status: ACCEPTED
phase: P3
protocol: p3-protocol-v2
last_updated: 2026-07-18
---

# P3 因果、泛化与网络 MVP 实施规格

P3 验证人工模块化网络是否产生可重复的计算收益。所有结论只适用于本项目的人工模型，不表示脑区一一对应、生物等价或临床意义。3D、atlas、WebSocket、SNN 和生成模型不属于本阶段。

## 治理与阶段边界

- `GATE-3` 验收预注册矩阵、失败记录、恢复、复现、统计和报告是否完整。被否证的科学假说不阻止科研阶段完成；缺失 mandatory cell、少于三个有效 seed 或不可复现则不能通过。
- `GATE-NN-MVP` 独立验收网络收益。只有该 Gate 通过后才能生成 `network-mvp-v1` bundle 或使用“网络 MVP qualified”表述。
- 正式结果不得驱动阈值、seed、任务、比较 family 或预算变更。需要变更时建立新 protocol 和 deviation，保留原结果。
- seed `7` 仅用于 pilot/qualification；正式训练 seeds 固定为 `[17, 29, 43]`。

## 配置与预算

- `p3-suite-v1` 描述完整矩阵和总墙钟；`p3-experiment-v1` 描述一个可恢复 cell。
- shared 模型是主比较：每任务最多 5,000 updates，round-robin 总上限 15,000；每任务每 100 updates 执行全任务 validation，patience 10，`min_delta=0.001`。
- per-task 模型是正式敏感性比较：每 run 最多 5,000 updates，沿用相同 validation/early-stop 规则。
- 正式 MPS 矩阵累计墙钟最多 72 小时，重启或恢复不得重置预算。训练、评估和成本分析都会检查期限与磁盘；超限在已有 checkpoint 上记录 `RESOURCE_LIMIT`，不得删减矩阵后宣称完成。

## 双主基线与训练公平性

- shared modular 同时与参数匹配 shared GRU 和 shared Transformer 比较；两者都是 confirmatory 主基线，不能事后挑选较弱者。
- shared 模型计入所有 adapters、共享 backbone 和三个 heads；参数匹配误差必须不超过 ±5%。
- per-task 报告总参数、可训练参数和真实活跃路径参数，只能在 per-task regime 内比较。
- 同一 regime/seed 使用相同 sample index 顺序、batch、训练 token、验证时点和 test/OOD checkpoint 选择规则。
- `loss-reduction-v2` 按有效 token/event 归约；P1/P2 reduction 保持不变。

## 数据、AULC 与泛化

- 保留三个 v1 ID 任务；P3 OOD view 使用独立 task version 和 content hash，覆盖容量/干扰、延迟/规则组合和图规模/拓扑转移。
- 新增 `analysis` split，seed `5501`，正式 512 样本。它只用于 AULC、probe 和顺序学习分析，不参与调参、早停或 checkpoint 选择。
- AULC 每 100 task updates 在 analysis split 计算；横轴为累计训练样本占最大预算的比例，使用梯形积分。早停后以前次冻结值延伸到预算终点。
- ID task score 先按 chance 标准化再三任务等权平均。OOD score 为 `(OOD - chance) / (ID - chance)`，不裁剪；`ID <= chance` 时为 undefined，相关收益不得通过。
- 顺序学习按 seed 使用 Latin orders：17=`AR→DRS→SG`、29=`DRS→SG→AR`、43=`SG→AR→DRS`，每阶段精确 1,500 updates，不应用通用 early-stop。analysis 曲线以全局 step 保留跨阶段历史，`forgetting = 历史最佳 analysis score - 最终 score`。

## 因果与统计

- 确认性 retrained contrasts：episodic no-read/no-write、working reset-every-step、predictive-loss-zero。每项从相同 seed 的随机初始化开始，不能继承 full checkpoint。
- 急性 predictive 关闭是行为零效应负对照，因为 P2 时序不让预测路径反馈同一步动作。
- 逐样本配对 key 为 model、variant、seed、task、distribution、sample index 和 stratum；缺 seed、重复或不匹配记录必须在统计前失败。
- 使用 10,000 次 seed→stratum/sample 配对 bootstrap、RNG `20260715`、双侧 95% CI 和 Holm 校正。
- 网络收益 family 为 task score、OOD、AULC、forgetting × GRU/Transformer 共八项。一个类别只有同时达到两种基线的阈值、CI 支持方向且 Holm-adjusted `p <= 0.05` 才通过。
- 因果 family 门槛：episodic ≥15pp、working ≥10pp、predictive AULC 相对 ≥10%；三项必须全部通过。
- 稀疏收益相对 dense modular：optional active MAC 减少 ≥20%，任务分数非劣界 2pp。

## 资格与正式运行分离

- `qualification` profile 使用 seed 7、`64/32/32/32/32` 样本和缩小 updates，固定 39 个 cell，覆盖所有模型/regime、三种急性干预、六种控制、恢复、统计、真实 analysis 表征和成本产物，但必须标记 `qualification_only=true`。
- `pilot` profile 独立运行 12 个预注册 cell，每 cell 1,000 updates，只访问 train/validation。选择顺序固定为 validation macro AULC 降序、最终 loss 升序、preset ID 升序；选择结果必须绑定 clean Git SHA 后冻结。
- 正式 profile 固定 81 个 cell，并从 pilot lock 读取三种模型的 preset；正式 runner 不重新选择超参数。
- 正式运行只能从 qualification 通过、远程 CI 绿色且 worktree 干净的冻结 SHA 启动。
- 后台 runner 只写入 ignored artifacts，不修改源码、提交或推送。
- `start`、`resume`、`stop --force`、`verify` 均校验记录的 PID、路径、SHA 和运行状态；qualification lock 只接受 clean SHA 的完整 MPS 资格结果。
