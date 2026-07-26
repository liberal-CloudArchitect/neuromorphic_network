# P4 首次机制矩阵资源上限记录

结论：`RESOURCE_LIMIT / INCONCLUSIVE`。该运行不是程序崩溃，也没有形成 `GATE-4-MECH` 科学结论。

## 运行事实

- run：`p4-mechanism-fb53443a-20260722T160041Z`
- source SHA：`fb53443ac65948317f840092dccbed7b71badebe`
- 累计墙钟：`86,887.18 s`（约 `24.14 h`）
- cell：完成 `15/24`，失败 `0`，资源停止 `1`，尚未开始 `8`
- 产物：`62` 个已注册 artifact checksum 全部有效，`514,560` 条逐样本记录
- 缺口：seed 29 的 `legacy-capacity` 控制和 seed 43 的全部八个 cell
- 未生成 mechanism report 或 mechanism lock；full 阶段仍必须拒绝启动

## 原因与处置

已完成的零训练步控制 cell 仍分别耗时约 36～44 分钟。主要原因是每个 cell 的 SmallGraph test 与三个 OOD live rollout 共 8,192 个 episode，旧实现逐样本发起 MPS 前向，评估吞吐不足以让冻结的 24-cell 矩阵在 24 小时内结束。

不延长原协议预算、不删减样本、不跳过 cell，也不把不同 SHA 的结果拼接。`CR-004` 仅把独立 episode 批量执行并增加结构化日志；新 SHA 必须重新通过 CI、MPS qualification 和 pilot 后，从新的 run ID 重跑完整机制矩阵。

## 优化前置验证

在随机初始化的同一 modular-v2 模型、P4 test 图和 MPS 上：

- 32 个样本：标量/批量逐样本记录完全一致；小规模受启动开销影响，速度比 `0.94×`。
- 128 个样本、batch `64`：标量/批量逐样本记录完全一致；标量 `24.72 s`，批量 `8.26 s`，速度比约 `2.99×`。
- 扩展吞吐验证保持逐样本记录完全一致：batch `64/128/256` 分别约 `15.5/21.2/29.4 samples/s`；batch `512/1024` 在 1,024 个样本上分别约 `46.2/52.6 samples/s`。正式 live rollout 因此使用 inference-only batch 上限 `1024`，训练 batch 仍为 `64`。

这些数字只验证执行等价性与吞吐方向，不能替代新 clean SHA 的 qualification 或正式机制结果。
