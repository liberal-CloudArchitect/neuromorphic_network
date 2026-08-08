# P4 第二次机制矩阵资源上限记录

结论：`RESOURCE_LIMIT / INCONCLUSIVE`。批量 live rollout 修复通过资格验证，但冻结的 24 小时机制预算仍不足以完成三个正式 seed。

## 运行事实

- run：`p4-mechanism-bcbbe9e2-20260727T044447Z`
- source SHA：`bcbbe9e2492c6483f56050562032ae61111395d5`
- 累计墙钟：`86,953.05 s`（约 `24.15 h`）
- cell：完成 `8/24`，失败 `0`，资源停止 `1`，未开始 `15`
- 产物：`35` 个登记 artifact checksum 全部有效，`274,432` 条逐样本记录
- 完成范围：seed 17 的全部八个 cell；seed 29 的 full 在训练完成后的评估阶段触发资源上限；seed 43 未开始
- 未生成 mechanism report 或 mechanism lock；full 阶段仍被前置锁拒绝

## 时间证据

seed 17 的八个完整 cell 合计 `44,080.27 s`：四个重训练 cell 共 12,600 updates，耗时 `34,950.28 s`；四个冻结 checkpoint 控制耗时 `9,130.00 s`。按同等吞吐执行三个 seed 约需 36.7 小时，24 小时上限不具可行性。

seed 29 full 在 step 2,400 后进入评估，最后训练 heartbeat 与资源停止之间约十小时。fresh-process 定向复现证明同一 checkpoint 的 256 图 rollout 可在 `8.68 s` 完成，因此该空窗更符合系统挂起或长进程资源退化，而不是模型固有推理成本。无论是否扣除该空窗，seed 17 的实测已足以否定 24 小时预算可行性。

## 处置

`CR-005` 建立 `p4-protocol-v2`：只把 mechanism 资源上限调整为 48 小时，并补充 evaluation heartbeat 与 cell 间 MPS cache 清理。科学协议其余部分不变。新协议必须在新 clean SHA 上重建完整资格链，不续接本 run。
