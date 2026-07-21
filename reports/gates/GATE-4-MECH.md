---
gate: GATE-4-MECH
status: NOT_RUN
protocol: p4-protocol-v1
template: true
last_updated: 2026-07-21
---

# GATE-4-MECH：预测闭环机制门禁模板

`NOT_RUN`。正式矩阵固定为 3 seeds × 8 cells，共 24 cells。必须同时满足：

- full 相对 retrained predictor-off 的三任务 macro AULC 提高至少 5%，且任一任务最终分数下降不超过 2pp；
- transition forecast error 相对 persistence baseline 降低至少 5%，至少两项任务方向为正；
- semantic top-1 相对 dense-memory 的 optional MAC 降低至少 20%，各任务分数下降不超过 2pp；
- `capacity_drops=0`，AR STORE/QUERY episodic reservation 与 execution 均为 100%；
- seeds `[17,29,43]`、10,000 次严格配对 bootstrap、RNG `20260715`、95% CI 与 Holm 校正完整。

本 Gate 失败时不得启动 full，也不得在同一协议内调整阈值、seed 或预测权重。
