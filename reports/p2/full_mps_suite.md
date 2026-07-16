# P2 完整 MPS Suite 证据摘要

本地完整运行 `p2-suite-s7-20260716T112848Z` 已通过。该结论只证明六个人工计算模块的工程闭环、真实稀疏执行、恢复与 telemetry 数值等价，不证明生物等价、模块功能分工或相对 P1 基线的收益。

## 运行身份

- commit：`9499a33645e0b446e2039b790387b240f5758bda`
- 工作树：clean
- 设备：Apple MPS，PyTorch 2.12.1
- 预算：四个预训练阶段各 100 updates；telemetry off/on 各 600 updates
- 总耗时：2,257.16 秒
- 大型运行目录：`artifacts/runs/p2-suite-s7-20260716T112848Z/`（按策略不提交）

## 硬条件结果

| 条件 | 结果 |
|---|---|
| 四阶段首 10/末 10 平均 loss | 全部下降 |
| 三任务联合首 10/末 10 平均 loss | 两分支全部下降 |
| 六模块 gradient coverage | 全部为 true |
| executed top-k | 每个有效 token 精确 2/3 |
| capacity drops | 0 |
| raw 专家份额 | 47.01%、34.85%、18.14%，均高于 5% 健康线 |
| optional MAC | active 22,489,618,944 < dense 32,631,603,200 |
| telemetry 参数最大差 | `9.7312e-07`，满足 MPS 容差 |
| telemetry 主指标最大差 | 0 |
| telemetry schema | 3,600/3,600 逐条有效 |
| artifact checksum | 15/15 匹配 |

路由改派率为 23.63%；这说明 `executed_mask` 与原始 score top-2 必须继续分开解释。当前只有三个 eligible experts，`eligible>=4 时 active<=60%` 条件仍为 N/A，不能据此声称端到端稀疏收益。

## 尚待关闭

本摘要不是 GATE-2 裁决。仍需独立 verifier 报告及 commit 对应的远程 CPU GitHub Actions 绿色证据；在两者齐备前，不升级到 0.3.0，也不把 P2 台账标记为 DONE。
