---
gate: GATE-4-MECH-CUDA
status: FAILED
protocol: p4-protocol-v2-cuda-mechanism
source_sha: 6fcc69a52b376234c34604a4cae241ac85923d6d
run_id: p4-cuda-mechanism-6fcc69a5-20260823T072800Z
last_updated: 2026-08-24
---

# GATE-4-MECH-CUDA：机制裁决

## 裁决

`FAILED`。工程执行完整且可复核，但预测行为因果和稀疏路由最坏 seed 非劣性没有达到预注册阈值。不得生成 mechanism lock、启动 81-cell full 或授予 network MVP 标签。

## 完整性

- 24/24 cells 完成，0 failed，0 resource-limited。
- 总墙钟 `40,301.14 s`，远低于 48 小时预算。
- 823,296 条逐样本记录、554 个登记产物，checksum 全部有效。
- 正式 seeds `[17,29,43]`、10,000 次配对分层 bootstrap、RNG `20260715` 与 Holm family 均按冻结协议执行。
- 失败终态未生成 `artifacts/p4-cuda/mechanism-lock.json`。

## 科学结果

| 检验 | 结果 | 证据 |
|---|---|---|
| 预测质量 | PASS | 覆盖率 100%；配对改善 `20.72%`，95% CI `[17.28%,24.52%]`，Holm p `0.0006` |
| 预测因果 | FAIL | macro AULC `+0.56%`，95% CI `[-0.35%,1.96%]`，Holm p `0.5193`；DRS seed 17/29 最终分数下降 `10.34/8.76pp` |
| 稀疏成本 | PASS | optional MAC 减少 `51.72%`，capacity drops `0`，AR reservation coverage `100%` |
| 稀疏非劣性 | FAIL | DRS seed 17/29 相对 dense 下降 `3.21/6.07pp`，超过 `2pp` 最坏 seed 界限 |

预测误差相对 persistence 在 AR/DRS 改善 `28.31%/48.49%`，但 SmallGraph 恶化 `14.21%`。DRS full router 的 working-memory 执行占比在 seeds 17/29/43 约为 `87.5%/25.6%/88.6%`，显示 seed 29 明显漂向 episodic。

## 解释边界

该结果否证 `predictive_adapter.v2` 的直接感觉表示反馈与 `sparse_router.v2` 的纯 top-1 学习路由，而不是否证所有预测学习或稀疏模块化。新架构必须使用新版本、新数据 namespace 和新协议；不得在 P4 内修改阈值、删 seed 或挑选性重跑。
