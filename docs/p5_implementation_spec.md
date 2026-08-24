---
status: ACCEPTED
protocol: p5-protocol-v1
date: 2026-08-24
scope: mechanism-qualification
---

# P5：预测 surprise 与稳定语义路由实施规格

## 假说

1. 预测误差更适合作为注意/路由调制信号，而不是在当前观察已经可用时再次改写感觉表示。
2. DRS 的 cue、delay 和 response 都依赖规则工作记忆，因此 working memory 是必经语义路径；少量不确定 token 可额外访问 episodic。
3. 稀疏路由必须同时接受任务梯度和 observation-derived 语义监督，不能只依赖 load-balance 形成分工。

## 冻结架构

- 模块 ID：`predictive_adapter.v3`、`sparse_router.v3`，组合器 `ModularBrainNetworkV3`。
- residual forecast 最大 delta `0.25`，初始严格等于 persistence。
- surprise 从合法相邻 transition 的 SmoothL1 error 计算并 detach；不直接进入 selector representation。
- AR store/query episodic coverage `100%`；DRS valid working coverage `100%`。
- DRS dual-route 每 step 不超过 `floor(valid × 25%)`；capacity drops 必须为 `0`。
- straight-through fusion 只改变 scorer 的反向估计，不执行未选择专家。
- P5 split seeds：train `21101`、validation `22201`、test `23301`、OOD `24401`、analysis `25501`。

## 资格 Gate

当前只授权 seed 7、`64/32` train/validation、batch 8、三任务各 4 updates 的 qualification：

- 三任务 forward/backward、loss、gradient 和 state 必须有限；
- predictor 与 router scorer 都必须获得非零梯度；
- 上述语义覆盖、dual budget、zero-drop 与 active<dense 不变量必须满足；
- forecast/persistence 指标只记录方向，不构成科学收益结论；
- qualification 不访问 analysis/test/OOD，不生成 network MVP。

## 后续正式协议待冻结项

在 CUDA/MPS qualification 都通过后，另行冻结 P5 pilot 与 mechanism：

- 正式 prediction quality 要求三个任务分别相对 persistence 改善至少 5%；
- predictive causal macro AULC 至少提高 5%，任一任务/seed 最终下降不超过 2pp；
- optional MAC 至少降低 20%，任一任务/seed 相对 dense 下降不超过 2pp；
- AR/DRS 语义覆盖 100%、dual-route ≤25%、drops=0；
- seeds `[17,29,43]`、10,000 bootstrap、RNG `20260715`、95% CI 与 Holm p≤0.05。

阈值必须在正式 test/OOD 访问前冻结。checkpoint-v5、telemetry-v3、后台控制和正式矩阵尚未完成，不能把当前 qualification 当成 P5 Gate 通过。
