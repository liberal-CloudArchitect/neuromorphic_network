# P4 accelerator 执行隔离验证报告

状态：`ENGINEERING PASSED`

日期：2026-08-22

实现 SHA：`d9a49b861dcd02a0225c221c1fb4c54545924792`

## 结论

`CR-006` 的 supervisor/worker 执行隔离已在 CPU、Apple MPS、Windows RTX 4060/CUDA 和 GitHub Actions CPU CI 上完成验证。原有长生命周期 MPS 进程中的小时级 SmallGraph rollout 拖尾未在 fresh-process 资格矩阵中复现；worker 结果、逐样本记录与 telemetry 均经过原子写入、登记和 checksum 校验。

本报告只关闭执行可靠性工作项 `P4-17`。它不构成 `GATE-4-QUAL` 独立科学评审，不证明预测模块有效、稀疏路由非劣或网络 MVP 合格，也不改变旧 mechanism runs 的 `RESOURCE_LIMIT / INCONCLUSIVE` 结论。

## 根因证据

- 同一 `predictor-off/s29` checkpoint 的 OOD scale rollout 在老化 MPS 进程中耗时约 7.49 小时，在新进程中完成 2,048 样本约 61.3 秒。
- 老化进程物理内存约 69.3 GB，线程采样反复出现 `index_select_mps` 与 `MPSGraphCache::CreateCachedGraph`。
- SmallGraph active rows 与 optional expert rows 产生动态 shape；MPSGraph 的进程级缓存不会被 `torch.mps.empty_cache()` 清空。
- 因此修复边界选择进程隔离，而非改变模型、样本、loss、指标或科学阈值。

## 实现边界

- accelerator 训练按 cell 在独立 worker 中执行。
- 标准评估按 task 隔离；SmallGraph test 与三个 OOD live rollout 按 view 隔离。
- supervisor 不加载模型或执行前后向，只协调 registry、worker 和墙钟预算。
- worker 结果先写临时文件再原子替换；只有 config、matrix、protocol、checkpoint 与产物 checksum 全部一致才允许复用。
- Windows 控制器补充 CUDA preflight、Conda 环境 PATH、后台进程存活检查、job breakaway 和 stale launch 恢复。

## 验证结果

| 平台 | 证据 | 结果 |
|---|---|---|
| macOS / MPS | `p4-qualification-20260821T180407Z` | 8/8 cells；0 failed；0 resource-limit；4,864 records；186 artifacts；checksums valid；1008.06 s |
| Windows 11 / RTX 4060 | `p4-cuda-qualification-d9a49b86-20260821T180356Z` | 8/8 cells；0 failed；0 resource-limit；4,864 records；186 artifacts；checksums valid；473.60 s |
| Windows 定向回归 | E 盘 `brain` 环境 | 49 passed、2 skipped；CUDA forward/backward smoke 通过 |
| 本地完整测试 | Python 3.12 / PyTorch 2.12.1 | 255 passed、12 skipped；Ruff、mypy、pre-commit 通过 |
| GitHub Actions | run `32511375072` | `success` |

MPS clean-SHA qualification 中，观察到的 SmallGraph 独立 live-rollout worker 单 view 用时约 3.05–6.49 秒。该结果证明小时级退化已被进程生命周期隔离，但不能外推正式规模 mechanism 的总墙钟或科学效果。

## CUDA 环境清单

- 主机：Windows 11 `10.0.26200`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB
- 驱动：572.16
- Python：3.12.13，环境位于 `E:\conda\envs\brain`
- PyTorch：2.12.1+cu126
- CUDA runtime：12.6

CUDA qualification 是跨平台工程证据。若未来把 CUDA 纳入 P4 科学 Gate，必须另行冻结设备、driver、wheel、数值容差与成本口径。

## 后续约束

1. 在 implementation SHA 变化后，qualification、pilot 和 mechanism 锁必须重建；不得拼接旧 run。
2. `P4-10` 仍需独立 verifier 生成新的 `GATE-4-QUAL` 报告和资格锁。
3. 只有新的 qualification 与 pilot 同 SHA 闭环后，才能启动 24-cell mechanism。
4. 旧 resource-limit 运行继续作为诊断证据保留，不能升级为通过或用于补齐缺失 seed。
