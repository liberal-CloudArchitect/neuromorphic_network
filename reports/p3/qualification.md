# P3 小样本资格运行记录

状态：`ENGINEERING PASSED — CLEAN-SHA RERUN REQUIRED`

本记录只证明工程路径，不进入 `GATE-3` 或 `GATE-NN-MVP` 的科学统计。

## 已完成证据

| 设备 | 代码状态 | Run ID | Cell | 墙钟 | 结论 |
|---|---|---|---:|---:|---|
| CPU | dirty worktree on `eef8135` | `p3-qualification-20260718T130205Z` | 39/39 | 135.04 s | ENGINEERING PASSED |
| MPS | dirty worktree on `eef8135` | `p3-qualification-20260718T130439Z` | 39/39 | 1,017.66 s | ENGINEERING PASSED |

两次运行均登记 9,728 条逐样本记录和 159 个 artifact；严格 verifier 确认无重复 sample、无缺失 cell 且 checksum 全部通过。MPS telemetry-on/off 的 logits、梯度和更新后参数最大差均为 `0`（容差 `1e-5`）。资格运行还生成了不可用于正式选择的 `QUALIFICATION_ONLY` pilot fixture。

此次审计补充验证了后台累计 72 小时预算、PID 防误杀、低磁盘安全 checkpoint、失败矩阵非零退出、checkpoint-v3 在模型变更前验证 sampler/RNG/shape/dtype，以及 continual 三阶段固定预算与遗忘曲线。冻结逻辑会拒绝 dirty worktree 的结果，因此这两次工程通过不能生成 qualification lock。

## 仓库门禁

- Ruff：PASSED
- Ruff format：PASSED（102 files）
- mypy：PASSED（102 source files）
- pytest：184 passed、1 个仅适用于无 MPS 主机的条件测试跳过
- CPU 39-cell qualification：PASSED
- MPS 39-cell qualification：PASSED
- 严格 artifact verifier：CPU/MPS 均 PASSED
- `make check`：当前受限执行器不允许 `make` 子进程访问外部 Conda env；其 Ruff、format、mypy、pytest、CPU smoke 等价命令已分别执行

## 解除阻塞

在最终提交并推送、远程 CI 绿色后，从该 clean SHA 重新执行：

```bash
make qualify-p3-mps
./scripts/p3_full_run.sh freeze-qualification artifacts/runs/<qualification-run-id>
./scripts/p3_full_run.sh record-ci
./scripts/p3_full_run.sh start
```

第一次 `start` 运行 12-cell pilot。完成并执行 `freeze-pilot` 后，第二次 `start` 才会加载冻结 preset 并启动 81-cell 正式矩阵。
