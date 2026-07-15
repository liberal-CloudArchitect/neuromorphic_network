---
gate: GATE-0
status: PASSED
reviewed_commit: 472c2c5b06cc9e9d9ce6568b3aeb3483dec1c38f
review_date: 2026-07-15
reviewer: independent-verifier-agent
---

# GATE-0：科学假说与计算规格评审

## 裁决

`PASSED`。本裁决只接受 P0 科学与计算规格，不证明任何类脑模块有效，也不关闭尚待远程 GitHub Actions 证据的 `GOV-005`。

## 验收结果

| Test ID | 结果 | 证据 |
|---|---|---|
| `AT-P0-01` | PASS | 六模块各有至少两项来源，并登记 Evidence、Abstraction、Hypothesis、输入输出、state、loss、对照和否证条件。 |
| `AT-P0-02` | PASS | `BrainPacket`、`ModuleContext`、`ModuleState`、`ModuleOutput`、`BrainModule` 及 valid/invalid/shape/dtype/device 测试；`reset_mask` 统一为 `[B,T]`。 |
| `AT-P0-03` | PASS | 三任务 split seeds `1101/2201/3301/4401`、smoke seed 7、正式 seeds 17/29/43、预算、指标、基线和统计协议一致。 |
| `AT-P0-04` | PASS | `telemetry-v1` schema、事件 round-trip、六 module IDs 和无 atlas/Web/viewer 字段或依赖检查。 |
| `AT-P0-05` | PASS | 科学边界禁止一脑区一功能、生物等价、BOLD 或真实神经活动声称。 |

## 独立复验

- Ruff check：PASS。
- Ruff format check：47 files PASS。
- mypy strict：47 files PASS。
- pytest：58 passed、2 skipped；独立审查沙箱无法访问 MPS，MPS 由主线在本机单独验收。
- `git diff --check`：PASS。
- module IDs、专家分区、split seeds、task versions 和切换位置结构比对：PASS。

## 审查中发现并关闭的问题

1. `reset_mask` 的 `[B]` / `[B,T]` 冲突：统一为 `[B,T]` 并加入时间维验证。
2. Delayed Rule Switch OOD 切换位置不一致：统一为 trial 3，即归一化位置 0.75。
3. `small-graph-v1` task version 命名不一致：统一为带连字符版本。
4. Associative Recall 干扰 key 可能与存储/query key 碰撞：生成器改为从未使用 key 集合采样并加入回归测试。

## 未关闭的治理证据

远程仓库已配置并推送；`GOV-005` 仅在 GitHub Actions 对相应 commit 首次显示绿色后改为 `DONE`。该事项不改变本报告的 P0 技术裁决，但会阻止 GATE-1 的最终关闭。
