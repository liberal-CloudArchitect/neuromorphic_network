---
gate: GATE-4-QUAL
status: PASSED
protocol: p4-protocol-v1
target_release: 0.5.0
template: false
last_updated: 2026-07-21
---

# GATE-4-QUAL：P4 工程资格裁决

## 裁决

`PASSED`。提交 `1856c9dcb5e88b735466bfe2ee5da312a2d4ecd7` 的本机 MPS qualification、Linux CPU CI、checkpoint 恢复、产物校验与独立代码审查全部通过。该裁决只允许启动 4-cell pilot，不构成预测因果、泛化、稀疏收益或 network MVP 结论。

## 冻结输入

| 项目 | 实际值 | 状态 | 证据 |
|---|---|---|---|
| protocol | `p4-protocol-v1` | PASS | protocol hash 与 qualification registry 一致 |
| source SHA | `1856c9dcb5e88b735466bfe2ee5da312a2d4ecd7`，clean 且 `HEAD == origin/main` | PASS | qualification report `git_dirty=false` |
| qualification profile | seed `7`、CPU/MPS、`qualification_only=true` | PASS | MPS run `p4-qualification-20260721T092723Z`；CI run `29818335673` |
| P4 split seeds | train `11101`、validation `12201`、test `13301`、OOD `14401`、analysis `15501` | PASS | split/hash 单元与集成测试 |
| matrix | `8/8`，0 failed，0 resource-limited | PASS | registry 与 34 个 artifact checksum |

## 工程验收

| 检查 | 结果 | 证据 |
|---|---|---|
| 跨步预测与反馈 | PASS | forecast path 被覆盖，logits 反馈非零；模块级多步调用被拒绝，序列由 network 逐步执行 |
| 三种训练干预与四种冻结控制 | PASS | full、predictor-off、loss-zero、feedback-zero、acute-feedback-off、shuffle-forecast、dense-memory、legacy-capacity 全部 COMPLETED |
| semantic top-1 与 AR reservation | PASS | sparse active MAC 小于 dense；AR reservation/execution 100% |
| 容量与 dense 对照 | PASS | `capacity_drops=0`；dense executed MAC 与 dense accounting 一致 |
| 数值健康 | PASS | 8 个 summary 均有限，无 NaN/Inf 违规 |
| checkpoint-v4 | PASS | 4 个训练 cell 写出 checkpoint；额外 2 项 MPS 恢复等价测试通过 |
| telemetry-v2 | PASS | 每 cell 6 个 schema-valid 聚合事件，合计 48 个 |
| 产物与逐样本记录 | PASS | 4,864 条 sample records；34 个 checksum；`verify_p4_run` 无缺失或不匹配 |
| 本地质量门禁 | PASS | Ruff、format、mypy、pre-commit；`230 passed, 12 MPS-only skipped`，另有 2 项 MPS 测试通过 |
| 远程 CPU CI | PASS | [GitHub Actions run 29818335673](https://github.com/liberal-CloudArchitect/neuromorphic_network/actions/runs/29818335673) 全部 13 个质量/测试步骤成功 |
| 独立审查 | PASS | 首轮发现并修复 mechanism-lock 与 predictor 多步契约问题；复审 0 issue、APPROVE |

## 后续限制

- 下一步仅可运行 pilot；pilot 必须只访问 train/validation，并独立满足 coverage、forecast improvement 与 feedback 资格条件。
- 24-cell mechanism 只有同时通过预测因果、预测质量、稀疏非劣性和冻结统计后才生成同 SHA `mechanism-lock.json`。
- 81-cell full 在 mechanism lock 缺失或失败时必须拒绝启动。
- 当前版本保持 `0.4.0`，不得生成 `network-mvp-v2` bundle，也不得改写 P3 的负结果。
