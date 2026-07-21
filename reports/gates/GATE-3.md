---
gate: GATE-3
status: PASSED
reviewed: 2026-07-21
source_run: p3-full-dc6c259c-20260719T034230Z
source_commit: dc6c259c559eb3af6e4cc74c905cc2dfadf3690a
protocol: p3-protocol-v2
---

# GATE-3 独立完整性裁决

## 结论

`GATE-3 PASSED`。本结论只接受 P3 预注册矩阵、恢复纪律、产物完整性和诚实统计；
它允许科研工具版本升级到 `0.4.0`，不表示模块化网络优于双主基线，也不授予网络 MVP。

## 复核证据

| 项目 | 结果 | 证据 |
|---|---|---|
| 正式矩阵 | PASS | 81/81 mandatory cells 为 `COMPLETED`；0 failed、0 invalid、0 resource-limit |
| 正式重复 | PASS | seeds `[17, 29, 43]` 各 27 cells，未使用 pilot seed 7 |
| 产物完整性 | PASS | 352 个登记产物 checksum 正确；1,605,120 条 sample records；无缺失 cell |
| 运行预算 | PASS | MPS suite 176,968.54 秒，即 49.16 小时，小于 72 小时上限 |
| 冻结选择 | PASS | Modular/GRU/Transformer 均使用 pilot 选出的 `preset-2` |
| 数据纪律 | PASS | 逐样本证据只含 analysis/test/OOD；没有 train/validation 选择记录 |
| 配对统计 | PASS | 10,000 次 seed→stratum/sample bootstrap，RNG `20260715`，双侧 95% CI 与 Holm |
| 独立复算 | PASS | `python -I` 隔离解释器二次复算；JSON 与 Markdown 均逐字节一致 |
| 源提交 CI | PASS | GitHub Actions run `29648030725` 对源 SHA 的 CPU 门禁为 success |
| 科学边界 | PASS | 所有失败与协议缺口原样报告；未创建正式 MVP bundle |

正式 JSON SHA-256 为
`3311d2fecbceda5a8f4866dd74e5f2a73888d5391356dd366cba1686e18e9053`；
Markdown SHA-256 为
`d1e6a97bd3469c38fa3d7dc1d2b210fc508727059e15dfad2ed9f06c54bb93dc`。

## 偏差处理

`DR-001`～`DR-003` 分别记录 chance/OOD primary metric、forgetting 阈值和正式容量计数
缺口。它们使相关科学类别不能通过，但没有删除 mandatory cell、替换 seed、改变预算或
污染模型选择，因此不阻止研究完整性 Gate。任何后续修订必须使用新 protocol 与新数据，
不能追溯改写本次结果。

## 使用约束

P3 的科学收益由独立的 `GATE-NN-MVP` 裁决。本报告不得被引用为脑区一一对应、生物等价、
临床意义或类脑优势的证据。
