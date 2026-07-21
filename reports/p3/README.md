# P3 Reports

P3 分为两层裁决：`GATE-3` 验收预注册实验与复现完整性，`GATE-NN-MVP` 验收人工模块网络的确认性收益。qualification 报告只证明程序路径可运行，不进入正式统计。

正式 run `p3-full-dc6c259c-20260719T034230Z` 已完成。`formal.json` 是可机器复核的
10,000 次配对统计，`formal.md` 是对应摘要。`GATE-3 PASSED`，但
`GATE-NN-MVP FAILED`；项目不会生成正式 MVP bundle。

大型 checkpoint 与逐样本记录保存在 ignored `artifacts/runs/<run_id>/`；Git 只跟踪汇总、
偏差记录和 Gate 报告。
