---
report: P4 qualification
status: PASSED
qualification_only: true
source_commit: 1856c9dcb5e88b735466bfe2ee5da312a2d4ecd7
last_updated: 2026-07-21
---

# P4 qualification 摘要

clean-SHA MPS run `p4-qualification-20260721T092723Z` 完成 8/8 个预注册 cell，0 failed、0 resource-limited。远程 Linux CPU CI run `29818335673` 同时通过。该结果关闭工程资格项 `P4-10`，只允许进入 4-cell pilot。

## 可复核证据

- 八种路径全部完成：full、predictor-off、loss-zero、feedback-zero、acute-feedback-off、shuffle-forecast、dense-memory、legacy-capacity。
- 写出 4 个 checkpoint-v4、4,864 条逐样本记录和 34 个带 checksum 的产物；校验无遗漏或不匹配。
- forecast path 与非零 logits 反馈均被实际覆盖；semantic reservation、capacity、dense/legacy routing 与 active/dense MAC 统计一致。
- 本地 Ruff、format、mypy、pre-commit 与完整测试通过：`230 passed, 12 MPS-only skipped`；另外 2 项 MPS checkpoint 等价测试通过。
- 独立复审在修复 predictor 多步契约与 mechanism-lock 闭环后给出 0 issue、APPROVE。

## 科学边界

qualification 是 seed 7 小样本工程验证，不能证明预测因果、forecast 相对 persistence 的正式改善、稀疏非劣性、OOD 泛化或 network MVP。下一步 pilot 仍只能读取 train/validation；test/OOD 与正式三 seed 统计必须等待冻结选择和后续 Gate。
