# 贡献指南

本仓库当前为专有研究项目，接受贡献不代表授予使用或分发许可。提交任何内容前，贡献者应确认拥有相应权利并同意项目维护者在本仓库中使用该贡献。

## 开发流程

1. 使用 `make env` 或 `make env-update` 准备 `brain` 环境。
2. 从 `main` 创建短生命周期分支。
3. 保持类脑网络主线与可选可视化之间的单向依赖：viewer 可以读取 telemetry，网络包不得导入展示代码。
4. 代码标识符、docstring 和提交信息使用英文；研究说明可使用中文。
5. 提交前执行 `make check`，Apple Silicon 环境额外执行 `make smoke-mps`。

## 科学要求

- 区分 Evidence、Abstraction 与 Hypothesis。
- 不得把人工模型活动称为 BOLD、真实神经放电或意识证据。
- 新模块必须给出输入、输出、状态、损失、任务指标、消融和否证条件。
- P0 文档的 `DRAFT` 标记只能由对应 Gate 评审移除。

## 提交要求

提交信息遵循仓库 AGENTS.md 中的 Lore Commit Protocol：首行说明意图，正文记录约束与方案，并使用 `Confidence`、`Scope-risk`、`Tested`、`Not-tested` 等原生 Git trailers 记录证据。
