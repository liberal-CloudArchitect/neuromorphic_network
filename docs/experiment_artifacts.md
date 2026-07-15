---
title: 实验产物与证据存档
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-15
---

# 实验产物与证据存档规则

> 本规则已作为 P0 产物治理规范接受；规范存在不代表任何实验有效或 Gate 已通过。

## Run ID

建议格式：`YYYYMMDDTHHMMSSZ_<task>_<model>_<seed>_<short-commit>`。Run ID 创建后不可复用；重跑必须获得新 ID，并通过 `parent_run_id` 或 `supersedes` 建立关系。

## 最小产物集合

| 产物 | 建议格式 | 必含内容 |
|---|---|---|
| run manifest | JSON/YAML | run ID、parent、commit、dirty flag、环境、设备、seed、任务/数据版本、配置 checksum |
| resolved config | YAML | 完整合并后的训练、模型、任务、预算和 telemetry 配置 |
| environment manifest | JSON/TXT | Python、PyTorch、OS、accelerator、依赖与锁文件 checksum |
| metrics | JSONL/Parquet | step、split、metric、value、timestamp、schema version |
| checkpoint manifest | JSON | checkpoint URI、checksum、step、兼容版本、包含的状态 |
| telemetry | JSONL/Parquet | `telemetry-v1` 事件或版本化后继；允许完全关闭 |
| statistical report | Markdown/JSON | 每 seed、CI、效应量、阈值、成本、失败与偏差 |
| gate report | Markdown | Test ID、证据链接、commit hash、评审裁决与签署 |

## 目录规范

```text
artifacts/runs/<run_id>/
├── manifest.json
├── config.resolved.yaml
├── environment.json
├── metrics.jsonl
├── checkpoints/manifest.json
├── telemetry/                  # 可选、可为空
├── reports/
└── logs/
```

大体积 checkpoint、数据和 telemetry 不提交 Git；仓库只保存 schema、模板、checksum、可复现命令和稳定 URI。任何外部存储位置必须记录保留期与访问控制。

## 证据链与状态

- `DONE` 任务必须链接到具体 commit、测试、run manifest、统计报告或评审记录。
- 未评审文档、空目录、smoke 和单 seed 结果不能作为 Gate 通过证据。
- 每份 Gate 报告记录被评审 commit hash；dirty worktree 的结果默认不可作为正式证据。
- 失败和中断 run 保留 manifest、日志、退出原因和最后 checkpoint，不得静默删除。
- 产物 schema 迁移必须保留原始文件、迁移工具版本和校验前后 checksum。

## 完整性检查

- [x] manifest 规范要求可解析且引用的配置、环境和数据版本存在。
- [x] checksum 规范要求校验通过，run ID 唯一且父子关系无环。
- [x] 正式结果规范要求可从 fresh 环境按记录命令复现。
- [x] telemetry 规范禁止无界原始张量或敏感输入，并要求可关闭。
- [x] 报告规范包含失败、方差、成本、限制和替代解释。
