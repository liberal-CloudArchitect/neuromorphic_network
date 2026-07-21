---
gate: GATE-4-QUAL
status: NOT_RUN
protocol: p4-protocol-v1
target_release: 0.5.0
template: true
last_updated: 2026-07-21
---

# GATE-4-QUAL：P4 工程资格模板

## 裁决

`NOT_RUN`。本文是待填写模板，不表示通过、失败或任何科学收益。

## 冻结输入

| 项目 | 预期值 | 实际值 | 状态 | 证据 |
|---|---|---|---|---|
| protocol | `p4-protocol-v1` | PENDING | PENDING | PENDING |
| source SHA | clean SHA，且与远程 CI 通过的 SHA 一致 | PENDING | PENDING | PENDING |
| target release | `0.5.0` | PENDING | PENDING | PENDING |
| qualification profile | seed `7`、CPU micro 与 MPS、`qualification_only=true` | PENDING | PENDING | PENDING |
| P4 split seeds | train `11101`；validation `12201`；analysis `15501`；test `13301`；OOD `14401` | PENDING | PENDING | PENDING |
| matrix | 8/8 qualification cells，无重复 cell ID | PENDING | PENDING | PENDING |

## 工程验收

| 检查 | 通过条件 | 结果 | 证据 |
|---|---|---|---|
| 跨步闭环 | 仅消费相邻前一步 forecast；动作后提交下一 forecast；terminal/padding 清理正确 | PENDING | PENDING |
| 干预区分 | predictor-off、loss-zero、feedback-zero 产生各自冻结语义 | PENDING | PENDING |
| 反馈边界 | 第一步反馈为零；有效后续反馈非零、有限且绝对值 ≤`0.25` | PENDING | PENDING |
| semantic top-1 | 每个有效 token 精确执行一个 episodic/working expert；未选 expert 不执行 | PENDING | PENDING |
| AR reservation | STORE/QUERY 的 episodic reservation/execution=`100%` | PENDING | PENDING |
| 容量健康 | `capacity_drops=0` | PENDING | PENDING |
| 稀疏记账 | active optional calls 小于 dense calls；成本字段有限且可核验 | PENDING | PENDING |
| 数据隔离 | P4 namespace/content hash 使用冻结 split seeds；无 P1–P3 样本流复用 | PENDING | PENDING |
| 数值健康 | loss、gradient、参数、forecast error 和 persistence error 均为有限值 | PENDING | PENDING |
| 恢复一致性 | checkpoint、sampler、RNG、lock hash、配置 hash 和矩阵 hash 校验通过 | PENDING | PENDING |
| artifact 完整性 | 8 个 mandatory cell 均 COMPLETED；registry、summary、checksum 无缺失 | PENDING | PENDING |
| 失败保留 | INVALID、FAILED、RESOURCE_LIMIT 不被覆盖或伪装为完成 | PENDING | PENDING |

## 后续阶段资格

本 Gate 通过后只允许启动 4-cell pilot。pilot 仍须独立满足 coverage ≥`0.95`、forecast 相对 persistence 改善 ≥`5%`、feedback 非零，并形成绑定同一 clean SHA 的 pilot lock；随后才可启动 24-cell mechanism。81-cell full 还必须持有通过的 mechanism lock。

## 使用约束

- 填写前保持 `status: NOT_RUN`；只有真实运行、严格 verifier 和证据审阅完成后才能改判。
- qualification 使用缩小预算，所有数值只证明工程路径可运行，不进入三个正式 seed 的 CI。
- ID、OOD、AULC、forgetting、预测因果、预测质量和稀疏收益由后续正式报告与 `GATE-NN-MVP-v2` 裁决，不在本 Gate 宣称通过。
- 本模板不得回填或改写 P3 的 `GATE-NN-MVP FAILED`。若协议、矩阵或阈值改变，应建立新 protocol 和新 Gate 文件。
