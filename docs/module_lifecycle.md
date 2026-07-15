---
title: 模块状态生命周期
status: DRAFT
phase: P0
gate: GATE-0
last_updated: 2026-07-15
---

# 模块状态生命周期（DRAFT）

> 本文只定义待验证的生命周期问题和不变量，不冻结状态类型或模块接口。

## 生命周期状态

| 阶段 | 必需行为 | 禁止行为 |
|---|---|---|
| `initialize` | 由显式 device/dtype、batch 元数据创建状态 | 硬编码 MPS/CUDA；读取上个 run 状态 |
| `episode_start` | 清理 episode-local 状态，装载允许的持久状态 | 跨样本泄漏标签、目标或梯度图 |
| `step_read` | 只读当前合法状态，应用 valid mask | 读取未来 observation 或未完成转移 |
| `step_update` | 按冻结时序更新状态并记录事件 | 原地破坏 autograd 所需张量 |
| `detach/truncate` | 按协议截断图并记录边界 | 隐式 detach 或无界计算图增长 |
| `episode_end` | 完成允许的写回、统计与清理 | 未声明地保留 transient state |
| `checkpoint` | 保存可复现恢复所需状态和版本 | 序列化无界 telemetry 或运行时句柄 |
| `restore` | 校验 schema/config 兼容后恢复 | 静默接受不兼容 state |

## 六模块状态草案

| 模块 | 暂态状态 | 可持久状态 | 重置边界 | 必测失败模式 |
|---|---|---|---|---|
| 感觉编码 | 可选短时上下文 | 参数、归一化统计 | batch/episode（待定） | padding 污染、device/dtype 不一致 |
| 情景记忆 | 当前查询与写缓冲 | 允许的 episode 记忆；长期 replay 后置 | episode；任务切换策略待定 | 样本泄漏、容量失控、错误删除 |
| 工作记忆 | 规则、目标、门控与容量槽 | MVP 默认无跨 episode 状态 | episode | 延迟状态丢失、无界保持 |
| 预测适配 | 待匹配的 `(s_t,a_t)` | 参数与可选慢状态 | 转移完成/episode | 使用未来信息、错配 action target |
| 动作选择 | 候选、冲突与策略上下文 | MVP 默认无跨 episode 状态 | step/episode | 候选置乱、无效动作未 mask |
| 稀疏路由 | gate、eligible-set、负载统计 | 参数与校准统计 | step；统计按 phase 隔离 | 必经模块误入分母、路由塌缩 |

## 生命周期验收模板

- [ ] 相同 seed/config/checkpoint 的恢复结果在预注册数值容差内一致。
- [ ] batch 排列、padding 和 episode 重置不会造成状态串扰。
- [ ] 训练与推理模式切换不会悄然改变持久状态语义。
- [ ] 反向图长度受控，无非预期引用和内存增长。
- [ ] 每次 state schema 变更都有版本、迁移或明确拒绝策略。
- [ ] telemetry on/off 不改变状态更新、loss、gradient 或参数更新。
