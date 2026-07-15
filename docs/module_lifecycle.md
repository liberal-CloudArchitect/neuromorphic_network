---
title: 模块状态生命周期
status: ACCEPTED
phase: P0
gate: GATE-0
last_updated: 2026-07-15
---

# 模块注册与状态生命周期

本文冻结模块状态的所有权、重置、detach、checkpoint 与兼容规则。实际类型和验证错误由公共契约实现。

## 注册与兼容

- 注册表只接受六个版本化 ID：`sensory_encoder.v1`、`episodic_memory.v1`、`working_memory.v1`、`predictive_adapter.v1`、`action_selector.v1`、`sparse_router.v1`。
- 相同 ID 的输入、输出、状态语义和 reset 行为不得静默变化；破坏性变化必须新增版本 ID。
- registry 必须拒绝未知、重复和接口验证失败的模块。配置引用的是注册 ID，不是 Python 导入路径。
- P0/P1 不提供跨版本 state 迁移；checkpoint 的模块 ID、state schema 或配置哈希不一致时明确拒绝恢复。
- 参数冻结/解冻只在 optimizer step 边界执行，并重建 optimizer parameter groups；变化写入 run manifest，不允许在 forward 中切换。

## 生命周期状态机

| 阶段 | 必需行为 | 禁止行为 |
|---|---|---|
| `initialize` | 使用显式 batch size、device、dtype 创建版本化空状态 | 硬编码 MPS/CUDA；读取上一 run 状态 |
| `episode_start` | 在当前 step 读取前按 `reset_mask` 清理 item-local 瞬时状态 | 跨样本保留标签、目标、记忆或梯度图 |
| `step_read` | 只读当前合法状态并应用 `valid_mask` | 读取未来 observation、未完成转移目标或 padding |
| `step_update` | 按计算图冻结顺序返回新状态 | 原地破坏 autograd 张量；跨 item 写入 |
| `detach/truncate` | 每 32 个有效 step 保留数值并 detach 历史 | 隐式 detach、无界图增长或把 padding 计入窗口 |
| `episode_end` | 清理记忆、工作状态和悬空预测；完成有界统计 | 未声明保留 transient state |
| `checkpoint` | 保存版本、配置哈希和恢复所需张量/RNG/游标 | 序列化运行时句柄或无界 telemetry |
| `restore` | 先验证 schema、模块 ID、shape、dtype/device 策略和 config hash | 静默迁移、部分加载后继续训练 |

## 模块状态所有权

| 模块 | 瞬时状态 | 持久内容 | reset/detach | 必测失败模式 |
|---|---|---|---|---|
| 感觉编码 | P1 无；未来可有短时上下文 | 参数、显式归一化统计 | episode reset；32 step detach | padding 污染、device/dtype 不一致 |
| 情景记忆 | item-local slots、占用 mask、写指针 | P0/P1 仅参数，无跨 episode 记忆 | episode 全清；32 step detach | 当前事件自命中、跨样本泄漏、容量越界 |
| 工作记忆 | 规则/目标槽、门值 | P0/P1 仅参数 | episode 全清；32 step detach | 未来标签门控、无界保持、reset 失效 |
| 预测适配 | 未完成 `(s_t,a_t)` | 参数 | 转移完成消费；episode 丢弃；32 step detach | action/target 错配、用到未来信息 |
| 动作选择 | P1 无；候选只活到当前 step | 参数 | step 结束释放 | 无效动作未 mask、候选次序改变语义 |
| 稀疏路由 | 当前 gate/mask | 参数与显式 phase-local 聚合统计 | step 结束；phase 切换清统计 | 必经模块计入稀疏率、top-k 越界、塌缩 |

每个 state tensor 只能由所属模块更新；编排器可重置、detach、序列化和搬移 device，但不能解释或修改模块私有字段。

## mask 与边界语义

- `valid_mask=false`：不读写状态、不累计 loss、不推进 TBPTT 计数；输出位置可为零但不得影响有效位置。
- `reset_mask=true`：先清零再处理该 step，因此该 step 是新 episode 的首个合法输入。
- 同一 batch item 的 episode ID 变化必须伴随 reset；反之视为数据错误并在 forward 前拒绝。
- batch 重排必须同步重排所有 state tensor；不允许按隐式全局顺序寻址。

## checkpoint 与恢复

checkpoint 必须包含模型/优化器/scheduler、模块状态版本、训练游标、最佳指标、Python/NumPy/Torch RNG、任务 sampler 状态和配置哈希。CPU 连续与恢复训练要求逐位一致；MPS 参数/状态使用 `rtol=1e-5, atol=1e-6`，恢复前后主指标差不超过 `1e-4`。不兼容恢复必须在任何 optimizer update 前失败。

## 验收不变量

- 相同 seed/config/checkpoint 的恢复满足预注册数值要求。
- batch 重排、padding、episode reset 和 TBPTT 不产生状态串扰。
- train/eval 切换不改变瞬时状态语义；validation/test/OOD 不写训练统计。
- telemetry on/off 不改变 packet、state、loss、gradient 或 optimizer update。
- unknown/duplicate registry ID、state 版本不匹配和部分 checkpoint 均被明确拒绝。
