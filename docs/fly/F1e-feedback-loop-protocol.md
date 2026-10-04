---
title: F1e MBON→DAN 反馈回路结构检验协议
status: FROZEN-BEFORE-RUN
change_request: CR-013
date: 2026-10-04
---

# F1e：MBON 是否优先反馈到“教导自己”的 DAN

## 1. 动机

F1b 支持了“预测误差门控可塑性”（DAN = r − v）。要在回路层面实现 r − v，需要 MBON（输出端，代表预测 v）把信号反馈到为**同一 compartment** 提供教学信号的 DAN 上，形成互惠回路（参见 Eschbach et al. 2020，幼虫）。

结构背景（FlyWire 783，右半球）：164/165 个 DAN 接收 MBON 输入；DAN 类型对 MBON 类型的投射有明显偏好，中位纯度为 0.58。本检验只用连接组，不涉及任何学习任务。

## 2. 定义（FlyWire 783，右半球，突触数在各 neuropil 上求和）

- 神经元：`cell_class=DAN` 与 `cell_class=MBON`，`side=right`。
- 教导关系 P：DAN d 到 MBON m 的突触数 ≥ 5 时，记 (d, m) ∈ P，表示 d 支配 m 所在的 compartment。
- 反馈边：MBON m 到 DAN d 的全部突触。
- 统计量 S：落在互惠对上的 MBON→DAN 反馈突触数，占全部 MBON→DAN 反馈突触的比例。互惠对指 (m → d) 满足 (d, m) ∈ P。

## 3. 零模型

- 对 MBON→DAN 二部图做保度重连：双边交换目标 DAN，保持每个 MBON 的出度与每个 DAN 的入度，突触数随边移动；交换次数为 10 × 边数。
- 教导关系 P 保持不变。
- 共 1,000 次实现，零模型 RNG seed 20261004。

## 4. 预注册假说

| ID | 内容 | 通过条件 |
|---|---|---|
| O1 | 反馈富集于互惠对 | S_real 大于零模型分布的 99.9 百分位（单侧 p < 0.001），且 S_real / mean(S_null) ≥ 2 |
| O2 | 互惠反馈偏抑制 | 互惠反馈突触中来自 GABA 或谷氨酸能 MBON 的比例，高于非互惠反馈中的比例；用 1,000 次置换检验（打乱 MBON 的递质标签），单侧 p < 0.05 |

描述性报告：不同阈值（≥ 1、≥ 10）下的 S、各 DAN 簇（PAM、PPL1）分别的 S、互惠对的数量。

## 5. 解释边界

- 结构富集只说明回路具有实现 compartment 特异预测误差的解剖条件，不证明 DAN 的实际活动等于 r − v。
- 谷氨酸在果蝇中通常是抑制性的，但存在受体依赖的例外；O2 采用的符号规则与 F0 相同（GABA 与谷氨酸记为抑制）。
