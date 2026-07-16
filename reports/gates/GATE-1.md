---
gate: GATE-1
status: PASSED
reviewed_commit: 259caf1f2bcf7b8f4f027eaba65e70edd0f7978b
review_date: 2026-07-15
reviewer: independent-verifier-agent
---

# GATE-1：任务基线与训练骨架评审

## 裁决

`PASSED`。P1 的任务、单体基线、训练与恢复、评估统计、运行产物和 CPU/MPS 可移植性证据满足冻结的 GATE-1 验收条件。

本裁决只确认一套可复现的**描述性基线**已经建立。P1 没有实现或比较类脑模块，因此本报告不证明任何类脑收益、生物等价性或相对于单体模型的性能优势。

## 验收矩阵

| 验收项 | 结果 | 独立核验证据 |
|---|---|---|
| 三任务确定性、划分隔离、mask、oracle 与 OOD | PASS | 三任务 fixture/oracle/OOD 测试、split seed 与内容哈希隔离测试通过；无效 padding mask、非连续边界和跨 episode ID 注入均被拒绝。 |
| GRU/Transformer × 三任务 × CPU/MPS | PASS | CPU 全量测试通过；宿主 MPS 定向矩阵覆盖 12 个组合，forward、masked loss、backward 与有限梯度均通过。 |
| 训练收敛 smoke | PASS | 三任务 seed-7 smoke 均运行 200 steps，首 10 步至末 10 步平均 loss 下降，未出现 NaN/Inf。 |
| checkpoint 与确定性恢复 | PASS | CPU 连续/恢复逐位一致；MPS 在 `rtol=1e-5, atol=1e-6` 内一致；配置哈希、model/optimizer、训练游标、sampler、Python/NumPy/Torch CPU/MPS RNG 均核验。 |
| 故障注入与失败证据 | PASS | 非有限 loss/梯度、无效 mask、跨 episode、序列边界与配置不兼容均被捕获；真实 MPS 不支持算子的失败 run 保留 `failed` manifest 和校验和。 |
| 三 seed 正式基线 | PASS | seeds `17/29/43` 全部完成；数据规模、预算、最佳 checkpoint、逐样本记录与产物校验和一致。 |
| 统计协议 | PASS | 从 12,288 条 test/OOD 逐样本记录独立重算 10,000 次 seed→stratum/sample 两级 percentile bootstrap，结果与跟踪 JSON 完全一致。单模型描述性分析没有多主比较，Holm 标记为 N/A。 |
| 参数、计算与运行成本 | PASS | 每条 run 均记录参数量、MAC/operator coverage、训练 steps/examples/tokens、P50/P95 延迟、吞吐、wall-clock、峰值内存及失败状态。 |
| 本地与远程质量门禁 | PASS | Ruff、格式、mypy、pytest、pre-commit、CPU/MPS smoke 通过；GitHub Actions run `29408988494` 的 `quality` job 及所有步骤为 `success`。 |
| 科学边界 | PASS | 报告明确为单体 GRU 描述性基线，不含类脑模型比较、脑区映射收益或生物等价声称。 |

## 正式运行证据

三条正式运行均满足：`status=completed`、`failure=null`、`dirty=false`、commit 为 `259caf1f2bcf7b8f4f027eaba65e70edd0f7978b`；manifest 通过 `run-manifest-v1`，所有登记产物的 SHA-256 均匹配。

| Seed | Run ID | Steps / best step | 首/末 10 步 loss | Test query accuracy | OOD query accuracy | Examples | Tokens | Wall-clock |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | `associative_recall-v1-gru-s17-20260715T104305Z` | 5000 / 4400 | 3.466496 / 2.031020 | 0.164062 | 0.069824 | 320,000 | 2,876,730 | 721.453 s |
| 29 | `associative_recall-v1-gru-s29-20260715T105525Z` | 3400 / 2400 | 3.473048 / 2.270516 | 0.166504 | 0.060059 | 217,600 | 1,956,048 | 443.548 s |
| 43 | `associative_recall-v1-gru-s43-20260715T110304Z` | 4800 / 3800 | 3.466547 / 2.015204 | 0.167969 | 0.080078 | 307,200 | 2,761,505 | 608.616 s |

Seeds 29 与 43 按预注册的 patience 提前停止，均符合“最多 5,000 steps”的预算。每个 run 使用 train `8,192`、validation/test/OOD 各 `2,048` 个冻结样本；test 与 OOD 各保存 2,048 条不重不漏的逐样本记录。三个 run 的参数量均为 `112,032`，估算 MAC/sequence 为 `1,111,040`，受支持算子覆盖率为 `1.0`。

## 统计复验

- Test query accuracy：`0.166178 [0.156901, 0.175618]`，95% CI。
- Test interference drop（low − high）：`0.007362 [-0.013266, 0.026708]`，95% CI；区间跨越 0，不支持确定性干扰效应结论。
- OOD query accuracy：`0.069987 [0.059245, 0.081380]`，95% CI。
- OOD interference drop：N/A；冻结 OOD 分布只有 high-interference 分层，报告已明确缺失原因，没有补造不可识别的对照。
- bootstrap 次数 `10,000`，随机种子 `20260715`，置信度 `0.95`。当前仅有一个描述性模型，没有模型间多重比较，因此 Holm 校正为 N/A。

跟踪汇总位于 `reports/p1/associative_recall_gru.json` 和 `reports/p1/associative_recall_gru.md`。独立 verifier 直接从三条 run 重新载入逐样本记录并调用冻结统计实现，重建的 JSON 与跟踪文件逐字段一致。

## Smoke 与失败链路

| 任务 / 模型 | Clean run | Loss 首/末窗口 | 主要 smoke 结果 |
|---|---|---:|---|
| Associative Recall / GRU | `associative_recall-v1-gru-s7-20260715T104120Z` | 3.464531 / 0.181607 | Test/OOD query accuracy `0.000000 / 0.062500`。 |
| Delayed Rule Switch / Transformer | `delayed_rule_switch-v1-transformer-s7-20260715T104143Z` | 0.615550 / 0.001293 | Test/OOD response accuracy `1.000000 / 0.984375`。 |
| SmallGraph / GRU | `small_graph-v1-gru-s7-20260715T104216Z` | 1.431220 / 0.102203 | Test/OOD rollout success `0.437500 / 0.250000`；next-state accuracy `0.171875 / 0.041667`。 |

三条 clean smoke 均为 seed 7、MPS、200 steps、clean commit `259caf1`，manifest/schema/checksum、best checkpoint、config hash、sampler/RNG、test/OOD 各 32 条记录、10 次 latency warmup 与 50 次采样均通过复验。

失败 run `delayed_rule_switch-v1-transformer-s7-20260715T103946Z` 正确记录了旧 commit `37ca25a` 上 `aten::_nested_tensor_from_mask_left_aligned` 不受 MPS 支持的 `NotImplementedError`。commit `259caf1` 禁用 Transformer nested-tensor 快速路径后，对应 clean run 完成，证明该失败链路已经关闭而非被隐去。

## 质量门禁与 CI

- Ruff check：PASS。
- Ruff format check：PASS，55 files。
- mypy：PASS，55 source files。
- pytest：PASS，`90 passed`；独立 verifier 沙箱内 8 个 MPS case 因设备不可见而 skipped。
- 宿主 MPS 定向复验：PASS，`15 passed, 1 skipped`；唯一 skip 是只适用于无 MPS 主机的反向分支。
- pre-commit：Ruff lint、Ruff format、repository hygiene 全部 PASS。
- `git diff --check`：PASS。
- verifier 使用环境内工具逐项执行了 `make check` 的 lint、format、typecheck、pytest 与 CPU smoke 等价门禁；Make 包装器在 verifier 沙箱内不能解析沙箱外 Conda env 路径，该限制不属于项目失败。
- GitHub Actions：[run 29408988494](https://github.com/liberal-CloudArchitect/neuromorphic_network/actions/runs/29408988494)，commit `259caf1f2bcf7b8f4f027eaba65e70edd0f7978b`，`2026-07-15T10:41:14Z` 至 `2026-07-15T10:43:37Z`，结论 `success`。独立查询确认 job `87331282177` 的环境创建、CPU PyTorch 安装、lint、format、typecheck、test、CPU smoke 与 P1 deterministic training smoke 步骤全部成功。

## 边界与后续动作

- 较低的 Associative Recall 与 OOD accuracy 是已记录的基线结果，不是 GATE-1 的性能收益证明；后续 P2 比较必须沿用相同数据、预算和统计协议。
- P1 不含类脑模块、3D、atlas、WebSocket 或生成模型；不能用本 Gate 对这些能力作任何完成声明。
- `0.2.0` 版本升级、P1 台账关闭与本报告提交属于 GATE-1 通过后的治理动作，不属于本独立裁决的前置证据。
