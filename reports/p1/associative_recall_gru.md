# P1 Associative Recall GRU 基线报告

- Commit: `259caf1f2bcf7b8f4f027eaba65e70edd0f7978b`
- Seeds: `[17, 29, 43]`
- 性质：P1 描述性单体基线，不构成类脑模型收益结论。

| Seed | Run ID | Test | OOD | Steps | Params | MAC/seq | Coverage | P50/P95 ms | Wall s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | `associative_recall-v1-gru-s17-20260715T104305Z` | 0.164062 | 0.069824 | 5000 | 112032 | 1111040 | 1.000 | 2.397/2.656 | 721.453 |
| 29 | `associative_recall-v1-gru-s29-20260715T105525Z` | 0.166504 | 0.060059 | 3400 | 112032 | 1111040 | 1.000 | 2.314/2.669 | 443.548 |
| 43 | `associative_recall-v1-gru-s43-20260715T110304Z` | 0.167969 | 0.080078 | 4800 | 112032 | 1111040 | 1.000 | 2.330/2.573 | 608.616 |

## Bootstrap 95% CI

- Test query accuracy: 0.166178 [0.156901, 0.175618]
- Test interference drop (low - high): 0.007362 [-0.013266, 0.026708]
- OOD query accuracy: 0.069987 [0.059245, 0.081380]
- OOD interference drop: N/A（缺少对照分层）
- Seed median (test/OOD): 0.166504 / 0.069824
- 方法：10,000 次 seed→stratum/sample 两级 percentile bootstrap；seed=20260715。
- Holm 校正：N/A；当前只有一个描述性基线，没有模型间多重比较。
- 失败/缺失 run：0；所有冻结 seed 均纳入。
