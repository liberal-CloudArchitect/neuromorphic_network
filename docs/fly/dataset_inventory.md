# 果蝇（Drosophila）连接组 / 脑数据集下载清单（仅清单，未下载任何数据）

- 编制日期：2026-10-04
- 范围：F0（加载 FlyWire 全脑连接组 + 复现 Shiu 2024 LIF 验证）、F1（蘑菇体持续学习、惊奇/多巴胺门控学习、中央复合体环形吸引子）及后续可选资源。
- 方法：只读取了网页、README、API 元数据（GitHub REST、Zenodo REST、Harvard/Edmond Dataverse API、GCS JSON listing、PyPI JSON）和小型源码文件；对数据文件只发了 `HEAD` 请求。没有下载任何 parquet/csv/feather/zip 数据，没有登录，没有使用任何 token 或 API key。
- 机器可读清单（543 行，每个文件一行）：`/private/tmp/claude-501/-Volumes-Samsung-neuromorphic/22c09952-2c9b-4f99-8613-bcf6f3447218/scratchpad/fly_datasets.json`，字段为 `dataset, version, tier, file_name, url, size_bytes, checksum_algo, checksum, license, access, notes`。
- 分级：**T0** = F0 必需；**T1** = F1 环路研究必需；**T2** = 可选或以后再用。
- 体积单位：MB = 10^6 字节，GB = 10^9 字节。

---

## 0. 结论速览

| 推荐集 | 文件数 | 合计字节 | 约合 |
|---|---:|---:|---:|
| T0-min（只跑 Shiu v783 模型） | 2 | 104,131,989 | 104.1 MB |
| T0-paper（再加论文实际使用的 v630 数据 + 2 个参考输出） | 6 | 195,326,919 | 195.3 MB |
| **T0 完整推荐** | 12 | **1,064,132,991** | **1.06 GB** |
| **T1 推荐** | 2 | **91,464,363** | **91.5 MB** |
| **T0 + T1** | 14 | **1,155,597,354** | **1.16 GB** |

/Volumes/Samsung 有 668 GB 可用，T0+T1 只占约 0.17%。下面标为 LARGE 的条目（EM 影像、网格、骨架、全突触表、BANC influence）全部不推荐下载。

---

## 1. 总表

| 数据集 | 版本（已核实） | 主要文献 | 访问方式 | 许可 | 规模（公开可下载部分） | 分级 |
|---|---|---|---|---|---|---|
| FlyWire FAFB 全脑（Zenodo 连接数据） | 783.0（Zenodo 上没有更新版本） | Dorkenwald 2024 Nature；Buhmann 2021；Eckstein 2024 Cell（NT） | Zenodo 直链 | CC-BY-4.0 | 5 个文件，10.60 GB | T0（2 个文件）/ T2 |
| FlyWire Codex 导出（FAFB 783） | 783；GCS 对象在 2024-12 到 2026-02 间有原地更新 | 同上 + Schlegel 2024 | 公共 GCS 对象，无需认证；Codex 网页需要 Google 登录 | CC-BY-4.0（FlyWire） | 40 个表，4.69 GB | T0（3 个小表）/ T2 |
| flywire_annotations | v3.2.0（2026-09-29，最新）；论文对应版本 v2.1.0 | Schlegel 2024；Berg 2026；Tastekin 2026；Matsliah 2024；Dorkenwald 2024 | Zenodo / GitHub | CC-BY-4.0（Zenodo 元数据；GitHub 仓库无 LICENSE 文件） | zip 12.2 MB | T0 |
| Shiu 2024 LIF 模型仓库 | GitHub `main`@`91bdd1e`（2024-09-14） | Shiu et al. 2024 Nature 634:210 | GitHub raw | MIT | 数据文件约 197 MB | T0 |
| Shiu 2024 模拟结果（Edmond） | Dataverse v3（2023-09-20） | 同上 | Edmond 直链 | MIT | results.zip 4.50 GB | T2 |
| eonsystemspbc/fly-brain（PyTorch 等移植） | `main`@`a3db62f`（2026-08-29） | — | GitHub | **GPL-2.0** | 数据与 Shiu 相同 | T2（仅作参考） |
| BANC（脑 + 神经索） | v888；Dataverse v3.0（2026-07-01） | Bates, Phelps, Kim, Yang et al. **Nature 2026**，doi:10.1038/s41586-026-10735-w | Harvard Dataverse | CC-BY-4.0 | 379 个文件，536.06 GB（其中 277 个 influence 分块共 308.26 GB 需申请访问） | T2 |
| hemibrain | v1.2 批量导出；neuPrint 上为 `hemibrain:v1.2.1` | Scheffer 2020 eLife；Li 2020 eLife（MB）；Hulse 2021 eLife（CX） | 公共 GCS `gs://hemibrain` | CC-BY-4.0 | v1.2 共 12 个文件，13.0 GB | T1（2 个文件）/ T2 |
| Male CNS（MCNS） | **v1.0**（neuPrint `male-cns:v1.0`，2026-06-08 更新） | Berg et al. 2026 Cell 189:5504 | 公共 GCS `gs://flyem-male-cns` | CC-BY-4.0 | flat-connectome 11 个文件，31.32 GB | T2 |
| MANC | neuPrint `manc:v1.2.1` / `v1.2.3`；公共批量导出只有 v1.0 | Takemura 2024；Marin 2024；Cheong 2024（eLife） | `gs://flyem-manc-exports`（v1.0）；Codex GCS（v1.2.1） | CC-BY-4.0 | 6.66 GB（v1.0） | T2 |
| Male optic lobe | v1.1 | Nern et al. 2025 Nature | `gs://flyem-optic-lobe` | 按 FlyEM 惯例为 CC-BY（未逐页核实） | 3.47 GB | T2 |
| FANC | CAVE materialization v1444（2025-10-09）与 v1237 | Azevedo et al. 2024 Nature | 公共 GCS `gs://lee-lab_female-adult-nerve-cord` | **桶内未声明，待核实** | v1444 共 0.99 GB | T2 |
| 幼虫全脑（Winding 2023） | Science 2023 | Winding, Pedigo et al. 2023 Science 379:eadd9330 | PMC 补充材料（脚本访问被拦截） | CC-BY（Europe PMC 作者稿） | 约 1 MB zip + 3 个 csv | T2 |
| flyvis | PyPI 1.2.0（2026-08-06） | Lappalainen et al. 2024 Nature | pip + Google Drive | MIT | wheel 386 KB；预训练 zip 大小未知 | T2 |
| MB / CX 专题 | hemibrain 内含；Caron 2013 补充 PDF | Li 2020；Hulse 2021；Caron 2013 Nature | 见上 | — | — | T1 / T2 |
| 身体仿真 | flygym 2.1.0；flybody `main` | Wang-Chen 2024（NMF v2）；Vaxenburg 2025 | pip / git | Apache-2.0 | 3.2 MB / 约 90 MB | T2 |
| 功能数据 | 见第 11 节 | — | Zenodo | CC-BY-4.0 | 15.6 MB 到 48.8 GB | T2 |

---

## 2. 推荐下载集

### 2.1 T0（F0：加载 FlyWire 全脑连接组 + 复现 Shiu 2024）

| # | 文件 | 字节 | 约合 | 校验 | URL |
|---|---|---:|---:|---|---|
| 1 | `Connectivity_783.parquet` | 100,804,642 | 100.8 MB | git-blob-sha1 `d386555d1a5f40ebfa1380bcb05b1fab044855fd` | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Connectivity_783.parquet |
| 2 | `Completeness_783.csv` | 3,327,347 | 3.3 MB | git-blob-sha1 `b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b` | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Completeness_783.csv |
| 3 | `2023_03_23_connectivity_630_final.parquet` | 86,630,944 | 86.6 MB | md5 `3b42c0fc2cc525bd9784e38c05f9d79d`（Edmond） | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/2023_03_23_connectivity_630_final.parquet |
| 4 | `2023_03_23_completeness_630_final.csv` | 3,057,611 | 3.1 MB | md5 `0dc6b58650bd3cd3a2661773d9228e78`（Edmond） | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/2023_03_23_completeness_630_final.csv |
| 5 | `results/example/sugarR.parquet` | 948,526 | 0.9 MB | git-blob-sha1 `1204cb3ed453ad03b0141028fa511d924f99d599` | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/results/example/sugarR.parquet |
| 6 | `results/example/sugarR_100Hz.parquet` | 557,849 | 0.6 MB | git-blob-sha1 `bd76730d5e991d64b756f354a749f94ffb874c64` | https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/results/example/sugarR_100Hz.parquet |
| 7 | `proofread_connections_783.feather` | 852,022,274 | 852.0 MB | md5 `f48f972d262323a102aed49af1396b8a` | https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1 |
| 8 | `proofread_root_ids_783.npy` | 1,114,168 | 1.1 MB | md5 `e0e6c19732fd8c7a4e39a2d170105421` | https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1 |
| 9 | `flywire_annotations-v3.2.0.zip` | 12,153,637 | 12.2 MB | md5 `4bdf63184d35177e56f88de7b3d400ae` | https://zenodo.org/records/23037340/files/flyconnectome/flywire_annotations-v3.2.0.zip?download=1 |
| 10 | `neurons.csv.gz`（Codex FAFB 783） | 1,679,884 | 1.7 MB | md5 `f60333e9e4124160b9b203b1712a6f91` | https://storage.googleapis.com/flywire-data/codex/data/fafb/783/neurons.csv.gz |
| 11 | `classification.csv.gz`（Codex FAFB 783） | 934,402 | 0.9 MB | md5 `701c0faf054ceddb460bea4e87d6a624` | https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz |
| 12 | `consolidated_cell_types.csv.gz`（Codex FAFB 783） | 901,707 | 0.9 MB | md5 `18cc628156bede3129c0e19d14cf52d6` | https://storage.googleapis.com/flywire-data/codex/data/fafb/783/consolidated_cell_types.csv.gz |
| | **合计** | **1,064,132,991** | **1.06 GB** | | |

每个文件的用途：
- 第 1、2 项：Shiu 模型直接读取的 v783 权重和神经元表。它们本身就是“全脑 FlyWire 连接组（带符号）”的最小可运行形态。
- 第 3、4 项：论文实际使用的 v630 数据。example/figures notebook 里的神经元 ID 都是 v630 ID，所以严格复现论文要用这两项。
- 第 5、6 项：仓库自带的参考输出（v630，30 trial × 1 s），可以直接作为回归测试的期望值。
- 第 7、8 项：官方、可引用、带 md5 的 proofread 边表（按神经元对 × neuropil 一行，包含六种 NT 平均概率）。用它可以独立重建并核对 Shiu 的符号权重，也可以按 neuropil 切出 MB 和 CX 子图。
- 第 9 项：系统化的细胞类型、superclass、`top_nt` 等注释，同时包含 `Supplemental_file5_hemibrain_meta.csv`。
- 第 10 到 12 项：Codex 的逐神经元 NT 预测（`nt_type`、`*_avg`）、分类和细胞类型，体积小，便于映射。

校验说明：
- 仓库的 `neuromorphic.fly.datasets.DatasetFile` 只接受 `md5` 和 `sha256`。GitHub 文件只公布了 git blob SHA1；raw 的 ETag 经实测**不是**内容的 sha256。
- 建议做法：第 1、2、5、6 项首次下载后用 `git hash-object` 核对 blob SHA1，核对通过后再计算 sha256 写入 manifest。
- 第 3、4 项也可以直接用 Edmond 公布的 md5 校验：Edmond 与 GitHub 上的文件字节数一致，推定为同一文件。

### 2.2 T1（F1：MB 持续学习 / 多巴胺门控 / CX 环形吸引子）

| # | 文件 | 字节 | 约合 | 校验 | URL |
|---|---|---:|---:|---|---|
| 1 | `exported-traced-adjacencies-v1.2.tar.gz`（hemibrain） | 45,872,577 | 45.9 MB | md5 `ec1e076d15442f368fa5978b00f4a61c` | https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz |
| 2 | `hemibrain-v1.2-body-mean-neurotransmitters.feather` | 45,591,786 | 45.6 MB | md5 `ef5d5d511d19ed9bca5d8dc5c8cca690` | https://storage.googleapis.com/hemibrain/v1.2/hemibrain-v1.2-body-mean-neurotransmitters.feather |
| | **合计** | **91,464,363** | **91.5 MB** | | |

为什么这样就够：
- FAFB 侧（双侧 MB、CX，包括 KC、MBON、DAN（PAM/PPL1）、APL、DPM、EPG、PEN、PEG、Δ7 等）可以直接由 T0 的 `proofread_connections_783.feather` 加 `flywire_annotations` 按 cell_type 和 neuropil 切出来，不需要再下载别的文件。
- hemibrain 是 Li 2020（MB）和 Hulse 2021（CX）两篇专题论文的底层数据。tar 包里有 `traced-neurons.csv`（bodyId, type, instance）、`traced-total-connections.csv`（bodyId_pre, bodyId_post, weight）和 `traced-roi-connections.csv`。这三个文件名来自 Janelia 历来的导出格式，本次没有解包核实。
- `flywire_annotations` 的 `Supplemental_file5_hemibrain_meta.csv` 已经在 T0 的 zip 里，可以做 FAFB 与 hemibrain 的类型对照。

T1 候选（暂不推荐，需要时再加）：
- Male CNS v1.0：`body-annotations-male-cns-v1.0-minconf-0.5.feather`（14,483,314 B）、`body-neurotransmitters-male-cns-v1.0.feather`（43,282,834 B）、`connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`（508,025,642 B）。合计 565.8 MB，用于第三个连接组上的跨个体复核。
- 同一桶里还有 `hemibrain-v1.2-all-traced-adjacencies.tar.gz`（80,815,554 B，2026-05-15 更新）。内容未核实，可能是更新或更全的 traced 导出，下载前先看一下。

---

## 3. FlyWire FAFB（v783）

### 3.1 版本
- 公开 materialization 只有 630 和 **783**。Codex 首页显示的 “FAFB v783 (CB): 139,255 neurons, 3,732,460 connections” 是 ≥5 突触阈值下的数。Zenodo 记录 10676866 只有 `783.0` 一个版本。
- 没有比 783 更新的公开 materialization。783 之后的变化有两类：
  - Codex 在 2025-07 新增了 **Princeton 新突触检测**（`connections_princeton*.csv.gz` 和 `fafb_v783_princeton_synapse_table.csv.gz`），神经元 ID 仍是 783；
  - `flywire_annotations` 在继续更新注释（v3.0.0 到 v3.2.0）。
- `connections.csv.gz` 与 `connections_buhmann.csv.gz` 的 md5 相同，所以 Codex 默认表仍基于 Buhmann 突触。

### 3.2 Zenodo 10676866：“FlyWire Whole-brain Connectome Connectivity Data”（CC-BY-4.0，2024-06-02）

| 文件 | 字节 | md5 | 说明 | 分级 |
|---|---:|---|---|---|
| `proofread_connections_783.feather` | 852,022,274 | f48f972d262323a102aed49af1396b8a | 列：pre_pt_root_id, post_pt_root_id, neuropil, syn_count, gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg | T0 |
| `proofread_root_ids_783.npy` | 1,114,168 | e0e6c19732fd8c7a4e39a2d170105421 | 全部 proofread root ID | T0 |
| `per_neuron_neuropil_count_pre_783.feather` | 16,853,770 | 90fcdb42c1ba05ed92820840fa1e6ba0 | 每个 segment 在每个 neuropil 的前突触数 | T2 |
| `per_neuron_neuropil_count_post_783.feather` | 233,843,050 | bb5999f10920ade803d9f37097a43a56 | 后突触数 | T2 |
| `flywire_synapses_783.feather` | 9,492,998,242 | f8f1b97c9d4b0ea9b4c8b287f6b99091 | **LARGE**：约 1.3 亿个突触（cleft_score≥50），含坐标与逐突触 NT 概率 | T2，不推荐 |

### 3.3 Codex FAFB 783 导出（`gs://flywire-data/codex/data/fafb/783/`）
- 访问：对象是公开的，HEAD 返回 200，不需要认证。但 Codex 网页和下载页要求 Google 登录，所以这些对象作为“非官方入口”记录，正式引用以 Zenodo 为准。对象会被原地覆盖（例如 `classification.csv.gz` 在 2026-02-24 更新过），**必须用 md5 锁定版本**。
- 列名来自 murthylab/codex 源码：
  - `connections*.csv.gz`：`pre_root_id, post_root_id, neuropil, syn_count, nt_type`，默认阈值为 ≥5 个突触；
  - `neurons.csv.gz`：`root_id, group, nt_type, nt_type_score, ach_avg, gaba_avg, glut_avg, oct_avg, ser_avg, da_avg`；
  - `classification.csv.gz`：`root_id, flow, super_class, class, sub_class, hemilineage, side, nerve …`。
- 主要文件（全部 40 个见 JSON）：

| 文件 | 字节 | 备注 |
|---|---:|---|
| neurons.csv.gz | 1,679,884 | T0 |
| classification.csv.gz | 934,402 | T0 |
| consolidated_cell_types.csv.gz | 901,707 | T0 |
| connections.csv.gz | 50,289,304 | ≥5 突触，含 nt_type；可作为 proofread_connections 的轻量替代 |
| connections_no_threshold.csv.gz | 212,093,967 | 不设阈值 |
| connections_princeton.csv.gz | 68,456,801 | 新突触检测，≥5 |
| connections_princeton_no_threshold.csv.gz | 275,679,780 | |
| labels.csv.gz / names.csv.gz / coordinates.csv.gz | 4,771,292 / 1,181,576 / 5,314,546 | |
| cell_stats.csv.gz / neuropil_synapse_table.csv.gz | 2,526,548 / 4,674,663 | |
| visual_neuron_types.csv.gz / column_assignment.csv.gz | 631,701 / 462,838 | 视叶类型与柱 |
| synapse_coordinates.csv.gz | 316,819,225 | 大 |
| fafb_v783_princeton_synapse_table.csv.gz | 2,695,106,039 | **LARGE** |

另外有 26 个 `neuron_db*.pickle.gz`（每个约 75 到 142 MB），是 Codex 应用内部的 pickle 缓存。它们不是公开数据产品，而且反序列化不可信 pickle 有执行代码的风险，已排除在清单之外。

### 3.4 flywire_annotations（GitHub `flyconnectome/flywire_annotations`）
- 最新版 **v3.2.0**（2026-09-29，commit `a83b2776`），Zenodo 10.5281/zenodo.23037340，zip 12,153,637 B，md5 `4bdf63184d35177e56f88de7b3d400ae`。
- `Supplemental_file1_neuron_annotations.tsv` 解压后为 31,720,298 B。主要列：root_id、flow、super_class、cell_class、cell_sub_class、cell_type、hemibrain_type、top_nt、top_nt_conf、side、fbbt_id、status、fru_dsx、dimorphism、supertype 等。
- 论文对应版本 **v2.1.0** 的 `Supplemental_file1` 为 27,015,208 B（git blob `1a3168731618ee62a47392252d3af7664e739e9e`），已作为 T2 写进 JSON。
- ⚠️ **v3.2.0 的重要警告**：唇瓣味觉 GRN 按 Tastekin 2026 重新分型。旧的 `sugar/water` 亚类同时包含水敏 LB3a 和高盐/重金属 LB3d 神经元。如果按新注释选“糖 GRN”，结果会和 Shiu 的 ID 列表不同。**复现 Shiu 时直接使用论文或代码里的 ID 列表**，不要用注释重新挑选。
- README 中 file3 和 file4 的文件名与仓库里的实际文件名对调了（实际是 `Supplemental_file3_summary_with_ngl_links.csv` 和 `Supplemental_file4_hemilineages_clustering.csv`）。
- 引用要求（≥3.2.0）：Berg 2026、Schlegel 2024、Tastekin 2026、Matsliah 2024、Dorkenwald 2024。

### 3.5 其他 FlyWire 相关 Zenodo（T2）
- 10877326，Schlegel 2024 补充文件，CC-BY-4.0：
  - `sk_lod1_783_healed_ds2.parquet`（5,355,543,468 B，**LARGE**，骨架）；
  - `nblast_flywire_all_right_aba_comp.feather`（809,046,036 B）；
  - `nblast_flywire_hemibrain_min_comp.feather`（212,095,362 B）；
  - `nblast_flywire_mirrored_hemibrain_min_comp.feather`（223,176,394 B）。
- 12588557，FlyWire Information Flow Ranks：17 个 feather，每个约 1.4 到 1.6 MB，共 25.3 MB。
- 22795740，FlyWire NT 符号确定性：**单作者社区记录，未经同行评审**，3 个文件共 15.6 MB，仅可用于符号不确定性的敏感性分析。

---

## 4. Shiu et al. 2024（Nature）全脑 LIF 模型

- 论文：Shiu PK, Sterne GR, Spiller N, et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature 634, 210–219 (2024)，doi:10.1038/s41586-024-07763-9（PMC11446845，CC BY）。
- 代码：https://github.com/philshiu/Drosophila_brain_model（MIT），锁定 commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`（2024-09-14，“Added new connectivity file for Flywire version 783”在其前两个提交）。
- 原始模拟输出：Edmond doi:10.17617/3.CZODIW（MIT），`results.zip` 4,499,610,373 B，md5 `f6f2e314821fa6a4196214e3b2b14bc4`（T2，**LARGE**）。

### 4.1 仓库数据文件（全部为 GitHub 普通 blob，不是 LFS）

| 文件 | 字节 | git blob SHA1 | Edmond md5 | 分级 |
|---|---:|---|---|---|
| Connectivity_783.parquet | 100,804,642 | d386555d1a5f40ebfa1380bcb05b1fab044855fd | — | T0 |
| Completeness_783.csv | 3,327,347 | b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b | — | T0 |
| 2023_03_23_connectivity_630_final.parquet | 86,630,944 | 8b4d9531bc0acbda7c2074ae577e6ac9e2fca166 | 3b42c0fc2cc525bd9784e38c05f9d79d | T0 |
| 2023_03_23_completeness_630_final.csv | 3,057,611 | be745f0ce054308df21accc5c4b3883aa38498f9 | 0dc6b58650bd3cd3a2661773d9228e78 | T0 |
| results/example/sugarR.parquet | 948,526 | 1204cb3ed453ad03b0141028fa511d924f99d599 | — | T0（参考输出） |
| results/example/sugarR_100Hz.parquet | 557,849 | bd76730d5e991d64b756f354a749f94ffb874c64 | — | T0（参考输出） |
| results/example/sugarR-720575940617937543.parquet | 537,629 | 1cdeee4a3d4aa5b16602ac0a1e76f1db14e19f9b | — | T2 |
| results/example/sugarR-720575940621754367.parquet | 553,968 | 17942838fe14639269fdff3636ebadbc13645436 | — | T2 |
| results/example/sugarR-720575940622695448.parquet | 578,825 | b4524c7d29476f0816b5162b8ebb4839d4e040fc | — | T2 |
| sez_neurons.pickle | 5,114 | 7fdfc5d9fe43789468d349d86d3f23df068962ed | ae0dc9ed0da47af67c8995ef6e57713f | T2（pickle，见安全提示） |

文件内容：
- `Connectivity_*.parquet`：model.py 用到的列是 `Presynaptic_Index`、`Postsynaptic_Index`、`Excitatory x Connectivity`。最后一列是带符号的突触数，兴奋记 +1，抑制记 −1。其余列未核实，因为不能下载数据。
- `Completeness_*.csv`：第 0 列（index）是 FlyWire root_id，**行号就是 Brian 神经元索引**。
- 行数推断（未直接核实）：eonsystems 副本是 CRLF 换行，比 Shiu 副本多出的字节数正好等于行数（v630 多 127,401 B，等于论文的 127,400 个神经元加 1 行表头）。按同样的差值，v783 的 `Completeness_783.csv` 应有 **138,639** 个神经元。

### 4.2 默认参数（model.py 中 `default_params`，逐字提取）

| 参数 | 代码键 | 值 | 来源注释 |
|---|---|---|---|
| 静息电位 v_0 | `v_0` | −52 mV | Kakaria & de Bivort 2017 |
| 复位电位 | `v_rst` | −52 mV | 同上 |
| 阈值 | `v_th` | −45 mV | 同上 |
| 膜时间常数 | `t_mbr` | 20 ms | C·R = 2 µF/cm² × 10 kΩ·cm² |
| 突触时间常数 τ | `tau` | 5 ms | Jürgensen et al. 2021 |
| 不应期 | `t_rfc` | 2.2 ms | Lazar et al. 2021 |
| 突触延迟 | `t_dly` | 1.8 ms | Paul et al. 2015 |
| 每突触权重 W_syn（唯一自由参数） | `w_syn` | 0.275 mV | 标定方法：100 Hz 糖 GRN 刺激时 MN9 达到最大放电率的约 80% |
| Poisson 速率 | `r_poi` | 150 Hz（代码默认值） | example.ipynb 文字写“默认 200 Hz”，与代码不一致 |
| 第二组 Poisson 速率 | `r_poi2` | 0 Hz | |
| Poisson 权重倍数 | `f_poi` | 250 | 每个 Poisson 事件给 v 加 0.275 × 250 = **68.75 mV**，必然越过 7 mV 的阈值差，即一个输入事件引发一个尖峰 |
| 时间步 dt | 未设置 | **0.1 ms**（Brian2 `defaultclock.dt` 默认值） | PyTorch 移植里写死 `DT = 0.1`，注释为 “matches Brian2 defaultclock.dt” |
| 单次时长 | `t_run` | 1000 ms | |
| trial 数 | `n_run` | 30 | |
| 积分方法 | `method='linear'` | 精确线性积分 | |

### 4.3 方程（逐字）

```
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
threshold: v > v_th
reset:     v = v_rst; w = 0; g = 0 * mV
refractory: rfc        (所有神经元 rfc = 2.2 ms；被 Poisson 驱动的神经元设为 0 ms)
Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly)
w_ij = (Excitatory x Connectivity)_ij * w_syn
PoissonInput(target=neu[i], target_var='v', N=1, rate=r_poi, weight=w_syn*f_poi)
沉默（silence）：把被沉默神经元的所有出突触 w 置 0
```

- 符号规则（论文 Methods）：每个前突触位点取 cleft_score≥50 时置信度最高的 NT。如果一个神经元超过一半的前突触被预测为 GABA 或 Glu，就把它记为抑制性；DA、OA、5-HT 都记为兴奋性。每个神经元只有一个符号。
- 论文把这一模型描述为 α 突触：g 先按单指数衰减，再驱动 v，合起来得到 α 形状的 PSP。
- 论文用的是 v630 的 127,400 个 proofread 神经元。代码 README 说明，把 `config` 改成 `Completeness_783.csv` 和 `Connectivity_783.parquet` 就可以跑 v783。

### 4.4 一个可复现的验证实验：糖 GRN → MN9（喙伸展 / 进食起始）

- 刺激集合：21 个唇瓣糖敏 GRN。以下是 example.ipynb 与 figures.ipynb 中的 **v630 ID**：
  ```
  720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
  720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
  720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
  720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
  720575940611875570
  ```
- v783 版本：eonsystemspbc/fly-brain 的 `code/benchmark.py` 用于 v783 的同一组 21 个 ID 中，只有第 14 个不同，`720575940620900446` 换成了 **`720575940639259967`**，其余 20 个相同。
- 读出神经元：**MN9 = `720575940660219265`**。代码注释称其为 “left”；Fig. 1F 中用作对侧 MN9。另一侧 MN9 在 v630 中为 `720575940645521262`，在 v783 中为 `720575940618238523`（来自 eon 的 v783 example.ipynb）。
- 左右标注不一致：论文正文写的是“unilateral **left** hemisphere activation”，代码注释却写 “right hemisphere”。这很可能是 FlyWire 早期左右约定翻转造成的。**以 ID 为准**。
- 规程：`run_exp('sugarR', neu_exc=neu_sugar, ...)`，30 trial × 1 s。Fig. 1D 把 `r_poi` 从 10 扫到 200 Hz，用 `utils.get_rate` 把尖峰换算成每个神经元的平均放电率 ± 标准差。
- 预期结果（论文原文）：
  1. 激活糖 GRN 会激活多个进食相关的喙运动神经元：MN6、MN8、MN9、MN11。
  2. 糖 GRN 以 100 Hz 驱动时，**100% 的模拟都出现 MN9 激活**；随机打乱连接权重（保持权重分布）后，100 次模拟中只有 1 次激活（Supplementary Table 1d）。
  3. W_syn 的标定点：100 Hz 糖刺激时 MN9 约为其最大放电率的 80%。
  4. MN9 放电率随刺激频率单调上升；同时激活苦味 GRN（`r_poi2`）会压低 MN9（Fig. 3A）；逐个沉默 top-200 下游神经元可以找出 MN9 激活所必需的神经元（Fig. 1F，判据为 MN9 降到对照的 80% 以下）。
  5. 整体准确率：论文能实验检验的 164 条预测中，91% 与实验一致（Supplementary Table 9）。
- 具体的 MN9 Hz 数值没有从论文正文中抽出。建议把 T0 中的 `sugarR_100Hz.parquet` 和 `sugarR.parquet`（v630 的仓库参考输出）作为数值回归目标：对 `flywire_id == 720575940660219265` 统计每个 trial 的尖峰数再求均值。

### 4.5 PyTorch 移植：`eonsystemspbc/fly-brain`
- 许可是 **GPL-2.0**，属于 copyleft。**不要把其代码复制进本专有仓库**，只能作为对照参考（独立重写可以，但不能照抄）。
- `code/run_pytorch.py` 的参数与 Brian2 一致：tauSyn 5、tDelay 1.8、v0/vReset/vRest −52、vThreshold −45、tauMem 20、tRefrac 2.2、scalePoisson 250、wScale 0.275，DT = 0.1 ms。
- 实现方式：稀疏矩阵乘法，延迟用环形缓冲；Poisson 输入是 `bernoulli(rate*dt)*250*0.275` 直接加到 v 上；被刺激的神经元不应期为 0。
- 也提供 GeNN、NEST GPU、Brian2CUDA 后端，以 Brian2 CPU 结果作为基准真值。
- 数据：`data/2025_Connectivity_783.parquet` 与 Shiu 的 `Connectivity_783.parquet` **是同一个 git blob**，不必重复下载。`data/2025_Completeness_783.csv`（3,465,987 B）是 CRLF 换行的版本。
- 该仓库的 `nature_2026_07` 基准尖峰数据放在 Google Drive 文件夹里，大小未知（T2，不需要）。

---

## 5. BANC（Brain And Nerve Cord）

- 文献：Bates, Phelps, Kim, Yang et al. *Distributed control circuits across a brain-and-cord connectome.* **Nature 2026**（任务描述写的是 2025，实际正式发表在 2026），doi:10.1038/s41586-026-10735-w。
- 数据：Harvard Dataverse doi:10.7910/DVN/7WTH1N，Dataverse 版本 3.0（2026-07-01），**CC-BY-4.0**，需同时引用论文和 Dataverse。
- 规模：Codex 显示 BANC v888 有 158,262 个神经元、3,037,361 条连接。BANC README 称约 18.8 万个神经元、1.99 亿个预测突触（口径不同）。
- 访问：379 个文件共 536.06 GB。下载地址为 `https://dataverse.harvard.edu/api/access/datafile/{id}`，会 303 跳转到 S3。
- `influence/all_to_all/chunk_0001..0277.parquet`（277 个文件，共 308.26 GB）标记为 **restricted**，需要在 Dataverse 上提交文件访问申请。
- 推荐的 BANC 入口（全部 T2；如果以后做脑-VNC 下行控制，可以升到 T1）：

| 文件（Dataverse 路径） | 字节 | md5 |
|---|---:|---|
| compiled_data/banc_888_meta.feather | 57,550,610 | 见 JSON |
| compiled_data/banc_888_edgelist_simple_v3.feather | 359,161,658 | 见 JSON |
| compiled_data/banc_888_edgelist_simple_v2.feather | 305,250,378 | 见 JSON |
| compiled_data/banc_888_neurotransmitter_prediction_v2.csv | 21,107,592 | 见 JSON |
| compiled_data/banc_888_metrics.feather | 9,681,618 | 见 JSON |
| annotations/v888/cell_info.parquet | 12,315,918 | 见 JSON |
| annotations/v888/codex_annotations.parquet | 74,350,281 | 见 JSON |

- **LARGE，不推荐**：
  - `banc_888_synapses_v3_enriched.parquet`（19.73 GB）、`..._v2_enriched.parquet`（17.15 GB）；
  - `synapses/raw/*human_readable.csv.gz`（12.26 GB 和 15.56 GB）；
  - `skeletons/banc_swc_skeletons.zip`（16.58 GB）；
  - `nblast/*verified_png_matches.zip`（最大 45.48 GB）；
  - `microCT`（1.31 GB）；
  - 线粒体表（1.11 GB）；
  - influence 分块（308 GB）。
- 更轻的替代：Codex GCS 上的 `codex/data/banc/888/neurons.csv.gz`（2,881,728 B）和 `connections_princeton.csv.gz`（29,164,674 B），都是公开对象。

---

## 6. Janelia / neuPrint 数据集

neuPrint 公共元数据接口 `https://neuprint.janelia.org/api/dbmeta/datasets` 不需要 token，当前列出的数据集有：`hemibrain:v1.2.1`、`male-cns:v0.9`、**`male-cns:v1.0`**、`manc:v1.0`、`manc:v1.2.1`、`manc:v1.2.3`（2024-08-31 更新了 segment property）、`mushroombody`（2017）、`optic-lobe:v1.0.1`、`optic-lobe:v1.1`。

**neuPrint 的查询 API 需要个人 token（Google 登录获取）**，本次没有使用。以下都是不需要 token 的公共批量导出。

### 6.1 hemibrain（`gs://hemibrain/v1.2/`，CC-BY-4.0）

| 文件 | 字节 | 说明 | 分级 |
|---|---:|---|---|
| exported-traced-adjacencies-v1.2.tar.gz | 45,872,577 | traced 神经元与连接（CSV） | **T1** |
| hemibrain-v1.2-body-mean-neurotransmitters.feather | 45,591,786 | 逐 body 的 NT 预测 | **T1** |
| hemibrain-v1.2-all-traced-adjacencies.tar.gz | 80,815,554 | 2026-05 新增，内容待核实 | T2 |
| hemibrain-v1.2-synapse-partners.feather | 859,399,658 | 突触对 | T2 |
| hemibrain-v1.2-tbar-neurotransmitters.feather.bz2 | 442,710,628 | | T2 |
| hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather | 1,838,033,050 | **LARGE** | T2 |
| hemibrain-v1.2-skeletons.tar.gz | 1,881,216,175 | **LARGE** 骨架 | T2 |
| hemibrain-v1.2-ftr.tar.gz | 6,330,212,651 | **LARGE** | T2 |
| roi-meshes.tar.gz | 193,047,019 | 网格 | T2 |

v1.1 的文件也在 JSON 里，例如 `exported-traced-adjacencies-v1.1.tar.gz`（45,473,816 B）。批量导出是 v1.2，neuPrint 上的 v1.2.1 只更新了注释。

### 6.2 Male CNS v1.0（`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`，CC-BY-4.0）
- 文献：Berg et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome*，Cell 189:5504 (2026)。
- 规模：Codex 显示 166,700 个神经元、6,242,118 条连接。
- 小文件：

| 文件 | 字节 |
|---|---:|
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14,483,314 |
| body-neurotransmitters-male-cns-v1.0.feather | 43,282,834 |
| connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather | 502,169,298 |
| connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather | 508,025,642 |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1,051,241,946 |

- **LARGE**：
  - `syn-points-male-cns-v1.0-minconf-0.5.feather`（13.06 GB）；
  - `syn-partners-male-cns-v1.0-minconf-0.5.feather`（6.78 GB）；
  - `syn-partners-…-significant-only.feather` 与 `syn-partners-…-traced-only.feather`（各约 2.97 GB）；
  - `tbar-neurotransmitters-male-cns-v1.0.feather`（2.65 GB）；
  - `body-stats`（0.78 GB）。
- 同一个桶里还有 EM、分割、网格、neo4j 数据库，全部是 EM 级的大体积数据，**不推荐**。

### 6.3 MANC
- 当前版本是 v1.2.1，Janelia 页面写的是“v1.2, released March 11, 2024”。CC-BY-4.0。文献为 eLife 上的三篇论文（Takemura 2024、Marin 2024、Cheong 2024）。
- **公共批量导出只有 v1.0**（`gs://flyem-manc-exports/v1.0/`）：
  - `traced-connections.csv` 75,262,163 B；
  - `traced-neurons.csv` 629,548 B；
  - `traced-connections-per-roi.csv` 159,195,790 B；
  - `manc-v1.0-neuron-properties.feather` 17,188,218 B；
  - `manc-synapse-partners-…-minconf-0.0.feather.bz2` 1,117,661,989 B（LARGE）；
  - `neuprint_manc_v1.0_csv.tar.gz` 5,291,966,339 B（LARGE）。
- v1.2.1 的无 token 来源：Codex GCS 上的 `codex/data/manc/1.2.1/neurons.csv.gz`（1,019,361 B）和 `connections_princeton.csv.gz`（26,747,152 B）。

### 6.4 Male optic lobe v1.1（`gs://flyem-optic-lobe/v1.1/optic-lobe-v1.1-flat-connectome/`）
- 文献：Nern et al. 2025 Nature。规模：Codex 显示 52,445 个神经元。
- 文件：`connectome-weights-…-ol.feather` 152,653,330 B、`…-primary-only` 116,552,210 B、`body-stats` 106,918,410 B，以及 `syn-partners` 和 `syn-points`（最大 1.60 GB）。
- 许可：按 FlyEM 惯例为 CC-BY，本次未逐页核实。T2。

### 6.5 mushroombody（neuPrint，2017）
- Takemura 2017 的 α 叶数据集，只能通过 neuPrint 查询（需要 token）。已被 hemibrain 取代，T2。

---

## 7. FANC（雌性成虫神经索）
- 文献：Azevedo, Lesser, Phelps, Mark et al. 2024 Nature（vol. 631），约 14,600 个胞体、约 4500 万个突触。
- 公共 GCS：`gs://lee-lab_female-adult-nerve-cord/CAVE/v1444/`（2025-10-09）和 `CAVE/v1237/`。v1444 的主要文件：
  - `synapses_nov2022_human_readable_connectioncountsperneuropil.parquet` 192,564,883 B；
  - `…_connectioncounts_countthresh3.parquet` 12,498,976 B；
  - `…_connectioncountsperneuropil_countthresh3.parquet` 15,524,381 B；
  - `cell_ids_v2.parquet` 1,037,439 B；
  - 逐突触表 768,822,583 B。
- 实时 CAVE 查询需要 CAVE token（未使用）。
- **许可在桶中没有声明**，需要向作者或 FANC wiki 核实，T2。BANC 已经把 FANC 1116 的 NBLAST 匹配纳入自己的数据。

---

## 8. 幼虫全脑连接组（Winding et al. 2023, Science 379:eadd9330）
- 规模：约 3,000 个神经元、约 0.55M 个突触。CATMAID 原始重建在 https://catmaid.virtualflybrain.org/（L1 Larval CNS）。
- 补充数据在 PMC7614541 作者稿，许可为 CC BY：
  - `EMS175448-supplement-Supplementary_Data_S1.zip`（约 1 MB，按 a-d/a-a/d-d/d-a 四种连接类型给出的邻接矩阵）；
  - `…_Data_S2.csv`（约 63.6 KB）、`…_Data_S3.csv`（约 58.3 KB）、`…_Data_S4.csv`（约 33.2 KB）；
  - `…Supplementary_Material.pdf`（约 8.1 MB）。
- URL 形如 `https://pmc.ncbi.nlm.nih.gov/articles/instance/7614541/bin/<file>`。
- **精确字节数和校验值无法获取**：PMC 和 science.org 对脚本请求返回机器人校验页或 403，只能在浏览器里手动下载。T2。

---

## 9. flyvis（Lappalainen et al. 2024 Nature）
- PyPI `flyvis` 1.2.0（2026-08-06，MIT，Python 3.9 到 3.12）：
  - `flyvis-1.2.0-py3-none-any.whl` 385,602 B，sha256 `af365b16e870111fcb06782a063f6551b05cce9a105c6eaab9c0bda3b017031f`；
  - sdist 30,073,117 B，sha256 `2f0776dd7208fdf857d7665be7ea4ade7f1b5dbac6dbf8002108ef0f5a24771d`。
- 包内自带连接组 `flyvis/connectome/fib25-fib19_v2.2.json`（618,148 B，柱状视叶模型）和 `flyvis/data/responses_norm.h5`。
- `flyvis download-pretrained` 通过 Google Drive API 下载 `results_pretrained_models.zip`（sha256 `71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1`）和 `results_umap_and_clustering.zip`（sha256 `6300ec1e678cde4d95300ac88e697b75fda32f6404ebe9a50f71e9f653aa9b19`），Drive 文件夹为 `15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj`。CLI 里内嵌了一个 API key；**本次没有用这个 key 查询，所以两个 zip 的大小未核实**。
- 训练和评估自然刺激要用 MPI Sintel：`MPI-Sintel-complete.zip` 5,627,783,629 B，`MPI-Sintel-depth-training-20150305.zip` 1,601,251,750 B。两者都是 **LARGE**，并受 Sintel 网站条款约束。
- 与 F0、F1 无直接关系，T2。

---

## 10. 蘑菇体（MB）与中央复合体（CX）专题资源
- **Li et al. 2020 eLife**（hemibrain MB 连接组）和 **Hulse et al. 2021 eLife**（hemibrain CX 连接组）：两篇的数据就是 hemibrain，没有另建批量仓库（未发现独立的数据仓库）。只要 T1 的 hemibrain 导出，加上 T0 的 FlyWire 边表和注释即可。
- **Caron et al. 2013 Nature 497:113**（PN→KC 随机连接）：补充信息 PDF 为 `41586_2013_BFnature12063_MOESM49_ESM.pdf`，4,712,421 B，ETag `8ca066f0321aee1695fc0417db89e274`，内含 Supplementary Table 1。需要从 PDF 里抽表，版权归出版社，**不是开放许可**。已被 EM 连接组中的 PN→KC 数据取代，T2。
- MB 功能数据：Zenodo 8166598（Campbell 2013 / Srinivasan 2023，KC 气味反应 GCaMP3，CC-BY-4.0），共 15,552,467 B：
  - `fly_supplementary_datasets.tar.gz` 14,766,080 B；
  - `fly_dataset_main.zip` 783,701 B。
  这些数据没有对应到连接组 ID，可作为 MB 稀疏编码统计的校验目标，T2（若 F1 需要 KC 稀疏度标定，可升为 T1）。

---

## 11. 身体与仿真软件（仅列出，低优先级）
- flygym（NeuroMechFly v2）：PyPI 2.1.0，Apache-2.0。wheel 3,156,760 B，sha256 `3ba7292683c1be73897b7f5585612ff57fb9eec28381178efbc0e4368c0e46fd`。论文代码在 Zenodo 12973000，共 26,204,588 B。
- flybody：GitHub TuragaLab/flybody，Apache-2.0，不在 PyPI。仓库约 90 MB（据 GitHub API），其中 `drosophila.blend` 20.8 MB，另有大量 `.obj` 网格。

## 12. 可对齐的功能数据（仅列出，低优先级）
- Gauthey et al. 2025，高速全脑成像处理数据：Zenodo 17618684，单个 `Data.zip` 48,752,619,900 B（**LARGE**），CC-BY-4.0。
- Campbell 2013 / Srinivasan 2023 KC 气味反应：见第 10 节。
- 以下数据本次**未核实存取信息**：Aimon et al. 2023 eLife 85202（全脑光场成像与行走）；Brezovec et al.（全脑成像与行走）；Schaffer et al. 2023 Cell；Pospisil et al. 2024（effectome）。

---

## 13. 需要登录或 token 的条目

| 条目 | 要求 | 无 token 替代 |
|---|---|---|
| FlyWire Codex 网页和下载页 | Google 登录 | 公共 GCS `gs://flywire-data/codex/data/...`（对象公开）；Zenodo 10676866 |
| FlyWire CAVE（实时 materialization） | CAVE token | Zenodo + flywire_annotations |
| neuPrint 查询（hemibrain、MANC、MCNS、optic-lobe） | neuPrint token（Google 登录） | 各 FlyEM 公共 GCS 桶的 flat-connectome 导出 |
| FANC CAVE | CAVE token | `gs://lee-lab_female-adult-nerve-cord/CAVE/v1444/` |
| BANC influence all_to_all（277 个文件，308 GB） | Dataverse 文件访问申请 | `influence_sensory_subclass_to_all.parquet`、`influence_all_to_effector_subclass.parquet`（公开） |
| flyvis 预训练模型 | Google Drive API（CLI 自带 key，本次未用） | 无；可用 `flyvis download-pretrained` 自行下载 |
| Science / PMC 补充文件 | 机器人校验（需浏览器） | 浏览器手动下载 |
| MANC v1.2.x 批量导出 | 公共桶只有 v1.0 | Codex GCS 的 manc/1.2.1 表 |

---

## 14. 许可与再分发注意（本仓库为专有、未公开）
1. **CC-BY-4.0**（FlyWire、flywire_annotations、hemibrain、MANC、MCNS、BANC、Zenodo 记录）：允许在专有仓库中使用和衍生，但凡是发布的结果、图、派生数据，都必须署名并引用指定文献。不要把原始数据文件提交进 git。建议放在 `/Volumes/Samsung/...` 的数据目录，用 manifest 和 md5/sha256 指向。如果对外再分发派生数据，必须保留署名并注明修改。
2. **MIT**（Shiu 仓库及 Edmond 数据）：复制代码或数据时要保留版权声明和许可文本。
3. **GPL-2.0**（eonsystemspbc/fly-brain）：**不得把其代码合入专有代码库**。只能阅读对照，独立实现；对方数据文件与 Shiu 的 MIT 文件相同，从 Shiu 仓库获取即可。
4. **GPL-3.0** 工具（`fafbseg` 3.2.2、`navis` 1.12.0）：只作为外部工具单独安装和调用，不要拷贝源码进仓库。`brian2` 2.10.1 采用 CeCILL-2.1（与 GPL 兼容），同样只作为依赖使用。
5. **出版社版权**（Caron 2013 补充 PDF、science.org 版本）：只用于研究，不要再分发。PMC 作者稿是 CC BY。
6. **FANC**：许可未声明，使用前需确认。
7. **MPI Sintel**：遵守其网站使用条款。
8. 引用清单（F0 与 F1 最少需要）：Dorkenwald 2024 Nature、Schlegel 2024 Nature、Eckstein 2024 Cell（NT）、Buhmann 2021 Nat Methods（突触）、Shiu 2024 Nature；使用 annotations v3.2.0 时再加 Berg 2026 Cell、Tastekin 2026 Cell、Matsliah 2024 Nature；用 hemibrain 时加 Scheffer 2020 eLife、Li 2020 eLife、Hulse 2021 eLife。
9. **安全**：`sez_neurons.pickle` 和 Codex 的 `*.pickle.gz` 都是 pickle 文件，只能在信任来源的前提下加载，或改用 CSV、JSON 等格式的等价数据。

---

## 15. 未能核实或存疑的点
- Shiu 的 `Connectivity_783.parquet` 除 3 个已用列以外的列名，以及 `Completeness_783.csv` 的精确行数：为了不下载数据，只做了推断（138,639）。
- 论文正文没有给出 MN9 的具体 Hz 数值，验证数值需要从参考输出 parquet 计算。example.ipynb 声称默认 200 Hz，代码默认是 150 Hz。论文写“左半球”，代码注释写“右半球”。
- GitHub raw 文件没有公布 md5 或 sha256（只有 git blob SHA1）。Edmond 的 v630 文件与 GitHub 副本字节数一致，但没有逐字节比对。
- Edmond 和 Dataverse 的下载经 303 跳转到预签名 S3 URL，对该 URL 发 HEAD 返回 403（签名只对 GET 有效），实际 GET 未测试。
- flyvis 两个预训练 zip 的大小；幼虫 Winding 2023 补充文件的精确字节数和校验值。
- FANC 和 male optic lobe 的许可声明；hemibrain 新增文件 `hemibrain-v1.2-all-traced-adjacencies.tar.gz` 的内容。
- hemibrain tar 包内的文件名和列名沿用 Janelia 历来格式，未解包核实。
- 功能数据：Brezovec、Aimon、Schaffer、Pospisil 等数据集的存放位置和大小未核实。
- Codex GCS 对象会被原地更新：本清单的 md5 对应 2026-10-04 查询时的状态，下载时如果 md5 不符，说明对象已经更新，而不是下载损坏。
