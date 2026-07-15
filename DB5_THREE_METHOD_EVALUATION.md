# DB5 三路评估方案

目标：把 HDOCK、AlphaFold/ColabFold、LightDock 三类 decoy 来源放到统一的、论文级的 DB5 评估框架下，评估 TraDock 的重排能力。

## 1. 评估原则

### 1.1 分清两个问题

1. 采样器好不好：HDOCK / AlphaFold / LightDock 能不能生成接近 native 的构象。
2. 打分器好不好：TraDock 能不能在同一个 decoy 池里把好构象排到前面。

所以最终表格要同时报告：

- 原始方法排序：HDOCK rank、AlphaFold ipTM/ranking_confidence、LightDock score。
- TraDock 重排：TraDock MDN score。
- Oracle 上限：按真实 DockQ/fnat 排序。

### 1.2 不混用不可比分数

AlphaFold 的 ipTM、PAE、pLDDT 只存在于 AlphaFold 生成的 decoy 上，不能直接给 HDOCK/LightDock decoy 打分。

因此：

- AF decoy 池：可以直接比较 AlphaFold 排序 vs TraDock 重排。
- HDOCK/LightDock decoy 池：比较原始 docking 排序 vs TraDock 重排。
- 混合池只能主要用于 TraDock 统一打分，不能说 AF score 覆盖整个混合池。

## 2. 数据集

使用 PPCBench 整理版 DB5：

| 数据集 | 数量 | 用途 |
|---|---:|---|
| DB5 | 218 | holo/bound 结构，适合已有 holo pose 或 AlphaFold 对齐后评估 |
| DB5-u | 218 | unbound/apo 输入，适合 HDOCK/LightDock 重新采样 |

DB5-u 不是单链数据。当前统计：

- 多 receptor 链 target：83 个
- 多 ligand 链 target：11 个

论文级主实验建议：

- HDOCK 和 LightDock：用 DB5-u 的 `*_r_u_f.pdb`、`*_l_u_f.pdb` 做输入，评估对齐到 DB5-u 的 fitted bound ground truth。
- AlphaFold：用序列 FASTA 生成复合物；评估前把 AF receptor 对齐到 DB5 bound receptor，再把同一变换作用到 AF ligand。

## 3. 三个 Decoy 池

### 3.1 HDOCK regenerated DB5-u Top100

目的：传统 docking 采样池，对标论文里 docking 方法生成 pose 再重排的评估。

设置：

- 输入：DB5-u unbound receptor/ligand。
- 每个 target 生成 Top100。
- 总 decoy 数：218 x 100 = 21800。
- 原始排序：HDOCK rank 1..100。
- TraDock 排序：MDN score 从高到低。
- Ground truth：DockQ、fnat、iRMSD、cRMSD。

当前状态：

- 已经验证 HDOCKlite 可运行。
- 已经验证 `1AY7` Top5、Top100 生成和 TraDock 评估链路。
- 全量任务曾启动，后按要求停止。
- 停止时已完成 `2I9B`、`3V6Z`、`2W9E` 的 Top100，`1F34` 运行中被停止。

输出规划：

```text
/root/autodl-tmp/hdock_regen_full/results/DB5-u/hdock_1 ... hdock_100
/root/TraDock/results/tradock_DB5u_hdock_regen_top100.csv
/root/TraDock/results/hdock_regen_top100_tradock_compare.summary.csv
/root/TraDock/results/hdock_regen_top100_tradock_compare.targets.csv
```

已有但不作为最终重排主结果的参考：

| 项目 | 数值 |
|---|---:|
| DB5 hdock_1 target | 218 |
| 有效评分 target | 208 |
| DockQ >= 0.23 成功数 | 161 / 208 |
| 有效成功率 | 77.4% |
| 按 218 计成功率 | 73.85% |
| mean DockQ | 0.756 |
| median DockQ | 0.9765 |

说明：已有 `hdock_1` 每个 target 只有一个 pose，不能评估同 target 内重排能力，只能作为 HDOCK top1 质量参考。

### 3.2 AlphaFold / ColabFold DB5 Top5

目的：比较 AlphaFold 自带置信分数与 TraDock 对 AF decoys 的重排能力。

设置：

- 输入：218 个 DB5 target 的 multimer FASTA。
- 每个 target 输出 Top5。
- 总 decoy 数：218 x 5 = 1090。
- 原始排序：ColabFold `ranking_confidence` / ipTM。
- TraDock 排序：MDN score 从高到低。
- Ground truth：DockQ、fnat、iRMSD、cRMSD。

格式转换关键点：

1. ColabFold 输出的是完整复合物，不是 ligand-only pose。
2. 先把 AF receptor 对齐到 DB5 native receptor。
3. 把同一个刚体变换作用到 AF ligand。
4. 输出 ligand decoy 给 PPCBench / TraDock 评估。
5. 多链 target 不能简单按 AF 链等于 PDB 链处理，因为 FASTA 是 receptor 合并序列 : ligand 合并序列。

当前 probe 结果：

| target | AF decoys | AF Top1 DockQ | TraDock Top1 DockQ | Oracle DockQ |
|---|---:|---:|---:|---:|
| 1AY7 | 2 | 0.910 | 0.913 | 0.913 |

相关文件：

```text
/root/TraDock/results/af2m_probe_1AY7_scores.csv
/root/TraDock/results/af_vs_tradock_probe_1AY7.merged.csv
/root/TraDock/results/af_vs_tradock_probe_1AY7.summary.csv
/root/TraDock/results/af_vs_tradock_probe_1AY7.aggregate.csv
```

论文级下一步：

- 先跑 20 个 target Top5。
- 验证多链拆分、DockQ、TraDock 打分、ipTM/PAE/pLDDT 提取。
- 再跑全 218。

### 3.3 LightDock DB5-u Top100

目的：第二个传统 docking 采样池，用来验证 TraDock 对低质量或更分散 decoy 池的重排能力。

设置：

- 输入：DB5-u unbound receptor/ligand。
- 每个 target 至少保留 Top100。
- 原始排序：LightDock score。
- TraDock 排序：MDN score 从高到低。
- Ground truth：DockQ、fnat、iRMSD、cRMSD。

当前已有结果：

| 实验 | target | decoy/target | 结论 |
---|---:|---:|---|
| LightDock Top5 existing | 218 | 5 | 207 个有效 target，Success@1/5/10/100 都为 0 |
| LightDock test20 fast oracle | 20 | 1000 | oracle success 2/20 = 10%，mean best DockQ 0.08475 |

说明：

- 当前 LightDock 结果质量明显偏低。
- 论文级结果不能只用现有 Top5。
- 应使用 DB5-u，重新统一生成 Top100 或 Top1000，并保留 LightDock 原始 score。
- 如果 oracle 本身很低，TraDock 再好也不可能显著提升 Top1 成功率；所以必须报告 oracle upper bound。

## 4. 指标

### 4.1 主指标

成功阈值：

- DockQ >= 0.23：acceptable docking pose。
- fnat > 0.3：与前面 CAPRI 113 表保持一致。

Top-k：

```text
Success@1, Success@5, Success@10, Success@20, Success@100
```

每个 decoy 池分别报告：

| 排序方式 | 含义 |
|---|---|
| Original | HDOCK rank / LightDock score / AlphaFold ranking_confidence |
| TraDock | TraDock MDN score |
| Oracle | 按真实 DockQ 或 fnat 排序 |

### 4.2 质量指标

对 Top1 和 best available 报告：

- mean / median DockQ
- mean / median iRMSD
- mean / median cRMSD
- high / medium / acceptable / unacceptable 比例

### 4.3 排序相关性

每个池内报告：

- per-target Spearman：score vs DockQ
- pooled Spearman：score vs DockQ
- pooled Pearson：score vs DockQ

优先看 per-target Spearman，因为 docking 重排本质是同一个 target 内排序。

### 4.4 分类指标

把 `DockQ >= 0.23` 或 `fnat > 0.3` 当正样本：

- AUROC
- AUPRC

注意：如果某个 decoy 池 oracle positive 很少，AUPRC 会受正样本比例影响，必须同时报告正样本数。

## 5. 论文级主表设计

### 5.1 Decoy pool 质量表

| Decoy source | Dataset | Targets | Decoys/target | Oracle@1 | Oracle@5 | Oracle@10 | Oracle@100 | Mean best DockQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HDOCK | DB5-u | 218 | 100 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| AlphaFold | DB5 | 218 | 5 | 待跑 | 待跑 | - | - | 待跑 |
| LightDock | DB5-u | 218 | 100 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

### 5.2 TraDock 重排表

| Decoy source | Original@1 | TraDock@1 | Original@5 | TraDock@5 | Original@10 | TraDock@10 | Original@100 | TraDock@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HDOCK Top100 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| AlphaFold Top5 | 待跑 | 待跑 | 待跑 | 待跑 | - | - | - | - |
| LightDock Top100 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

### 5.3 分数质量表

| Decoy source | Score | Spearman | Pearson | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| HDOCK Top100 | HDOCK rank/score | 待跑 | 待跑 | 待跑 | 待跑 |
| HDOCK Top100 | TraDock score | 待跑 | 待跑 | 待跑 | 待跑 |
| AlphaFold Top5 | ipTM / ranking_confidence | 待跑 | 待跑 | 待跑 | 待跑 |
| AlphaFold Top5 | TraDock score | 待跑 | 待跑 | 待跑 | 待跑 |
| LightDock Top100 | LightDock score | 待跑 | 待跑 | 待跑 | 待跑 |
| LightDock Top100 | TraDock score | 待跑 | 待跑 | 待跑 | 待跑 |

## 6. 推荐执行顺序

## 6.0 当前代码入口

统一环境配置：

```bash
cd /root/TraDock
vim environment
```

所有路径、采样数量和开关集中在 `environment`。脚本会自动读取这个文件；命令行临时变量仍然可以覆盖文件里的默认值。

统一入口：

```bash
cd /root/TraDock
bash scripts/run_db5_three_method_eval.sh
```

默认行为：

- HDOCK：DB5-u，Top100，重新生成 decoys，再 TraDock 打分和汇总。
- AlphaFold：DB5，默认按论文口径跑 `3 random seeds x 5 models`，再按 AF confidence 合并选 Top5；默认只转换已有 ColabFold 输出并评估。要真实跑 ColabFold，设置 `RUN_COLABFOLD=1`。
- LightDock：DB5-u，Top100，重新生成 decoys，按 LightDock final GSO `scoring` 排序导出，再导出为论文评估布局。

常用子集调试：

```bash
METHODS="hdock" HDOCK_LIMIT=5 bash scripts/run_db5_three_method_eval.sh
METHODS="alphafold" AF_LIMIT=20 RUN_COLABFOLD=1 bash scripts/run_db5_three_method_eval.sh
METHODS="lightdock" LIGHTDOCK_LIMIT=20 bash scripts/run_db5_three_method_eval.sh
```

只跑已有 decoy 的评估：

```bash
METHODS="alphafold" RUN_COLABFOLD=0 bash scripts/run_db5_three_method_eval.sh
METHODS="lightdock" RUN_LIGHTDOCK=0 bash scripts/run_db5_three_method_eval.sh
```

严格复现论文 Top5 口径：

```bash
METHODS="hdock" HDOCK_NMAX=5 bash scripts/run_db5_three_method_eval.sh
METHODS="alphafold" AF_NUM_SEEDS=3 AF_MODELS_PER_SEED=5 AF_NMAX=5 RUN_COLABFOLD=1 bash scripts/run_db5_three_method_eval.sh
```

说明：`HDOCK_NMAX=5` 表示 HDOCK 只导出和评估前 5 个 pose。论文保留 Top5；我们的 TraDock 重排主实验保留 Top100，是为了看重排空间。`LightDock_NMAX=5` 只能作为同口径补充，因为 LightDock 不是论文 11 个方法之一。

相关脚本：

- `examples/generate_db5_hdocklite_candidates.py`：HDOCKlite TopN 生成，已兼容 `Hdock.out` / `model_*.pdb` 固定输出。
- `examples/prepare_db5_colabfold_fastas.py`：生成 ColabFold multimer FASTA。
- `examples/convert_colabfold_db5.py`：AF complex 对齐 receptor 后导出 ligand pose。
- `examples/compare_af_tradock.py`：AF ranking_confidence/ipTM 与 TraDock 分数比较。
- `examples/prepare_db5_lightdock_inputs.py`：生成 LightDock DB5-u 输入。
- `examples/export_lightdock_to_paper_candidates.py`：解析 `swarm_*/gso_*.out`，按 LightDock `scoring` 排序，把 complex decoy 导出成 `lightdock_1..N` 论文布局；同时写出 `lightdock_<dataset>_selected.csv` 记录每个 pose 的 score / swarm / glowworm。
- `examples/summarize_paper_pose_scores.py`：对任意 TraDock detail CSV 汇总 Spearman/Pearson/AUROC/AUPRC。

### Step 1: HDOCK Top100 完整跑完

HDOCK 已经验证可跑。继续跑全 218：

```bash
cd /root/TraDock
mkdir -p results/logs
screen -dmS db5_hdock_top100 bash -lc \
  'cd /root/TraDock && METHODS="hdock" HDOCK_NMAX=100 bash scripts/run_db5_three_method_eval.sh > results/logs/db5_hdock_top100.log 2>&1'
```

监控：

```bash
tail -f /root/TraDock/results/logs/db5_hdock_top100.log
```

### Step 2: AlphaFold Top5 先跑 20 个 target

目的：验证多链 target、AF score 提取和 receptor alignment。

通过后再跑 218。

### Step 3: LightDock Top100 重新按 DB5-u 生成

先用 20 个 target 估计 oracle upper bound。

如果 oracle@100 仍很低，应在论文中明确说明 LightDock sampling pool 本身不足，不把 TraDock 失败归因于打分器。

### Step 4: 汇总三路表格

每一路都生成：

```text
detail.csv
target_summary.csv
aggregate.csv
```

最终只从 aggregate 和 target_summary 取数进论文主表。

## 7. 当前结论边界

现在不能直接写成最终论文结论，因为：

1. HDOCK 完整 Top100 还没跑完。
2. AlphaFold 只有 `1AY7` probe，不代表全 DB5。
3. LightDock 当前 full DB5 Top5 和 test20 都显示采样质量弱，需要重新做 DB5-u Top100/Top1000。
4. 已有 HDOCK `hdock_1` 很强，但它只有单 pose，不能证明 TraDock 重排能力。

可以写的中间结论：

- HDOCK top1 在当前 DB5 holo 结果中质量很高，按 218 计 Success@1 为 73.85%。
- AlphaFold probe `1AY7` 上 TraDock 能在 2 个 AF decoy 里选到更高 DockQ 的 pose，但样本太少。
- LightDock 当前采样池质量不足，oracle upper bound 偏低，必须在最终表里单独报告 oracle。
