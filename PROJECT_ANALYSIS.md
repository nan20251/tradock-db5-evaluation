# TraDock 项目分析

## 1. 项目目标

TraDock 是一个蛋白-蛋白 docking 打分模型。

当前主要任务：

- 给每个 decoy 打分
- 把接近 native 的 decoy 排到前面
- 在 CAPRI 113 和 DB5 上评估排序能力

当前活跃版本使用 11 维表面特征。19 维 pair-aware 实验已经从当前主流程移除。

## 2. 项目结构

```text
transformerdock/
  models.py                 模型结构和 MDN loss / score
  utils/data.py             PLY 读取、11维特征构造、PyG Data 构造

examples/
  train.py                  DIPS 预训练入口
  train_native_vs_decoy_v2.py native/decoy 排序训练入口
  eval_capri_fast.py        CAPRI 评估入口
  eval_db5_paper_tradock.py DB5 评估入口
  surface_gen.py            PDB 到表面 PLY 特征生成

scripts/
  run_step2_pretrain.sh     DIPS 预训练
  run_step7_eval.sh         CAPRI 评估
  run_step8_eval_db5_paper.sh DB5 评估
  check_*.py                数据和模型检查

data/
  dips/                     DIPS 元数据和 CAPRI 排除列表
  capri113_single_targets/  CAPRI 轻量元数据

Trained_models/
  pretrain_with_sasa/       当前默认 checkpoint

results/
  CAPRI / BioScore 对比 / 失败分析结果
```

## 3. 输入特征

模型读取两个 `.ply` 表面文件：

- receptor surface
- ligand surface

每个表面点是一个图节点。当前节点特征是 11 维：

| index | 特征 |
|---:|---|
| 0 | 法向量 nx |
| 1 | 法向量 ny |
| 2 | 法向量 nz |
| 3 | charge |
| 4 | hydrophobicity |
| 5 | hbond_donor |
| 6 | hbond_acceptor |
| 7 | curvature |
| 8 | shape_index |
| 9 | aa_polar |
| 10 | rSASA |

坐标 `pos` 不在这 11 维里，但会用于几何距离和 cross-pair 建模。

输入路径：

```text
read_ply()
  -> match_feature_dim(..., in_channels=11)
  -> PPI_Dataset / prepare_complex
  -> DeepDock_PPI
```

## 4. 模型和打分

核心模型在 `transformerdock/models.py`。

当前主 score 是 MDN score：

```text
score = mean log P(d_ij), for receptor-ligand point pairs with distance < threshold
```

默认距离阈值：

```text
dist_threshold = 10.0 Å
```

训练时主要优化 native interface pair 的 MDN negative log likelihood。

评估时对一个 target 的所有 decoy 逐个打分，然后按 score 从高到低排序。

## 5. CAPRI 113 评估结果

当前默认 checkpoint：

```text
Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

评估设置：

```text
targets = 113
fnat positive threshold = 0.3
DockQ positive threshold = 0.23
denominator = all targets
```

### 5.1 Top-k 成功率

`fnat > 0.3`：

| 方法 | Top1 | Top2 | Top5 | Top10 | Top20 | Top100 |
|---|---:|---:|---:|---:|---:|---:|
| TraDock | 25.7% | 35.4% | 46.0% | 51.3% | 57.5% | 65.5% |
| DeepRank-GNN-ESM | 27.4% | 31.0% | 40.7% | 49.6% | 54.0% | 65.5% |

`DockQ >= 0.23`：

| 方法 | Top1 | Top2 | Top5 | Top10 | Top20 | Top100 |
|---|---:|---:|---:|---:|---:|---:|
| TraDock | 29.2% | 39.8% | 45.1% | 49.6% | 57.5% | 67.3% |
| DeepRank-GNN-ESM | 27.4% | 32.7% | 40.7% | 48.7% | 53.1% | 67.3% |

### 5.2 最高 fnat / 最高 DockQ

全局最高 fnat decoy：

| target | model_id | score | fnat | DockQ | class |
|---|---:|---:|---:|---:|---|
| S-T086.2 | 275 | -0.7266 | 1.0000 | 0.8141 | medium |

全局最高 DockQ decoy：

| target | model_id | score | fnat | DockQ | class |
|---|---:|---:|---:|---:|---|
| S-T125.2 | 2133 | -0.6903 | 0.9474 | 0.9569 | high |

Oracle 覆盖率：

| oracle 指标 | 阈值 | target 数 | 比例 |
|---|---:|---:|---:|
| best fnat | `fnat >= 0.3` | 76 / 113 | 67.3% |
| best DockQ | `DockQ >= 0.23` | 78 / 113 | 69.0% |

### 5.3 结果解读

简单结论：

- `fnat > 0.3` 下，DeepRank-GNN-ESM 的 Top1 更高，TraDock 从 Top2 到 Top20 更高。
- `DockQ >= 0.23` 下，TraDock 从 Top1 到 Top20 都高于 DeepRank-GNN-ESM，Top100 持平。
- 但平均 Spearman 仍低，说明整体排序相关性弱。
- `success@100` 明显高于 `success@1`，说明很多 target 的好 decoy 在候选里，但没有排到最前面。

## 6. 主要问题

### 6.1 排序信号不够强

模型能找到一部分好 decoy，但 top1 不稳定。

表现：

- 有些 target 的 best decoy 排在很后面
- top5/top10 明显好于 top1
- mean Spearman 只有 0.082

### 6.2 训练目标和评估目标不完全一致

训练主要学 native interface distance likelihood。

评估要解决的是：

```text
同一个 target 下，哪个 decoy 更接近 native？
```

这需要 ranking 信号。

### 6.3 只靠 MDN score 不够

MDN score 看局部距离概率，但 docking 排序还需要考虑：

- clash
- interface size
- contact quality
- shape complementarity
- electrostatic / hydrophobic consistency
- target 内归一化

### 6.4 19 维实验失败原因

19 维 pair-aware 实验已经移除。

失败原因总结：

- 新增特征是 pose-aware，但训练没有足够 decoy ranking 约束
- 从 11 维 checkpoint 初始化时输入层 shape 不匹配，第一层没有继承
- CAPRI 上 score 和 fnat 相关性变差
- 结果是好 decoy 仍在候选里，但排序更差

## 7. 改进方向

### 7.1 加 ranking loss

目标：让好 decoy 分数高于差 decoy。

可以用：

```text
loss = mdn_loss + λ * ranking_loss
```

ranking pair：

- native vs decoy
- high DockQ vs low DockQ
- high fnat vs low fnat

建议先做最简单版本：

```text
same target 内采样两个 decoy
如果 DockQ_a > DockQ_b + margin
要求 score_a > score_b
```

### 7.2 做 target 内 score 校准

不同 target 的 score 尺度不一定一致。

可以在每个 target 内加简单 rescore：

```text
final_score = zscore(mdn_score)
            + a * zscore(contact_score)
            - b * zscore(clash_score)
```

先不要上复杂模型，先做线性组合和验证集搜索。

### 7.3 加轻量物理后处理

不要直接重启 19 维模型。

更稳的做法：

- 保留 11 维模型
- 评估后追加物理特征
- 用小模型或线性权重 rerank

候选特征：

- `mdn_score`
- interface pair count
- close contact count
- clash count
- contact density
- buried SASA 近似
- DockQ/fnat 可用时只作为训练 label，不作为测试输入

### 7.4 建一个小 validation set

不能只看 DIPS test loss。

需要固定一个 CAPRI validation split：

```text
train: DIPS + decoy ranking data
val: held-out CAPRI targets
test: final CAPRI / DB5 report
```

每次改模型都看：

- success@1
- success@5
- mean Spearman
- mean AUC

### 7.5 失败案例优先修

先看 `results/top1_failure_analysis_tradock_fnat03.csv`。

优先处理两类：

- 好 decoy 排名 2-10：适合轻量 rerank
- 好 decoy 排名很后：需要训练目标改进

## 8. 推荐下一步

短期：

1. 固定当前 11 维 baseline。
2. 从 CAPRI 结果提取 per-decoy 表格。
3. 训练一个简单 reranker，只用现成 score 和几何统计。
4. 用 held-out target 验证 success@1 是否提升。

中期：

1. 在 `train_native_vs_decoy_v2.py` 中加入 pairwise ranking loss。
2. 训练 native/decoy 混合模型。
3. 和当前 `pretrain_with_sasa` checkpoint 对比。

长期：

1. 重新设计 pair-aware 特征。
2. 先保证输入层继承 11 维权重。
3. 每次新增特征都必须配套 ranking validation。

## 9. 当前不要做的事

- 不要直接恢复 19 维主流程。
- 不要只看 DIPS loss 判断模型好坏。
- 不要用全 CAPRI 调参后再报告同一个 CAPRI 结果。
- 不要大规模重构模型，先把 ranking 问题验证清楚。
