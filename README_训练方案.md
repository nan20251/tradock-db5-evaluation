# TraDock 训练方案

当前推荐流程：DIPS surfaces 预训练，必要时再用 decoy 数据微调，最后只跑 CAPRI 113 fast 评估。

## 路径约定

```bash
export TRADOCK_DIR=/root/TraDock
export DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full
export CAPRI_DIR=/root/TraDock/data/database
export CHECKPOINT=$TRADOCK_DIR/Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

本地仓库只保留轻量元数据；完整 DIPS/CAPRI 数据放 AutoDL。

## Step 0: 环境检查

```bash
cd "$TRADOCK_DIR"
python scripts/quick_check.py
bash scripts/verify_sasa_deployment.sh
```

## Step 1: DIPS surfaces 检查

```bash
python scripts/check_ply_fields.py "$DIPS_SURFACES"
python scripts/check_ply_dimensions.py "$DIPS_SURFACES"
python scripts/check_data_range.py "$DIPS_SURFACES"
python scripts/audit_dips_surfaces.py "$DIPS_SURFACES" --samples 1yk0_A_B 1u0c_A_B
```

模型读入特征应为 11 维；原始 PLY vertex 字段通常为 14 个。
如果确认单样本有问题，再跑全量审计：

```bash
python scripts/audit_dips_surfaces.py "$DIPS_SURFACES" \
  --out results/dips_surface_audit.csv
```

## Step 2: Native 预训练

```bash
DIPS_SURFACES="$DIPS_SURFACES" bash scripts/run_step2_pretrain.sh
```

训练脚本会先生成 `results/dips_with_sasa_full.filtered_pairs.csv`，默认排除 `1u0c_A_B`、`1yk0_A_B` 两个 rSASA NaN 样本，并按 `data/dips/exclude_capri.txt` 排除 CAPRI 相关 PDB 前缀。

等价手动命令：

```bash
python examples/train.py \
  --data_dir "$DIPS_SURFACES" \
  --save_dir Trained_models/pretrain_with_sasa \
  --epochs 30 \
  --batch_size 2 \
  --lr 1e-4 \
  --contrast_weight 0.0
```

训练中若所有验证 batch 都被跳过，验证 loss 会显示 `inf`，不会再误报为 0。

## Step 3: 模型质量检查

```bash
python scripts/check_model_quality.py \
  --data_dir "$DIPS_SURFACES" \
  --checkpoint "$CHECKPOINT"
```

## Step 4: Decoy 微调（可选）

如果已经有 decoy surfaces 和 `decoys.csv`：

```bash
python examples/train_native_vs_decoy_v2.py \
  --native_dir "$DIPS_SURFACES" \
  --decoy_dir /root/autodl-tmp/lightdock_surfaces \
  --decoy_csv /root/autodl-tmp/lightdock_surfaces/decoys.csv \
  --init_from "$CHECKPOINT" \
  --save_dir Trained_models/finetune_v1 \
  --epochs 20 \
  --lr 5e-5
```

## Step 5: CAPRI 113 fast 评估

```bash
CAPRI_DIR="$CAPRI_DIR" CHECKPOINT="$CHECKPOINT" bash scripts/run_step7_eval.sh
```

## Step 6: DB5 apo/holo 论文对齐评估

该步骤对齐论文 *Revisiting Protein-protein Docking: A Systematic Evaluation Framework*：
使用官方 `PPCBench` 仓库/Zenodo 数据中的 `dataset/DB5`（holo）和
`dataset/DB5-u`（apo），指标为 C-RMSD、I-RMSD、DockQ，成功判据为
`DockQ >= 0.23`，并汇总 `Success@1/3/5/10/100`。

TraDock 当前是 scoring/reranking 模型，不生成 docking poses。因此这里输入必须是
论文官方 `results/<dataset>/<pose_model>/<pdb>/...pdb` 候选构象；脚本用
TraDock 对这些候选构象重新打分排序，再按论文指标计算 Top-N success。

```bash
export PAPER_ROOT=/root/PPCBench
export CHECKPOINT=$TRADOCK_DIR/Trained_models/pretrain_with_sasa/TransformerDock_best.chk

# holo: dataset/DB5 + results/DB5/hdock_1..5
# apo:  dataset/DB5-u + results/DB5-u/hdock_1..5
bash scripts/run_step8_eval_db5_paper.sh
```

`DB5-g-u` 是官方 GeoDock flexible 结果目录，不等同于刚性 DB5 apo
主评估的 `DB5-u`。只有在额外评估 GeoDock flexible poses 时才覆盖：

```bash
DB5_APO_DATASET=DB5-g-u \
APO_POSE_MODELS=geodock \
bash scripts/run_step8_eval_db5_paper.sh
```

输出：

```text
results/tradock_DB5_paper.csv
results/tradock_DB5_paper.summary.csv
results/tradock_DB5_paper.summary.aggregate.csv
results/tradock_DB5-u_paper.csv
results/tradock_DB5-u_paper.summary.csv
results/tradock_DB5-u_paper.summary.aggregate.csv
```
