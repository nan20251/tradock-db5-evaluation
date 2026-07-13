# TraDock / BioScore Project Memory

Last updated: 2026-06-19 Asia/Shanghai

## Purpose

Compare TraDock and BioScore-PPI on the same CAPRI 113 benchmark with a fair
deduplicated training setup. The main evaluation requested by the user is full
113 targets with positive defined as `fnat > 0.3`, and every target should be
written into CSV.

## Local Project

- Local TraDock repo: `/Users/nanyang/Documents/AutoDL-Projects/TraDock`
- Important local scripts:
  - `examples/eval_capri_fast.py`
  - `scripts/run_step7_eval.sh`
  - `scripts/run_step2_pretrain.sh`
  - `scripts/filter_bad_dips_pairs.py`
  - `scripts/compare_capri_results.py`

## Remote Servers

Do not store passwords in this file.

- TraDock server:
  - SSH: `ssh -p 19464 root@connect.bjb1.seetacloud.com`
  - Project: `/root/TraDock`
  - Data disk: `/root/autodl-tmp`
- BioScore server:
  - SSH: `ssh -p 46984 root@connect.westc.seetacloud.com`
  - Project: `/root/BioScore-PPI`
  - Data disk: `/root/autodl-tmp`

## TraDock Server State

- TraDock project: `/root/TraDock`
- Backup made earlier: `/root/TraDock_backup_20260618_131456`
- Symlinks:
  - `/root/TraDock/Trained_models -> /root/TraDock_backup_20260618_131456/Trained_models`
  - `/root/TraDock/data/database -> /root/TraDock_backup_20260618_131456/data/database`

### CAPRI 113 Data

- CAPRI directory: `/root/TraDock/data/database`
- Contains 113 PDB and 113 CSV files.
- PDB and CSV target names match.
- Sorted target list SHA256 previously checked:
  - `74443102917a1ced9b171b6b559d4aaec5e7f2982360c0e68bc696049adf53ba`

### DIPS / SASA Training Data

- Full SASA data: `/root/autodl-tmp/dips_with_sasa_full`
- Pair count: 40517 before filtering.
- Vertex fields include `rSASA`.
- Known bad samples with literal NaN in `rSASA`:
  - `1u0c_A_B`
  - `1yk0_A_B`
- CAPRI exclusion file:
  - `/root/TraDock/data/dips/exclude_capri.txt`
  - 122 PDB prefixes.
- Filtered pair list:
  - `/root/TraDock/results/dips_with_sasa_full.filtered_pairs.csv`
- Filter report from prior run:
  - kept: 40285
  - removed bad samples: 2
  - removed CAPRI-prefix samples: 230
  - final bad hits: 0
  - final CAPRI-prefix hits: 0

### TraDock Training

Training command used for the completed run:

```bash
python -u examples/train.py \
  --data_dir /root/autodl-tmp/dips_with_sasa_full \
  --pairs_csv results/dips_with_sasa_full.filtered_pairs.csv \
  --save_dir Trained_models/pretrain_with_sasa \
  --epochs 30 \
  --batch_size 2 \
  --lr 1e-4 \
  --contrast_weight 0.0 \
  --save_every 1 \
  --resume Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

Training completed on 2026-06-19.

Final usable checkpoint:

- `/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk`
- `best_epoch = 27`
- `best_loss = 0.7242806092681128`

Backup of pre-resume best:

- `/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best_before_resume_epoch021.chk`
- `epoch = 21`
- `loss = 0.7621496034023484`

Per-epoch validation losses from resume:

```text
epoch 22  test 0.7738892954868951
epoch 23  test 0.7766287056447259
epoch 24  test 0.7666781854097068
epoch 25  test 0.7470925003955737
epoch 26  test 0.7645827901866241
epoch 27  test 0.7242806092681128
epoch 28  test 0.7818909325019894
epoch 29  test 0.7648152951862912
epoch 30  test 0.7679548269170093
```

Generated training files:

- `/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_epoch_030.chk`
- `/root/TraDock/Trained_models/pretrain_with_sasa/training_loss.csv`
- `/root/TraDock/Trained_models/pretrain_with_sasa/training_curve.png`

No NaN, Inf, traceback, CUDA OOM, or explicit training error was found in logs.

### TraDock Evaluation In Progress

Active evaluation was started with:

```bash
python -u examples/eval_capri_fast.py \
  --data_dir data/database \
  --checkpoint Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
  --out results/capri_eval_113_fnat03_best_epoch027_single.csv \
  --pos_metric fnat \
  --pos_threshold 0.3 \
  --success_denominator all \
  --score_type mdn \
  --n_workers 1
```

Why single worker:

- A previous `--n_workers 4` run stalled on target 1.
- The stalled parent and orphan workers were killed.
- Single worker run is slower but has been writing CSV correctly.

Evaluation files:

- Detail CSV:
  - `/root/TraDock/results/capri_eval_113_fnat03_best_epoch027_single.csv`
- Summary CSV:
  - `/root/TraDock/results/capri_eval_113_fnat03_best_epoch027_single.summary.csv`
- Log:
  - `/root/autodl-tmp/tradock_eval_fnat03_single_20260619_172127.log`

Last observed progress:

- done: 28 / 113
- current target: `[29/113] S-T072.1`
- ETA at that time: around 2026-06-20 00:00 to 00:30
- Only warning observed:
  - PyTorch `enable_nested_tensor` user warning due `norm_first=True`
  - This is not an evaluation error.

Partial metrics at 24 / 113:

```text
Partial mean AUC: 0.3964491107748535
Success@1:   7 / 24 = 29.17%
Success@2:  10 / 24 = 41.67%
Success@5:  11 / 24 = 45.83%
Success@10: 13 / 24 = 54.17%
Success@100:15 / 24 = 62.50%
```

## BioScore Server State

Project:

- `/root/BioScore-PPI`

CAPRI 113 data:

- `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/database`
- Same 113 targets as TraDock.
- Sorted target list SHA256 previously checked:
  - `74443102917a1ced9b171b6b559d4aaec5e7f2982360c0e68bc696049adf53ba`

Original BioScore DIPS data:

- `/root/autodl-tmp/dips_processed/train.pkl`
- `/root/autodl-tmp/dips_processed/valid.pkl`
- Processed caches:
  - `/root/autodl-tmp/dips_processed/train.BlockGeoAffDataset_processed.pkl`
  - `/root/autodl-tmp/dips_processed/valid.BlockGeoAffDataset_processed.pkl`

Original data had leakage:

- train CAPRI-prefix hits: 210
- valid CAPRI-prefix hits: 22
- raw train/valid exact ID overlap: 267

Deduplicated BioScore processed input:

- `/root/autodl-tmp/dips_processed_tradock_dedup`
- Report:
  - `/root/autodl-tmp/dips_processed_tradock_dedup/filter_report.json`
- Filter results:
  - train input: 37900
  - train kept: 37679
  - train bad exact: 2
  - train CAPRI-prefix removed: 219
  - valid input: 4212
  - valid kept: 3919
  - valid CAPRI-prefix removed: 22
  - valid overlap removed: 271
  - final train/valid overlap: 0
  - final CAPRI hits: 0
  - final bad hits: 0

### BioScore Dedup Training

Config:

- `/root/BioScore-PPI/scripts/exps/configs/DIPS-pretrain-tradock-dedup.json`

Output:

- `/root/autodl-tmp/DIPS_pretrain_tradock_dedup_run/version_0/checkpoint/`

Best checkpoint:

- `/root/autodl-tmp/DIPS_pretrain_tradock_dedup_run/version_0/checkpoint/epoch99_step129100.ckpt`
- validation: `0.5717168060034069`

### BioScore Dedup Evaluation

Prediction output:

- `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_tradock_dedup/dips_dedup_best/results.jsonl`
- Finite filtered:
  - `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_tradock_dedup/dips_dedup_best/results.finite.jsonl`
- Nonfinite report:
  - `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_tradock_dedup/dips_dedup_best/nonfinite_predictions.csv`

Important note:

- `results.jsonl` field `gt` is DockQ-like, not true fnat.
- For `fnat > 0.3`, align prediction IDs to CAPRI database CSV and read true `fnat`.

Dedup full 113, true `fnat > 0.3`, high score better:

```text
Targets: 113
Targets with fnat > 0.3: 76
Mean AUC fnat>0.3: 0.46347448735628816
Mean Spearman fnat: 0.0072546058601718946
Mean Pearson fnat: 0.014628443881652228
Success@1:  18 / 113 = 15.93%
Success@2:  28 / 113 = 24.78%
Success@5:  38 / 113 = 33.63%
Success@10: 44 / 113 = 38.94%
Success@100:72 / 113 = 63.72%
```

Low score better alternative was only slightly different:

```text
Mean AUC fnat>0.3: 0.5365255126437118
Success@1: 17 / 113 = 15.04%
Success@10: 48 / 113 = 42.48%
Success@100: 75 / 113 = 66.37%
```

Conclusion:

- Dedup BioScore has weak ranking signal on CAPRI.
- Pearson/Spearman are near zero.

### BioScore Original Non-Dedup Evaluation

Existing original non-dedup checkpoint:

- `/root/autodl-tmp/DIPS_pretrain_run/version_0/checkpoint/epoch98_step128502.ckpt`

Existing predictions:

- `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_dips/dips_pretrain/results.jsonl`

Recomputed full 113 with true CAPRI CSV `fnat > 0.3`, high score better:

Output:

- `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_dips/dips_pretrain/per_target_full_fnat03_true.summary.csv`
- `/root/BioScore-PPI/scripts/test/capri_scoreset_v2022/work_dips/dips_pretrain/per_target_full_fnat03_true.csv`

Results:

```text
Targets: 113
Targets with positive: 76
Mean Spearman fnat: 0.0757001995484071
Mean Pearson fnat: 0.09025244959721025
Mean Top-20 Spearman fnat: -0.07296964261644863
Mean AUC fnat>0.3: 0.5414061880452503 +/- 0.19074710717681587
Success@1:   27 / 113 = 23.89%
Success@2:   34 / 113 = 30.09%
Success@5:   44 / 113 = 38.94%
Success@10:  51 / 113 = 45.13%
Success@100: 75 / 113 = 66.37%
```

Interpretation:

- Non-dedup result is higher than dedup result.
- It can be reported as raw/leaky baseline only, not as fair final result.

## Current User Preferences / Conventions

- Main evaluation should be full CAPRI 113 only.
- Positive definition: `fnat > 0.3`.
- Denominator for full 113 success: all 113 targets.
- Every target should be written into CSV.
- Other old evaluation variants should not be used unless explicitly requested.
- For BioScore, use original BioScore input format, but apply TraDock-like deduplication and same CAPRI target set.
- Do not replace BioScore input with TraDock surface data.

## Common Pitfalls

- Do not trust BioScore `results.jsonl["gt"]` as fnat. It is DockQ-like.
- Do not compare old no-dedup BioScore as fair final result.
- TraDock evaluation with `--n_workers 4` stalled; use `--n_workers 1` unless the script is fixed.
- Some checkpoint files are old:
  - TraDock `epoch_030.chk` from 2026-06-14 is old.
  - New final `epoch_030.chk` was generated on 2026-06-19 after resume.
- Training `TransformerDock_best.chk` after resume is valid and points to epoch 27.

