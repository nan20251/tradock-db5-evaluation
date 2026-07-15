# TraDock

Protein-protein docking scoring and CAPRI/DB5 evaluation code.

Current project convention: model inputs use the 11-dimensional surface feature
set from `transformerdock/utils/data.py`. The experimental 19-dimensional
pair-aware path has been removed from active scripts.

## Setup

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tradock
```

2. Install PyTorch and GPU support for the target machine.

Use the PyTorch selector for the exact CUDA/runtime combination on the machine.
The AutoDL runs used PyTorch 2.4.1 with CUDA 12.1.

3. Install PyTorch Geometric native extensions matching that PyTorch/CUDA build:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric
```

4. Run the environment check:

```bash
python scripts/quick_check.py
```

## Common Workflows

Pretrain on DIPS surface data:

```bash
DIPS_SURFACES=/path/to/dips_with_sasa_full bash scripts/run_step2_pretrain.sh
```

Evaluate the current checkpoint on CAPRI:

```bash
CAPRI_DIR=/path/to/capri/database \
CHECKPOINT=Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
bash scripts/run_step7_eval.sh
```

Generate LightDock decoys from a small native set and fine-tune:

```bash
PDB_DIR=/path/to/native_pdbs TARGET_LIMIT=10 \
bash scripts/run_step6_lightdock_finetune.sh
```

Run DB5 paper-style evaluation:

```bash
bash scripts/run_step8_eval_db5_paper.sh
```

Restore the portable DB5 three-method evaluation on a new server:

```bash
git clone https://github.com/nan20251/tradock-db5-evaluation.git /root/TraDock
cd /root/TraDock
bash scripts/restore_repo_data.sh
METHODS=hdock bash scripts/verify_full_eval.sh
```

The DB5 portable data is also stored directly in the repository as split files
under `portable_data/`, so the HDOCK evaluation data can be restored without
GitHub Release downloads. If you want to use Release assets and optional packed
conda environments instead:

For non-root users, clone anywhere and run the same restore command:

```bash
git clone https://github.com/nan20251/tradock-db5-evaluation.git ~/tradock-db5-evaluation
cd ~/tradock-db5-evaluation
bash scripts/restore_repo_data.sh
METHODS=hdock bash scripts/verify_full_eval.sh
```

By default this restores data to `~/tradock_data` and writes
`environment.local`, which is auto-sourced by the DB5 evaluation scripts.

If you want to use Release assets and optional packed conda environments
instead:

```bash
bash scripts/bootstrap_full_eval.sh
```

For software-only setup without packed conda environments:

```bash
conda env create -f environment_full.yml
conda activate tradock-full
bash scripts/verify_full_eval.sh
```

Run only the HDOCK sampling pool plus TraDock reranking:

```bash
METHODS=hdock bash scripts/run_db5_three_method_eval.sh
```

Install LightDock (into the active conda env), then run:

```bash
bash scripts/install_lightdock.sh
METHODS=lightdock bash scripts/verify_full_eval.sh
METHODS=lightdock bash scripts/run_db5_three_method_eval.sh
```

For AlphaFold/ColabFold, install LocalColabFold under `$HOME/localcolabfold`, place AF2 params under `$HOME/tradock_data/colabfold_params_v3`, then:

```bash
bash scripts/setup_colabfold_paths.sh
METHODS=alphafold RUN_COLABFOLD=1 bash scripts/verify_full_eval.sh
METHODS=alphafold RUN_COLABFOLD=1 bash scripts/run_db5_three_method_eval.sh
```

Defaults prefer `$HOME/tradock_data` over `/root/...` so non-root servers work without manual path exports.

## Important Paths

- `transformerdock/`: model and data-loading package
- `examples/`: training, evaluation, and dataset-preparation entry points
- `scripts/`: operational helpers for checks, filtering, training, and evaluation
- `data/`: lightweight metadata and exclusion lists
- `Trained_models/`: local checkpoints
- `results/` and `results_remote_*`: local and synced evaluation outputs

See `PROJECT_GUIDE.md` for the compact project map and `RUNBOOK.md` for
operational notes from previous remote runs.
