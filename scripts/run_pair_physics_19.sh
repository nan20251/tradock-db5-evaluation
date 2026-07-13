#!/bin/bash
# Train and evaluate the 19-channel pair-aware physics TraDock model.

set -e
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DIPS_SURFACES="${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
CAPRI_DIR="${CAPRI_DIR:-/root/TraDock_backup_20260618_131456/data/database}"
SAVE_DIR="${SAVE_DIR:-Trained_models/pretrain_pair_physics_19}"
INIT_FROM="${INIT_FROM:-Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"
PAIRS_CSV="${PAIRS_CSV:-results/dips_with_sasa_full.filtered_pairs.csv}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_WORKERS_EVAL="${N_WORKERS_EVAL:-1}"
MAX_PAIRS="${MAX_PAIRS:-}"
MAX_MODELS_EVAL="${MAX_MODELS_EVAL:-}"
OUT_PREFIX="${OUT_PREFIX:-capri_eval_113_pair_physics_19}"

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "examples/train.py" ]; then
    echo "[错误] 未找到 examples/train.py，项目目录不正确: $PROJECT_ROOT" >&2
    exit 1
fi
if [ ! -f "$DIPS_SURFACES/pairs.csv" ]; then
    echo "[错误] 未找到 DIPS surface pairs.csv: $DIPS_SURFACES/pairs.csv" >&2
    exit 1
fi

mkdir -p Trained_models results "$SAVE_DIR"

python scripts/filter_bad_dips_pairs.py \
    --input "$DIPS_SURFACES/pairs.csv" \
    --output "$PAIRS_CSV" \
    --exclude 1u0c_A_B 1yk0_A_B \
    --exclude_file data/dips/exclude_capri.txt

INIT_ARGS=()
if [ -f "$INIT_FROM" ]; then
    INIT_ARGS=(--init_from "$INIT_FROM")
fi
MAX_PAIR_ARGS=()
if [ -n "$MAX_PAIRS" ]; then
    MAX_PAIR_ARGS=(--max_pairs "$MAX_PAIRS")
fi

python -u examples/train.py \
    --data_dir "$DIPS_SURFACES" \
    --pairs_csv "$PAIRS_CSV" \
    --save_dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --in_channels 19 \
    --contrast_weight 0.0 \
    "${MAX_PAIR_ARGS[@]}" \
    "${INIT_ARGS[@]}"

if [ -d "$CAPRI_DIR" ] && ls "$CAPRI_DIR"/S-T*.pdb 1>/dev/null 2>&1; then
    MAX_MODELS_ARG=()
    if [ -n "$MAX_MODELS_EVAL" ]; then
        MAX_MODELS_ARG=(--max_models "$MAX_MODELS_EVAL")
    fi
    python -u examples/eval_capri_fast.py \
        --data_dir "$CAPRI_DIR" \
        --checkpoint "$SAVE_DIR/TransformerDock_best.chk" \
        --out "results/${OUT_PREFIX}.csv" \
        --pos_metric fnat --pos_threshold 0.3 \
        --success_denominator all \
        --score_type mdn \
        --n_workers "$N_WORKERS_EVAL" \
        "${MAX_MODELS_ARG[@]}"
else
    echo "[跳过] CAPRI 评估：未找到 $CAPRI_DIR/S-T*.pdb"
fi

echo "=== pair-aware physics 19-channel run complete ==="
echo "model:   $SAVE_DIR/TransformerDock_best.chk"
echo "result:  results/${OUT_PREFIX}.csv"
