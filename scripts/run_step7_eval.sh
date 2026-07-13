#!/bin/bash
# Step 7: CAPRI 113 快速评估（逐 target 写入 CSV）

set -e
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DIPS_SURFACES="${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
CAPRI_DIR="${CAPRI_DIR:-$PROJECT_ROOT/data/database}"
CHECKPOINT="${CHECKPOINT:-Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4

mkdir -p results

# ============================================================
# 7.1 CAPRI Score_set 全量快速评估
# ============================================================

echo "=== 7.1 CAPRI Score_set 113 快速评估 ==="

if [ -d "$CAPRI_DIR" ] && ls "$CAPRI_DIR"/S-T*.pdb 1>/dev/null 2>&1 && ls "$CAPRI_DIR"/S-T*.csv 1>/dev/null 2>&1 && [ -f "$CHECKPOINT" ]; then
    MAX_MODELS_ARG=()
    if [ -n "${MAX_MODELS:-}" ]; then
        MAX_MODELS_ARG=(--max_models "$MAX_MODELS")
    fi
    python -u examples/eval_capri_fast.py \
        --data_dir "$CAPRI_DIR" \
        --checkpoint "$CHECKPOINT" \
        --out results/capri_eval_113_fast.csv \
        --pos_metric classification --pos_threshold 0.3 \
        --success_denominator with_positives \
        --score_type mdn \
        --n_workers "${N_WORKERS:-1}" \
        "${MAX_MODELS_ARG[@]}"
else
    echo "[跳过] CAPRI 评估：缺少完整 CAPRI 数据（$CAPRI_DIR/S-T*.pdb + .csv）或 $CHECKPOINT"
    echo "可设置 CAPRI_DIR=/path/to/database CHECKPOINT=/path/to/model.chk"
fi

echo ""
echo "=== 全部评估完成 ==="
echo "结果文件:"
ls results/capri_eval_113_fast.csv 2>/dev/null && echo "  results/capri_eval_113_fast.csv"
ls results/capri_eval_113_fast.summary.csv 2>/dev/null && echo "  results/capri_eval_113_fast.summary.csv"
