#!/bin/bash
# Step 6: 用一批 native 经 LightDock 生成 decoy 后做 native-vs-decoy 微调。
#
# 输入 PDB_DIR 支持 examples/run_lightdock.py 的两种格式：
#   1) <stem>.pdb + <stem>.chains
#   2) <stem>_receptor.pdb + <stem>_ligand.pdb
#
# 常用：
#   PDB_DIR=/path/to/native_pdbs TARGET_LIMIT=10 bash scripts/run_step6_lightdock_finetune.sh

set -e

PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PDB_DIR="${PDB_DIR:-}"
LD_ROOT="${LD_ROOT:-results/lightdock_native_decoys}"
NATIVE_SURFACES="${NATIVE_SURFACES:-results/lightdock_native_surfaces}"
DECOY_SURFACES="${DECOY_SURFACES:-results/lightdock_decoy_surfaces}"
SAVE_DIR="${SAVE_DIR:-Trained_models/finetune_lightdock_native_vs_decoy}"
INIT_FROM="${INIT_FROM:-Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"

TARGET_LIMIT="${TARGET_LIMIT:-}"
SWARMS="${SWARMS:-20}"
GLOWWORMS="${GLOWWORMS:-100}"
STEPS="${STEPS:-50}"
LD_WORKERS="${LD_WORKERS:-1}"
PREP_WORKERS="${PREP_WORKERS:-4}"
VOXEL_SIZE="${VOXEL_SIZE:-3.5}"
MAX_DECOYS_PER_TARGET="${MAX_DECOYS_PER_TARGET:-500}"
MAX_PER_STEM_DECOY="${MAX_PER_STEM_DECOY:-300}"

EPOCHS="${EPOCHS:-20}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-300}"
VAL_BATCHES_PER_EPOCH="${VAL_BATCHES_PER_EPOCH:-60}"
VAL_TARGETS="${VAL_TARGETS:-2}"
DECOY_PER_NATIVE="${DECOY_PER_NATIVE:-15}"
LR="${LR:-5e-5}"
LAMBDA_RANK="${LAMBDA_RANK:-1.0}"
LAMBDA_MSE="${LAMBDA_MSE:-0.3}"
LAMBDA_MDN="${LAMBDA_MDN:-0.3}"

mkdir -p results Trained_models "$LD_ROOT" "$NATIVE_SURFACES" "$DECOY_SURFACES" "$SAVE_DIR"

if [ "${SKIP_LIGHTDOCK:-0}" != "1" ]; then
    if [ -z "$PDB_DIR" ] || [ ! -d "$PDB_DIR" ]; then
        echo "[错误] 请设置 PDB_DIR=/path/to/native_pdbs，或设置 SKIP_LIGHTDOCK=1 复用已有 LD_ROOT" >&2
        exit 1
    fi
    LIMIT_ARG=()
    if [ -n "$TARGET_LIMIT" ]; then
        LIMIT_ARG=(--limit "$TARGET_LIMIT")
    fi
    python examples/run_lightdock.py \
        --pdb_dir "$PDB_DIR" \
        --out_root "$LD_ROOT" \
        --swarms "$SWARMS" \
        --glowworms "$GLOWWORMS" \
        --steps "$STEPS" \
        --workers "$LD_WORKERS" \
        "${LIMIT_ARG[@]}"
fi

LIMIT_ARG=()
if [ -n "$TARGET_LIMIT" ]; then
    LIMIT_ARG=(--limit "$TARGET_LIMIT")
fi
python examples/prep_lightdock_native_surfaces.py \
    --ld_root "$LD_ROOT" \
    --out_dir "$NATIVE_SURFACES" \
    --voxel_size "$VOXEL_SIZE" \
    --workers "$PREP_WORKERS" \
    "${LIMIT_ARG[@]}"

MAX_DECOY_ARG=()
if [ -n "$MAX_DECOYS_PER_TARGET" ]; then
    MAX_DECOY_ARG=(--max_per_target "$MAX_DECOYS_PER_TARGET")
fi
LIMIT_ARG=()
if [ -n "$TARGET_LIMIT" ]; then
    LIMIT_ARG=(--limit "$TARGET_LIMIT")
fi
python examples/prep_lightdock_decoys.py \
    --ld_root "$LD_ROOT" \
    --out_dir "$DECOY_SURFACES" \
    --voxel_size "$VOXEL_SIZE" \
    --workers "$PREP_WORKERS" \
    "${MAX_DECOY_ARG[@]}" \
    "${LIMIT_ARG[@]}"

if [ ! -f "$DECOY_SURFACES/decoys.csv" ]; then
    echo "[错误] 未找到 $DECOY_SURFACES/decoys.csv" >&2
    exit 1
fi
if [ ! -f "$NATIVE_SURFACES/pairs.csv" ]; then
    echo "[错误] 未找到 $NATIVE_SURFACES/pairs.csv" >&2
    exit 1
fi

INIT_ARGS=()
if [ -f "$INIT_FROM" ]; then
    INIT_ARGS=(--init_from "$INIT_FROM")
fi

python examples/train_native_vs_decoy_v2.py \
    --native_dir "$NATIVE_SURFACES" \
    --decoy_dir "$DECOY_SURFACES" \
    --decoy_csv "$DECOY_SURFACES/decoys.csv" \
    --save_dir "$SAVE_DIR" \
    "${INIT_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --batches_per_epoch "$BATCHES_PER_EPOCH" \
    --val_batches_per_epoch "$VAL_BATCHES_PER_EPOCH" \
    --val_targets "$VAL_TARGETS" \
    --decoy_per_native "$DECOY_PER_NATIVE" \
    --max_per_stem_decoy "$MAX_PER_STEM_DECOY" \
    --lr "$LR" \
    --lambda_rank "$LAMBDA_RANK" \
    --lambda_mse "$LAMBDA_MSE" \
    --lambda_mdn "$LAMBDA_MDN" \
    --in_channels 11

echo "=== LightDock 微调完成 ==="
echo "model: $SAVE_DIR/TransformerDock_best.chk"
echo "log:   $SAVE_DIR/training_log.csv"
