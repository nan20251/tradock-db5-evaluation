#!/usr/bin/env bash
# Evaluate TraDock on Shirali BM5_15complexes (DeepRank-GNN hold-out, ~7500 HADDOCK decoys).
#
# Prerequisites on the GPU server:
#   1) Extract zip, e.g.
#        mkdir -p ~/tradock_data/BM5_15complexes
#        unzip -q BM5_15complexes.zip -d ~/tradock_data/
#      so that ~/tradock_data/BM5_15complexes/PDBs/*.pdb exists.
#   2) Copy BM5_scores&labels.csv next to it (or set LABELS_CSV).
#
# Example:
#   CHECKPOINT=Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
#   BM5_15_DIR=~/tradock_data/BM5_15complexes \
#   LABELS_CSV=~/tradock_data/BM5_scores&labels.csv \
#   bash scripts/run_bm5_15_tradock_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/tradock_path_lib.sh" ]]; then
  # shellcheck source=scripts/tradock_path_lib.sh
  source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"
  tradock_source_env_files "$PROJECT_ROOT" || true
fi

BM5_15_DIR="${BM5_15_DIR:-$HOME/tradock_data/BM5_15complexes}"
LABELS_CSV="${LABELS_CSV:-$HOME/tradock_data/BM5_scores&labels.csv}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to TransformerDock_best.chk}"
OUT="${OUT:-$PROJECT_ROOT/results/bm5_15_tradock.csv}"
N_WORKERS="${N_WORKERS:-16}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT_TARGETS="${LIMIT_TARGETS:-0}"
LIMIT_PER_TARGET="${LIMIT_PER_TARGET:-0}"

if [[ ! -d "$BM5_15_DIR/PDBs" ]]; then
  echo "ERROR: missing $BM5_15_DIR/PDBs — extract BM5_15complexes.zip first" >&2
  exit 1
fi
if [[ ! -f "$LABELS_CSV" ]]; then
  echo "ERROR: missing labels CSV: $LABELS_CSV" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "BM5_15_DIR=$BM5_15_DIR"
echo "LABELS_CSV=$LABELS_CSV"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUT=$OUT"
echo "GPU=$CUDA_VISIBLE_DEVICES workers=$N_WORKERS"

extra=()
if [[ "$LIMIT_TARGETS" != "0" ]]; then
  extra+=(--limit_targets "$LIMIT_TARGETS")
fi
if [[ "$LIMIT_PER_TARGET" != "0" ]]; then
  extra+=(--limit_per_target "$LIMIT_PER_TARGET")
fi

python -u examples/eval_bm5_15_tradock.py \
  --data_dir "$BM5_15_DIR" \
  --labels_csv "$LABELS_CSV" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --n_workers "$N_WORKERS" \
  --device "$DEVICE" \
  "${extra[@]}"

echo "done -> $OUT"
echo "      -> ${OUT%.csv}.summary.csv"
echo "      -> ${OUT%.csv}.aggregate.csv"
