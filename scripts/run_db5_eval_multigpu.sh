#!/usr/bin/env bash
# Multi-GPU TraDock DB5/DB5-u rerank helper.
#
# Splits targets with --shard i/N, one process per GPU.
# Enables AMP (default in eval script) + CPU surface workers.
#
# Top-N example:
#   GPUS=3,6,7 N_WORKERS=8 \
#   PAPER_ROOT=$HOME/tradock_data/db5_three_method_eval/PPCBench_eval \
#   DATASET=DB5-u \
#   POSE_MODELS=$(python - <<'PY'
# print(','.join(f'hdock_{i}' for i in range(1,101)))
# PY
# ) \
#   CHECKPOINT=$HOME/tradock-db5-evaluation/Trained_models/.../best.chk \
#   OUT_PREFIX=$HOME/tradock-db5-evaluation/results/tradock_DB5-u_hdock_top100 \
#   bash scripts/run_db5_eval_multigpu.sh
#
# ALL decoys example:
#   GPUS=3,6,7 ALL_DECOYS=1 POSE_PREFIX=hdock \
#   PAPER_ROOT=... DATASET=DB5-u CHECKPOINT=... \
#   OUT_PREFIX=.../tradock_DB5-u_hdock_all \
#   bash scripts/run_db5_eval_multigpu.sh
#
# After all shards finish, merge:
#   python examples/merge_tradock_shards.py --prefix "$OUT_PREFIX" --n_shards 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="${GPUS:?set GPUS e.g. 3,6,7}"
PAPER_ROOT="${PAPER_ROOT:?}"
DATASET="${DATASET:?}"
CHECKPOINT="${CHECKPOINT:?}"
OUT_PREFIX="${OUT_PREFIX:?}"
N_WORKERS="${N_WORKERS:-8}"
SCORE_TYPE="${SCORE_TYPE:-mdn}"
MIN_TARGETS="${MIN_TARGETS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ALL_DECOYS="${ALL_DECOYS:-0}"
POSE_PREFIX="${POSE_PREFIX:-}"
POSE_MODELS="${POSE_MODELS:-}"

if [ "$ALL_DECOYS" = "1" ] || [ "${ALL_DECOYS,,}" = "true" ] || [ "${ALL_DECOYS,,}" = "yes" ]; then
  if [ -z "$POSE_PREFIX" ]; then
    echo "[error] ALL_DECOYS=1 requires POSE_PREFIX (e.g. hdock or lightdock)"
    exit 1
  fi
  POSE_ARGS=(--all_decoys --pose_prefix "$POSE_PREFIX")
elif [ -n "$POSE_MODELS" ]; then
  POSE_ARGS=(--pose_models "$POSE_MODELS")
else
  echo "[error] set POSE_MODELS=hdock_1,... or ALL_DECOYS=1 POSE_PREFIX=hdock"
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
if [ "$N" -lt 1 ]; then
  echo "[error] empty GPUS"
  exit 1
fi

mkdir -p "$(dirname "$OUT_PREFIX")"
LOG_DIR="$(dirname "$OUT_PREFIX")/logs"
mkdir -p "$LOG_DIR"

echo "=== multi-GPU TraDock eval ==="
echo "GPUS=$GPUS  shards=$N  n_workers=$N_WORKERS  amp=on"
echo "dataset=$DATASET"
if [ -n "$POSE_PREFIX" ] && [ "$ALL_DECOYS" != "0" ]; then
  echo "mode=all_decoys  prefix=$POSE_PREFIX"
else
  echo "mode=pose_list  n_models=$(awk -F, '{print NF}' <<<"$POSE_MODELS")"
fi
echo "out_prefix=$OUT_PREFIX"

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  out="${OUT_PREFIX}.shard${i}of${N}.csv"
  log="${LOG_DIR}/$(basename "$OUT_PREFIX").shard${i}of${N}.log"
  echo "launch shard $i/$N on GPU $gpu -> $out"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u examples/eval_db5_paper_tradock.py \
    --paper_root "$PAPER_ROOT" \
    --dataset "$DATASET" \
    "${POSE_ARGS[@]}" \
    --checkpoint "$CHECKPOINT" \
    --out "$out" \
    --score_type "$SCORE_TYPE" \
    --shard "${i}/${N}" \
    --n_workers "$N_WORKERS" \
    --amp \
    --min_targets "$MIN_TARGETS" \
    $EXTRA_ARGS \
    >"$log" 2>&1 &
  PIDS+=("$!")
done

echo "PIDs: ${PIDS[*]}"
echo "logs: $LOG_DIR"
echo "wait with: wait ${PIDS[*]}"
echo "merge after done:"
echo "  python examples/merge_tradock_shards.py --prefix \"$OUT_PREFIX\" --n_shards $N"
