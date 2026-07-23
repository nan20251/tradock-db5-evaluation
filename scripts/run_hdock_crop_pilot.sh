#!/usr/bin/env bash
# Pilot: TraDock HDock all500 with interface crop, first N targets.
#
# Example (AIDD):
#   conda activate tradock
#   cd ~/tradock-db5-evaluation
#   git pull
#   GPU=0 LIMIT=10 MAX_POSES=500 bash scripts/run_hdock_crop_pilot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"
tradock_source_env_files "$PROJECT_ROOT"

GPU="${GPU:-0}"
LIMIT="${LIMIT:-10}"
MAX_POSES="${MAX_POSES:-500}"
CROP_THRESHOLD="${CROP_THRESHOLD:-10}"
N_WORKERS="${N_WORKERS:-8}"
DATASET="${DATASET:-DB5-u}"
POSE_PREFIX="${POSE_PREFIX:-hdock}"

PPC_ROOT="${PPC_ROOT:-$(tradock_default_ppc_root)}"
RUN_ROOT="${DB5_EVAL_RUN_ROOT:-$(tradock_data_root)/db5_three_method_eval}"
RESULTS_ROOT="${DB5_EVAL_RESULTS_ROOT:-$RUN_ROOT/results}"
PAPER_EVAL_ROOT="${DB5_EVAL_PAPER_ROOT:-$RUN_ROOT/PPCBench_eval}"
CHECKPOINT="${CHECKPOINT:-$(tradock_default_checkpoint "$PROJECT_ROOT")}"
OUT="${OUT:-$PROJECT_ROOT/results/tradock_${DATASET}_hdock_all${MAX_POSES}_crop${CROP_THRESHOLD}_n${LIMIT}.csv}"

safe_link() {
  local src="$1"
  local dst="$2"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo "[error] refusing to replace non-symlink path: $dst"
    exit 1
  fi
  ln -sfn "$src" "$dst"
}

setup_eval_root() {
  local dataset="$1"
  mkdir -p "$RESULTS_ROOT/$dataset" "$PAPER_EVAL_ROOT/dataset" "$PAPER_EVAL_ROOT/results"
  safe_link "$PPC_ROOT/evaluate" "$PAPER_EVAL_ROOT/evaluate"
  safe_link "$PPC_ROOT/dataset/$dataset" "$PAPER_EVAL_ROOT/dataset/$dataset"
  safe_link "$RESULTS_ROOT/$dataset" "$PAPER_EVAL_ROOT/results/$dataset"
}

if [ ! -d "$PPC_ROOT/dataset/$DATASET" ] || [ ! -d "$PPC_ROOT/evaluate" ]; then
  echo "[error] missing PPCBench dataset/evaluate under $PPC_ROOT"
  exit 1
fi
if [ ! -d "$RESULTS_ROOT/$DATASET" ]; then
  echo "[error] missing HDock results: $RESULTS_ROOT/$DATASET"
  echo "Expected pose dirs like $RESULTS_ROOT/$DATASET/hdock_1/<PDB>/..."
  exit 1
fi

setup_eval_root "$DATASET"
# Eval must use PPCBench_eval (symlinks to dataset + HDock results), not raw PPCBench.
PAPER_ROOT="$PAPER_EVAL_ROOT"
if [ ! -d "$PAPER_ROOT/results/$DATASET" ]; then
  echo "[error] missing eval results link: $PAPER_ROOT/results/$DATASET"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
LOG_DIR="$(dirname "$OUT")/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(basename "${OUT%.csv}").log"

echo "=== HDock crop pilot ==="
echo "GPU=$GPU  limit=$LIMIT  max_poses=$MAX_POSES  crop=${CROP_THRESHOLD}A"
echo "ppc_root=$PPC_ROOT"
echo "results_root=$RESULTS_ROOT"
echo "paper_root=$PAPER_ROOT"
echo "checkpoint=$CHECKPOINT"
echo "out=$OUT"
echo "log=$LOG"

CUDA_VISIBLE_DEVICES="$GPU" python -u examples/eval_db5_paper_tradock.py \
  --paper_root "$PAPER_ROOT" \
  --dataset "$DATASET" \
  --all_decoys --pose_prefix "$POSE_PREFIX" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --score_type mdn \
  --limit "$LIMIT" \
  --max_poses "$MAX_POSES" \
  --crop_interface \
  --crop_threshold "$CROP_THRESHOLD" \
  --n_workers "$N_WORKERS" \
  --amp \
  --min_targets 1 \
  2>&1 | tee "$LOG"

SUMMARY="${OUT%.csv}.summary.csv"
echo
echo "Done. Summary: $SUMMARY"
python - <<PY
import csv
path = r'''$SUMMARY'''
rows = [r for r in csv.DictReader(open(path)) if r.get('status') == 'done']
n = len(rows) or 1
def m(k):
    return sum(int(float(r.get(k) or 0)) for r in rows) / n
print(
    'done', len(rows),
    'T@1=%.1f%% T@10=%.1f%% T@100=%.1f%% '
    'P@1=%.1f%% P@10=%.1f%% P@100=%.1f%% '
    'O@1=%.1f%% with_pos=%.0f' % (
        100 * m('tradock_success@1'),
        100 * m('tradock_success@10'),
        100 * m('tradock_success@100'),
        100 * m('paper_success@1'),
        100 * m('paper_success@10'),
        100 * m('paper_success@100'),
        100 * m('oracle_success@1'),
        sum(int(float(r.get('n_success_available') or 0)) > 0 for r in rows),
    )
)
print('Compare these T@K to the same first %d targets in your full-surface run.' % len(rows))
PY
