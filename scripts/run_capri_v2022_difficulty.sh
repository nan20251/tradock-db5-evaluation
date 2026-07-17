#!/usr/bin/env bash
# Evaluate TraDock on CAPRI Score v2022 Difficult / Easy target lists.
#
# Modes:
#   summarize  - reuse an existing CAPRI detail CSV (fast; recommended first)
#   eval       - run examples/eval_capri_fast.py on difficult/easy target lists
#
# Examples:
#   bash scripts/run_capri_v2022_difficulty.sh summarize
#   CAPRI_DIR=/root/TraDock/data/database \
#     CHECKPOINT=Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
#     bash scripts/run_capri_v2022_difficulty.sh eval

set -euo pipefail

PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

MODE="${1:-summarize}"
CAPRI_DIR="${CAPRI_DIR:-$PROJECT_ROOT/data/database}"
CHECKPOINT="${CHECKPOINT:-Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results/capri_v2022_difficulty}"
DETAIL_CSV="${DETAIL_CSV:-$PROJECT_ROOT/results/capri_eval_113_fnat03_best_epoch027_single.csv}"

SHIRALI_DIR="${SHIRALI_DIR:-}"
if [ -z "$SHIRALI_DIR" ]; then
  for cand in \
    "$HOME/Desktop/AUC&ClassificanMetrics&SuccessRate/AUC&ClassificanMetrics&SuccessRate" \
    "/mnt/c/Users/yang.nan/Desktop/AUC&ClassificanMetrics&SuccessRate/AUC&ClassificanMetrics&SuccessRate" \
    "C:/Users/yang.nan/Desktop/AUC&ClassificanMetrics&SuccessRate/AUC&ClassificanMetrics&SuccessRate"
  do
    if [ -d "$cand" ]; then
      SHIRALI_DIR="$cand"
      break
    fi
  done
fi

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/data/capri_v2022"

summarize_with() {
  local detail="$1"
  if [ -z "${SHIRALI_DIR}" ] || [ ! -d "$SHIRALI_DIR" ]; then
    echo "[error] set SHIRALI_DIR to the Shirali scores&labels folder" >&2
    exit 1
  fi
  python -u scripts/summarize_capri_v2022_difficulty.py \
    --tradock_detail "$detail" \
    --difficult_csv "$SHIRALI_DIR/CAPRI_v2022_difficult_targets_scores&labels.csv" \
    --easy_csv "$SHIRALI_DIR/CAPRI_v2022_easy_targets_scores&labels.csv" \
    --out_dir "$OUT_DIR"
}

case "$MODE" in
  summarize)
    summarize_with "$DETAIL_CSV"
    echo "Done. See $OUT_DIR"
    ;;
  eval)
    if [ ! -f "$CHECKPOINT" ]; then
      echo "[error] missing checkpoint: $CHECKPOINT" >&2
      exit 1
    fi
    if [ ! -d "$CAPRI_DIR" ]; then
      echo "[error] missing CAPRI_DIR: $CAPRI_DIR" >&2
      exit 1
    fi
    for split in difficult easy; do
      list="$PROJECT_ROOT/data/capri_v2022/capri_v2022_${split}_targets.txt"
      out="$OUT_DIR/tradock_${split}.csv"
      echo "=== eval $split ($list) ==="
      python -u examples/eval_capri_fast.py \
        --data_dir "$CAPRI_DIR" \
        --checkpoint "$CHECKPOINT" \
        --targets_file "$list" \
        --out "$out" \
        --pos_metric classification \
        --success_denominator all \
        --score_type mdn \
        --n_workers "${N_WORKERS:-4}" \
        --resume
    done
    python - <<PY
import csv
from pathlib import Path
out_dir = Path(r"""$OUT_DIR""")
paths = [out_dir / "tradock_difficult.csv", out_dir / "tradock_easy.csv"]
rows, fields = [], None
for p in paths:
    if not p.exists():
        continue
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        rows.extend(list(r))
merged = out_dir / "tradock_difficult_easy_merged.csv"
if fields and rows:
    with merged.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {merged} n={len(rows)}")
PY
    if [ -f "$OUT_DIR/tradock_difficult_easy_merged.csv" ]; then
      summarize_with "$OUT_DIR/tradock_difficult_easy_merged.csv"
    fi
    echo "Done. See $OUT_DIR"
    ;;
  *)
    echo "Usage: $0 {summarize|eval}" >&2
    exit 2
    ;;
esac
