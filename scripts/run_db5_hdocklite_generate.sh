#!/usr/bin/env bash
# Generate paper-style HDOCKlite candidates for PPCBench DB5-u.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=scripts/tradock_path_lib.sh
source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"
tradock_source_env_files "$PROJECT_ROOT"
PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"

PAPER_ROOT="${PAPER_ROOT:-${PPC_ROOT:-$(tradock_default_ppc_root)}}"
RUN_ROOT="${HDOCK_RUN_ROOT:-${DB5_EVAL_RUN_ROOT:-$(tradock_data_root)/db5_three_method_eval}/hdock}"
WORK_ROOT="${HDOCK_WORK_ROOT:-$RUN_ROOT/work}"
OUT_ROOT_BASE="${HDOCK_OUT_ROOT_BASE:-${DB5_EVAL_RESULTS_ROOT:-${DB5_EVAL_RUN_ROOT:-$(tradock_data_root)/db5_three_method_eval}/results}}"
NMAX="${HDOCK_NMAX:-100}"
DATASETS="${HDOCK_DATASETS:-DB5-u}"
HDOCK_BIN="${HDOCK_BIN:-$(tradock_default_hdock_bin)}"
CREATEPL_BIN="${CREATEPL_BIN:-$(tradock_default_createpl_bin)}"

cd "$PROJECT_ROOT"

echo "=== Generate HDOCKlite candidates ==="
echo "PAPER_ROOT=$PAPER_ROOT"
echo "OUT_ROOT_BASE=$OUT_ROOT_BASE"
echo "WORK_ROOT=$WORK_ROOT"
echo "NMAX=$NMAX"
echo "DATASETS=$DATASETS"
echo "HDOCK_BIN=$HDOCK_BIN"
echo "CREATEPL_BIN=$CREATEPL_BIN"
echo ""

if [ ! -d "$PAPER_ROOT/dataset" ]; then
    echo "[error] missing PPCBench dataset directory: $PAPER_ROOT/dataset"
    exit 1
fi

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    mkdir -p "$OUT_ROOT_BASE/$dataset" "$WORK_ROOT"
    HDOCK_ARGS=()
    if [ -n "${HDOCK_LIMIT:-}" ]; then
        HDOCK_ARGS+=(--limit "$HDOCK_LIMIT")
    fi
    if [ -n "${HDOCK_TARGETS:-}" ]; then
        HDOCK_ARGS+=(--targets "$HDOCK_TARGETS")
    fi
    if [ -n "${HDOCK_OVERWRITE:-}" ]; then
        HDOCK_ARGS+=(--overwrite)
    fi
    python -u examples/generate_db5_hdocklite_candidates.py \
        --paper_root "$PAPER_ROOT" \
        --dataset "$dataset" \
        --out_root "$OUT_ROOT_BASE/$dataset" \
        --work_root "$WORK_ROOT" \
        --hdock_bin "$HDOCK_BIN" \
        --createpl_bin "$CREATEPL_BIN" \
        --nmax "$NMAX" \
        "${HDOCK_ARGS[@]}"
done

echo "=== HDOCKlite candidate generation complete ==="
