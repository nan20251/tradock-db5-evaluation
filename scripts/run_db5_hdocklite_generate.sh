#!/bin/bash
# Generate paper-style HDOCKlite candidates for PPCBench DB5-u.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRADOCK_ENV_FILE="${TRADOCK_ENV_FILE:-$PROJECT_ROOT/environment}"
if [ -f "$TRADOCK_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TRADOCK_ENV_FILE"
    PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"
fi
PAPER_ROOT="${PAPER_ROOT:-/root/PPCBench}"
RUN_ROOT="${HDOCK_RUN_ROOT:-/root/autodl-tmp/hdock_regen_full}"
WORK_ROOT="${HDOCK_WORK_ROOT:-${RUN_ROOT}/work}"
OUT_ROOT_BASE="${HDOCK_OUT_ROOT_BASE:-${RUN_ROOT}/results}"
NMAX="${HDOCK_NMAX:-100}"
DATASETS="${HDOCK_DATASETS:-DB5-u}"

if [ -z "${HDOCK_BIN:-}" ] && [ -x /root/autodl-tmp/tools/hdocklite_full/hdock ]; then
    HDOCK_BIN=/root/autodl-tmp/tools/hdocklite_full/hdock
else
    HDOCK_BIN="${HDOCK_BIN:-hdock}"
fi
if [ -z "${CREATEPL_BIN:-}" ] && [ -x /root/autodl-tmp/tools/hdocklite_full/createpl ]; then
    CREATEPL_BIN=/root/autodl-tmp/tools/hdocklite_full/createpl
else
    CREATEPL_BIN="${CREATEPL_BIN:-createpl_linux}"
fi
unset SCRIPT_DIR

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
