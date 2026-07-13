#!/bin/bash
# Generate paper-style HDOCKlite candidates for PPCBench DB5 / DB5-u.

set -e

PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PAPER_ROOT="${PAPER_ROOT:-/root/PPCBench}"
WORK_ROOT="${HDOCK_WORK_ROOT:-/root/autodl-tmp/hdocklite_work}"
NMAX="${HDOCK_NMAX:-5}"
DATASETS="${HDOCK_DATASETS:-DB5 DB5-u}"
HDOCK_BIN="${HDOCK_BIN:-hdock}"
CREATEPL_BIN="${CREATEPL_BIN:-createpl_linux}"

cd "$PROJECT_ROOT"

echo "=== Generate HDOCKlite candidates ==="
echo "PAPER_ROOT=$PAPER_ROOT"
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
    python -u examples/generate_db5_hdocklite_candidates.py \
        --paper_root "$PAPER_ROOT" \
        --dataset "$dataset" \
        --work_root "$WORK_ROOT" \
        --hdock_bin "$HDOCK_BIN" \
        --createpl_bin "$CREATEPL_BIN" \
        --nmax "$NMAX" \
        ${HDOCK_LIMIT:+--limit "$HDOCK_LIMIT"} \
        ${HDOCK_TARGETS:+--targets "$HDOCK_TARGETS"} \
        ${HDOCK_OVERWRITE:+--overwrite}
done

echo "=== HDOCKlite candidate generation complete ==="
