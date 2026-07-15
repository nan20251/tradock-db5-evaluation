#!/usr/bin/env bash
# Reconstruct and restore the DB5 portable data stored as git split parts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/portable_data}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tradock_repo_data}"
INSTALL_ROOT="${INSTALL_ROOT:-/}"
ARCHIVE_NAME="${ARCHIVE_NAME:-tradock_db5_eval_pack_20260714_154813_portable.tar.gz}"

mkdir -p "$WORK_DIR"

if ! compgen -G "$DATA_DIR/$ARCHIVE_NAME.part-*" >/dev/null; then
    echo "[error] missing split data parts: $DATA_DIR/$ARCHIVE_NAME.part-*"
    exit 1
fi

echo "=== Rebuild portable archive ==="
cat "$DATA_DIR/$ARCHIVE_NAME".part-* > "$WORK_DIR/$ARCHIVE_NAME"
cp "$DATA_DIR/$ARCHIVE_NAME.sha256" "$WORK_DIR/$ARCHIVE_NAME.sha256"

echo "=== Verify checksum ==="
(cd "$WORK_DIR" && shasum -a 256 -c "$ARCHIVE_NAME.sha256")

echo "=== Extract outer archive ==="
rm -rf "$WORK_DIR/extracted"
mkdir -p "$WORK_DIR/extracted"
tar -xzf "$WORK_DIR/$ARCHIVE_NAME" -C "$WORK_DIR/extracted"

INNER_ARCHIVE="$(find "$WORK_DIR/extracted" -path '*/portable_data/tradock_db5_portable_data_latest.tar.gz' -print -quit)"
if [ -z "$INNER_ARCHIVE" ]; then
    echo "[error] missing inner portable data archive"
    exit 1
fi

echo "=== Restore data to $INSTALL_ROOT ==="
tar -xzf "$INNER_ARCHIVE" -C "$INSTALL_ROOT"

echo "=== Done ==="
echo "Restored expected paths:"
echo "  /root/PPCBench"
echo "  /root/autodl-tmp/tools/hdocklite_full"
echo "  /root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk"
echo ""
echo "Verify:"
echo "  cd $PROJECT_ROOT && METHODS=hdock bash scripts/verify_full_eval.sh"
