#!/usr/bin/env bash
# Detect LocalColabFold / ColabFold installs and write paths into environment.local.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-$PROJECT_ROOT/environment.local}"
# shellcheck source=scripts/tradock_path_lib.sh
source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"

DATA_ROOT="$(tradock_data_root)"
COLABFOLD_BIN_FOUND="$(tradock_default_colabfold_bin)"
COLABFOLD_DATA_FOUND="$(tradock_default_colabfold_data)"

mkdir -p "$(dirname "$LOCAL_ENV_FILE")"
touch "$LOCAL_ENV_FILE"

upsert() {
    local key="$1"
    local value="$2"
    local tmp
    tmp="$(mktemp)"
    awk -v k="export ${key}=" -v v="export ${key}=\"${value}\"" '
        BEGIN {done=0}
        index($0, k)==1 {print v; done=1; next}
        {print}
        END {if (!done) print v}
    ' "$LOCAL_ENV_FILE" > "$tmp"
    mv "$tmp" "$LOCAL_ENV_FILE"
}

upsert DB5_EVAL_RUN_ROOT "$DATA_ROOT/db5_three_method_eval"

if [ -x "$COLABFOLD_BIN_FOUND" ] || command -v "$COLABFOLD_BIN_FOUND" >/dev/null 2>&1; then
    upsert COLABFOLD_BIN "$COLABFOLD_BIN_FOUND"
    echo "[ok] COLABFOLD_BIN=$COLABFOLD_BIN_FOUND"
else
    echo "[missing] colabfold_batch not found under LocalColabFold/conda paths"
    echo "          Install LocalColabFold, then re-run this script."
fi

if [ -d "$COLABFOLD_DATA_FOUND" ]; then
    upsert COLABFOLD_DATA "$COLABFOLD_DATA_FOUND"
    echo "[ok] COLABFOLD_DATA=$COLABFOLD_DATA_FOUND"
else
    upsert COLABFOLD_DATA "$DATA_ROOT/colabfold_params_v3"
    echo "[warn] ColabFold params dir not found; wrote placeholder:"
    echo "       $DATA_ROOT/colabfold_params_v3"
    echo "       Place AF2/multimer params there, or set COLABFOLD_DATA manually."
fi

echo "Wrote/updated: $LOCAL_ENV_FILE"
echo "Verify:"
echo "  cd $PROJECT_ROOT && METHODS=alphafold RUN_COLABFOLD=1 bash scripts/verify_full_eval.sh"
