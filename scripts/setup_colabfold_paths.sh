#!/usr/bin/env bash
# Detect LocalColabFold / ColabFold installs and write paths into environment.local.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-$PROJECT_ROOT/environment.local}"
HOME_DIR="${HOME:-/root}"
DATA_ROOT="${TRADOCK_DATA_ROOT:-$HOME_DIR/tradock_data}"

candidates=(
    "$HOME_DIR/localcolabfold/.pixi/envs/default/bin/colabfold_batch"
    "$HOME_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch"
    "$HOME_DIR/conda_envs/colabfold/bin/colabfold_batch"
    "$DATA_ROOT/conda_envs/colabfold/bin/colabfold_batch"
)

COLABFOLD_BIN_FOUND=""
for path in "${candidates[@]}"; do
    if [ -x "$path" ]; then
        COLABFOLD_BIN_FOUND="$path"
        break
    fi
done

data_candidates=(
    "$DATA_ROOT/colabfold_params_v3"
    "$HOME_DIR/autodl-tmp/colabfold_params_v3"
    "$HOME_DIR/.cache/colabfold"
    /root/autodl-tmp/colabfold_params_v3
)

COLABFOLD_DATA_FOUND=""
for path in "${data_candidates[@]}"; do
    if [ -d "$path" ]; then
        COLABFOLD_DATA_FOUND="$path"
        break
    fi
done

mkdir -p "$(dirname "$LOCAL_ENV_FILE")"
touch "$LOCAL_ENV_FILE"

upsert() {
    local key="$1"
    local value="$2"
    if grep -q "^export ${key}=" "$LOCAL_ENV_FILE" 2>/dev/null; then
        # portable in-place edit without relying on GNU sed -i portability
        local tmp
        tmp="$(mktemp)"
        awk -v k="export ${key}=" -v v="export ${key}=\"${value}\"" '
            BEGIN {done=0}
            index($0, k)==1 {print v; done=1; next}
            {print}
            END {if (!done) print v}
        ' "$LOCAL_ENV_FILE" > "$tmp"
        mv "$tmp" "$LOCAL_ENV_FILE"
    else
        echo "export ${key}=\"${value}\"" >> "$LOCAL_ENV_FILE"
    fi
}

upsert DB5_EVAL_RUN_ROOT "$DATA_ROOT/db5_three_method_eval"

if [ -n "$COLABFOLD_BIN_FOUND" ]; then
    upsert COLABFOLD_BIN "$COLABFOLD_BIN_FOUND"
    echo "[ok] COLABFOLD_BIN=$COLABFOLD_BIN_FOUND"
else
    echo "[missing] colabfold_batch not found under LocalColabFold/conda paths"
    echo "          Install LocalColabFold, then re-run this script."
fi

if [ -n "$COLABFOLD_DATA_FOUND" ]; then
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
echo "  cd $PROJECT_ROOT && METHODS=alphafold bash scripts/verify_full_eval.sh"
