#!/usr/bin/env bash
# Verify that a restored DB5 three-method evaluation install is runnable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRADOCK_ENV_FILE="${TRADOCK_ENV_FILE:-$PROJECT_ROOT/environment}"
if [ -f "$TRADOCK_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TRADOCK_ENV_FILE"
fi
TRADOCK_LOCAL_ENV_FILE="${TRADOCK_LOCAL_ENV_FILE:-$PROJECT_ROOT/environment.local}"
if [ -f "$TRADOCK_LOCAL_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TRADOCK_LOCAL_ENV_FILE"
fi

PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"
_DATA_ROOT="${TRADOCK_DATA_ROOT:-$HOME/tradock_data}"
PPC_ROOT="${PPC_ROOT:-${PAPER_ROOT:-$_DATA_ROOT/PPCBench}}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"
if [ -z "${HDOCK_BIN:-}" ]; then
    if [ -x "$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/hdock" ]; then
        HDOCK_BIN="$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/hdock"
    else
        HDOCK_BIN="${HDOCK_BIN:-hdock}"
    fi
fi
if [ -z "${CREATEPL_BIN:-}" ]; then
    if [ -x "$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/createpl" ]; then
        CREATEPL_BIN="$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/createpl"
    else
        CREATEPL_BIN="${CREATEPL_BIN:-createpl_linux}"
    fi
fi
if [ -z "${COLABFOLD_BIN:-}" ]; then
    if [ -x "$HOME/localcolabfold/.pixi/envs/default/bin/colabfold_batch" ]; then
        COLABFOLD_BIN="$HOME/localcolabfold/.pixi/envs/default/bin/colabfold_batch"
    else
        COLABFOLD_BIN="colabfold_batch"
    fi
fi
COLABFOLD_DATA="${COLABFOLD_DATA:-$_DATA_ROOT/colabfold_params_v3}"
AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-${DB5_EVAL_RUN_ROOT:-$_DATA_ROOT/db5_three_method_eval}/colabfold_outputs/${AF_DATASET:-DB5}}"
RUN_COLABFOLD="${RUN_COLABFOLD:-0}"
METHODS="${METHODS:-hdock alphafold lightdock}"
PYTHON_BIN="${PYTHON_BIN:-python}"
unset _DATA_ROOT

fail=0

has_method() {
    local needle="$1"
    local method
    for method in $METHODS; do
        if [ "$method" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

check_file() {
    local label="$1"
    local path="$2"
    if [ -f "$path" ]; then
        echo "[ok] $label: $path"
    else
        echo "[missing] $label: $path"
        fail=1
    fi
}

check_dir() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        echo "[ok] $label: $path"
    else
        echo "[missing] $label: $path"
        fail=1
    fi
}

check_exec() {
    local label="$1"
    local path="$2"
    if [ -x "$path" ] || command -v "$path" >/dev/null 2>&1; then
        echo "[ok] $label: $path"
    else
        echo "[missing] $label: $path"
        fail=1
    fi
}

check_command() {
    local label="$1"
    local cmd="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "[ok] $label: $(command -v "$cmd")"
    else
        echo "[missing] $label: $cmd"
        fail=1
    fi
}

echo "=== TraDock DB5 full-eval verification ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PPC_ROOT=$PPC_ROOT"
echo "METHODS=$METHODS"
echo "PYTHON_BIN=$PYTHON_BIN"
echo ""

check_file "three-method runner" "$PROJECT_ROOT/scripts/run_db5_three_method_eval.sh"
check_file "environment file" "$PROJECT_ROOT/environment"
check_file "TraDock checkpoint" "$CHECKPOINT"
check_dir "PPCBench root" "$PPC_ROOT"
check_dir "PPCBench evaluate" "$PPC_ROOT/evaluate"
check_dir "DB5 dataset" "$PPC_ROOT/dataset/DB5"
check_dir "DB5-u dataset" "$PPC_ROOT/dataset/DB5-u"

if [ -f "$PPC_ROOT/dataset/DB5/DB5.json" ]; then
    echo "[ok] DB5 targets: $(wc -l < "$PPC_ROOT/dataset/DB5/DB5.json")"
fi
if [ -f "$PPC_ROOT/dataset/DB5-u/DB5-u.json" ]; then
    echo "[ok] DB5-u targets: $(wc -l < "$PPC_ROOT/dataset/DB5-u/DB5-u.json")"
fi

if has_method hdock; then
    check_exec "HDOCK binary" "$HDOCK_BIN"
    check_exec "createpl binary" "$CREATEPL_BIN"
fi

echo ""
echo "=== Python dependencies ==="
"$PYTHON_BIN" - <<'PY' || fail=1
import importlib
mods = [
    "torch",
    "torch_geometric",
    "torch_cluster",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "Bio",
    "freesasa",
]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append((mod, str(exc)))
if missing:
    for mod, exc in missing:
        print(f"[missing] python module {mod}: {exc}")
    raise SystemExit(1)
import torch
print(f"[ok] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
print("[ok] core Python modules")
PY

if has_method alphafold; then
    echo ""
    echo "=== AlphaFold/ColabFold ==="
    echo "RUN_COLABFOLD=$RUN_COLABFOLD"
    if [ "$RUN_COLABFOLD" = "1" ]; then
        check_exec "ColabFold command" "$COLABFOLD_BIN"
        check_dir "ColabFold data" "$COLABFOLD_DATA"
    else
        echo "[info] RUN_COLABFOLD=0: will use existing outputs under $AF_OUTPUT_ROOT"
        if [ -d "$AF_OUTPUT_ROOT" ] && find "$AF_OUTPUT_ROOT" -name '*.pdb' 2>/dev/null | grep -q .; then
            echo "[ok] existing ColabFold PDBs found under $AF_OUTPUT_ROOT"
        else
            echo "[missing] no ColabFold PDBs under $AF_OUTPUT_ROOT"
            echo "          Install ColabFold and set RUN_COLABFOLD=1, or populate AF_OUTPUT_ROOT."
            echo "          Helpers: bash scripts/setup_colabfold_paths.sh"
            fail=1
        fi
    fi
fi

if has_method lightdock; then
    echo ""
    echo "=== LightDock ==="
    check_command "lightdock3_setup.py" "lightdock3_setup.py"
    check_command "lightdock3.py" "lightdock3.py"
    check_command "lgd_generate_conformations.py" "lgd_generate_conformations.py"
    if [ "$fail" -ne 0 ]; then
        echo "          Install helper: bash scripts/install_lightdock.sh"
    fi
fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "[ok] full-eval prerequisites are present for METHODS=\"$METHODS\""
else
    echo "[error] missing prerequisites; fix the items above before full evaluation"
    exit 1
fi
