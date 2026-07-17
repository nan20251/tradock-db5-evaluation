#!/usr/bin/env bash
# Literature-style OpenFold3/AF3 DockQ success-rate evaluation (no TraDock).
#
# Pipeline:
#   1) (optional) RUN_OPENFOLD3=1  generate predictions
#   2) compute DockQ vs DB5 native; rank by OF3/AF3 confidence
#   3) report Success@1 / Success@5 / oracle (DockQ >= 0.23)
#
# Example:
#   RUN_OPENFOLD3=1 OF3_LIMIT=1 bash scripts/run_of3_dockq_success.sh
#   RUN_OPENFOLD3=0 bash scripts/run_of3_dockq_success.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=scripts/tradock_path_lib.sh
source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"
tradock_source_env_files "$PROJECT_ROOT"
PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"

PPC_ROOT="${PPC_ROOT:?set PPC_ROOT in environment.local}"
RUN_ROOT="${DB5_EVAL_RUN_ROOT:?}"
OUT_DIR="${DB5_EVAL_OUT_DIR:-$PROJECT_ROOT/results}"

OF3_DATASET="${OF3_DATASET:-DB5}"
OF3_JSON_DIR="${OF3_JSON_DIR:-$RUN_ROOT/openfold3_jsons/$OF3_DATASET}"
OF3_OUTPUT_ROOT="${OF3_OUTPUT_ROOT:-$RUN_ROOT/openfold3_outputs/$OF3_DATASET}"
OF3_BIN="${OF3_BIN:-run_openfold}"
OF3_NUM_SEEDS="${OF3_NUM_SEEDS:-5}"
OF3_NUM_SAMPLES="${OF3_NUM_SAMPLES:-5}"
OF3_USE_MSA_SERVER="${OF3_USE_MSA_SERVER:-1}"
OF3_EXTRA_ARGS="${OF3_EXTRA_ARGS:-}"
OF3_BUNDLE="${OF3_BUNDLE:-0}"
RUN_OPENFOLD3="${RUN_OPENFOLD3:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR" "$OF3_JSON_DIR" "$OF3_OUTPUT_ROOT"

echo "=== OF3/AF3 DockQ success evaluation ==="
echo "pred_root=$OF3_OUTPUT_ROOT"
echo "RUN_OPENFOLD3=$RUN_OPENFOLD3"
echo ""

JSON_ARGS=(--format openfold3)
if [ -n "${OF3_TARGETS:-}" ]; then
    JSON_ARGS+=(--targets "$OF3_TARGETS")
fi
if [ -n "${OF3_LIMIT:-}" ]; then
    JSON_ARGS+=(--limit "$OF3_LIMIT")
fi
if [ "$OF3_BUNDLE" = "1" ]; then
    JSON_ARGS+=(--bundle)
fi

python examples/prepare_db5_alphafold3_jsons.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$OF3_DATASET" \
    --out_dir "$OF3_JSON_DIR" \
    "${JSON_ARGS[@]}"

if [ "$RUN_OPENFOLD3" = "1" ]; then
    if ! command -v "$OF3_BIN" >/dev/null 2>&1 && [ ! -x "$OF3_BIN" ]; then
        echo "[error] missing $OF3_BIN; run: bash scripts/install_openfold3.sh"
        exit 1
    fi
    MSA_FLAG="--use-msa-server"
    if [ "$OF3_USE_MSA_SERVER" = "0" ]; then
        MSA_FLAG="--use-msa-server=False"
    fi
    run_one() {
        local json_path="$1"
        echo "=== OpenFold3 predict: $(basename "$json_path") ==="
        # shellcheck disable=SC2086
        "$OF3_BIN" predict \
            --query-json="$json_path" \
            --output-dir="$OF3_OUTPUT_ROOT" \
            --num-model-seeds="$OF3_NUM_SEEDS" \
            --num-diffusion-samples="$OF3_NUM_SAMPLES" \
            $MSA_FLAG \
            $OF3_EXTRA_ARGS
    }
    if [ "$OF3_BUNDLE" = "1" ]; then
        run_one "$OF3_JSON_DIR/${OF3_DATASET}_openfold3_queries.json"
    else
        shopt -s nullglob
        for json_path in "$OF3_JSON_DIR"/*.json; do
            base="$(basename "$json_path")"
            case "$base" in
                *_openfold3_queries.json) continue ;;
            esac
            run_one "$json_path"
        done
        shopt -u nullglob
    fi
elif [ ! -d "$OF3_OUTPUT_ROOT" ] || ! find "$OF3_OUTPUT_ROOT" \( -name '*_model.cif' -o -name '*_model.pdb' \) 2>/dev/null | grep -q .; then
    echo "[error] no predictions under $OF3_OUTPUT_ROOT"
    echo "        Set RUN_OPENFOLD3=1 or point OF3_OUTPUT_ROOT to existing outputs."
    exit 1
fi

STEM="of3_${OF3_DATASET}_dockq_success"
EVAL_ARGS=()
if [ -n "${OF3_TARGETS:-}" ]; then
    EVAL_ARGS+=(--targets "$OF3_TARGETS")
fi
if [ -n "${OF3_LIMIT:-}" ]; then
    EVAL_ARGS+=(--limit "$OF3_LIMIT")
fi

python examples/eval_af3_dockq_success.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$OF3_DATASET" \
    --pred_root "$OF3_OUTPUT_ROOT" \
    --out_detail "${OUT_DIR}/${STEM}.detail.csv" \
    --out_targets "${OUT_DIR}/${STEM}.targets.csv" \
    --out_summary "${OUT_DIR}/${STEM}.summary.csv" \
    "${EVAL_ARGS[@]}"

echo ""
echo "Done. Key file: ${OUT_DIR}/${STEM}.summary.csv"
echo "Look at success@1 / success@5 / success_oracle (DockQ>=0.23)."
