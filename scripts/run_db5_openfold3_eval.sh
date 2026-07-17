#!/usr/bin/env bash
# DB5 OpenFold3 evaluation: prepare query JSON -> (optional) run_openfold -> convert -> TraDock.
#
# Install (once):
#   bash scripts/install_openfold3.sh
#
# Smoke test (1 target, MSA via ColabFold server):
#   RUN_OPENFOLD3=1 OF3_LIMIT=1 bash scripts/run_db5_openfold3_eval.sh
#
# Convert + score existing outputs only:
#   RUN_OPENFOLD3=0 bash scripts/run_db5_openfold3_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=scripts/tradock_path_lib.sh
source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"
tradock_source_env_files "$PROJECT_ROOT"
PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"

PPC_ROOT="${PPC_ROOT:?set PPC_ROOT in environment.local}"
RUN_ROOT="${DB5_EVAL_RUN_ROOT:?}"
RESULTS_ROOT="${DB5_EVAL_RESULTS_ROOT:-$RUN_ROOT/results}"
PAPER_EVAL_ROOT="${DB5_EVAL_PAPER_ROOT:-$RUN_ROOT/PPCBench_eval}"
OUT_DIR="${DB5_EVAL_OUT_DIR:-$PROJECT_ROOT/results}"
CHECKPOINT="${CHECKPOINT:?}"
MIN_TARGETS="${MIN_TARGETS:-218}"

OF3_DATASET="${OF3_DATASET:-DB5}"
OF3_NMAX="${OF3_NMAX:-5}"
OF3_NUM_SEEDS="${OF3_NUM_SEEDS:-3}"
OF3_NUM_SAMPLES="${OF3_NUM_SAMPLES:-5}"
OF3_JSON_DIR="${OF3_JSON_DIR:-$RUN_ROOT/openfold3_jsons/$OF3_DATASET}"
OF3_OUTPUT_ROOT="${OF3_OUTPUT_ROOT:-$RUN_ROOT/openfold3_outputs/$OF3_DATASET}"
OF3_BIN="${OF3_BIN:-run_openfold}"
OF3_USE_MSA_SERVER="${OF3_USE_MSA_SERVER:-1}"
OF3_EXTRA_ARGS="${OF3_EXTRA_ARGS:-}"
OF3_BUNDLE="${OF3_BUNDLE:-0}"
RUN_OPENFOLD3="${RUN_OPENFOLD3:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_ROOT" "$PAPER_EVAL_ROOT/dataset" "$PAPER_EVAL_ROOT/results" "$OUT_DIR"

pose_models() {
    local prefix="$1"
    local n="$2"
    local sep=""
    local out=""
    local i
    for i in $(seq 1 "$n"); do
        out="${out}${sep}${prefix}_${i}"
        sep=","
    done
    printf "%s\n" "$out"
}

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

eval_with_tradock() {
    local dataset="$1"
    local pose_model_csv="$2"
    local detail_out="$3"
    local min_targets="$4"
    local eval_limit="${5:-}"
    local eval_args=()
    if [ -n "$eval_limit" ]; then
        eval_args+=(--limit "$eval_limit")
    fi
    python -u examples/eval_db5_paper_tradock.py \
        --paper_root "$PAPER_EVAL_ROOT" \
        --dataset "$dataset" \
        --pose_models "$pose_model_csv" \
        --checkpoint "$CHECKPOINT" \
        --out "$detail_out" \
        --score_type "${SCORE_TYPE:-mdn}" \
        --min_targets "$min_targets" \
        "${eval_args[@]}"
}

echo "=== OpenFold3 ${OF3_DATASET} Top${OF3_NMAX} ==="
echo "PPC_ROOT=$PPC_ROOT"
echo "OF3_JSON_DIR=$OF3_JSON_DIR"
echo "OF3_OUTPUT_ROOT=$OF3_OUTPUT_ROOT"
echo "RUN_OPENFOLD3=$RUN_OPENFOLD3"
echo "OF3_USE_MSA_SERVER=$OF3_USE_MSA_SERVER"
echo ""

setup_eval_root "$OF3_DATASET"

if [ "$RUN_OPENFOLD3" = "1" ]; then
    if ! command -v "$OF3_BIN" >/dev/null 2>&1 && [ ! -x "$OF3_BIN" ]; then
        echo "[error] RUN_OPENFOLD3=1 but missing: $OF3_BIN"
        echo "        Install with: bash scripts/install_openfold3.sh"
        exit 1
    fi
elif [ ! -d "$OF3_OUTPUT_ROOT" ] || ! find "$OF3_OUTPUT_ROOT" \( -name '*_model.cif' -o -name '*_model.pdb' \) 2>/dev/null | grep -q .; then
    echo "[error] RUN_OPENFOLD3=0 but no OpenFold3 models under $OF3_OUTPUT_ROOT"
    echo "        Set RUN_OPENFOLD3=1 after installing OpenFold3, or populate OF3_OUTPUT_ROOT."
    exit 1
else
    echo "RUN_OPENFOLD3=0, using existing OpenFold3 outputs in $OF3_OUTPUT_ROOT"
fi

OF3_JSON_ARGS=(--format openfold3)
if [ -n "${OF3_TARGETS:-}" ]; then
    OF3_JSON_ARGS+=(--targets "$OF3_TARGETS")
fi
if [ -n "${OF3_LIMIT:-}" ]; then
    OF3_JSON_ARGS+=(--limit "$OF3_LIMIT")
fi
if [ "$OF3_BUNDLE" = "1" ]; then
    OF3_JSON_ARGS+=(--bundle)
fi

python examples/prepare_db5_alphafold3_jsons.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$OF3_DATASET" \
    --out_dir "$OF3_JSON_DIR" \
    "${OF3_JSON_ARGS[@]}"

if [ "$RUN_OPENFOLD3" = "1" ]; then
    mkdir -p "$OF3_OUTPUT_ROOT"
    MSA_FLAG="--use-msa-server"
    if [ "$OF3_USE_MSA_SERVER" = "0" ]; then
        MSA_FLAG="--use-msa-server=False"
    fi

    run_one_query() {
        local json_path="$1"
        echo "=== OpenFold3: $(basename "$json_path") ==="
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
        bundle="$OF3_JSON_DIR/${OF3_DATASET}_openfold3_queries.json"
        if [ ! -f "$bundle" ]; then
            echo "[error] missing bundle JSON: $bundle"
            exit 1
        fi
        run_one_query "$bundle"
    else
        shopt -s nullglob
        for json_path in "$OF3_JSON_DIR"/*.json; do
            base="$(basename "$json_path")"
            case "$base" in
                *_openfold3_queries.json|*_openfold3_jsons.csv) continue ;;
            esac
            run_one_query "$json_path"
        done
        shopt -u nullglob
    fi
fi

OF3_SCORES="${OUT_DIR}/of3_${OF3_DATASET}_top${OF3_NMAX}_scores.csv"
OF3_CONVERT_ARGS=(--pose_prefix of3)
if [ -n "${OF3_TARGETS:-}" ]; then
    OF3_CONVERT_ARGS+=(--targets "$OF3_TARGETS")
fi
if [ -n "${OF3_LIMIT:-}" ]; then
    OF3_CONVERT_ARGS+=(--limit "$OF3_LIMIT")
fi

python examples/convert_alphafold3_db5.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$OF3_DATASET" \
    --dataset_json "$PPC_ROOT/dataset/$OF3_DATASET/$OF3_DATASET.json" \
    --af3_root "$OF3_OUTPUT_ROOT" \
    --results_root "$RESULTS_ROOT" \
    --scores_csv "$OF3_SCORES" \
    --max_models "$OF3_NMAX" \
    "${OF3_CONVERT_ARGS[@]}"

OF3_POSES="$(pose_models of3 "$OF3_NMAX")"
OF3_DETAIL="${OUT_DIR}/tradock_${OF3_DATASET}_of3_top${OF3_NMAX}.csv"
eval_with_tradock "$OF3_DATASET" "$OF3_POSES" "$OF3_DETAIL" "$MIN_TARGETS" "${OF3_EVAL_LIMIT:-${OF3_LIMIT:-}}"

python examples/compare_af_tradock.py \
    --af_scores "$OF3_SCORES" \
    --tradock_detail "$OF3_DETAIL" \
    --merged_out "${OUT_DIR}/of3_vs_tradock_${OF3_DATASET}_top${OF3_NMAX}.merged.csv" \
    --summary_out "${OUT_DIR}/of3_vs_tradock_${OF3_DATASET}_top${OF3_NMAX}.summary.csv" \
    --aggregate_out "${OUT_DIR}/of3_vs_tradock_${OF3_DATASET}_top${OF3_NMAX}.aggregate.csv"

echo "=== OpenFold3 evaluation finished ==="
echo "scores:  $OF3_SCORES"
echo "detail:  $OF3_DETAIL"
