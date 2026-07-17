#!/usr/bin/env bash
# DB5 AlphaFold 3 / OpenFold3 evaluation helpers.
#
# Prefer OpenFold3 (open weights, ColabFold MSA server):
#   bash scripts/install_openfold3.sh
#   RUN_OPENFOLD3=1 bash scripts/run_db5_openfold3_eval.sh
#
# Official DeepMind AlphaFold 3 (requires weight access + large DBs):
#   RUN_ALPHAFOLD3=1 AF3_CMD='python /path/to/run_alphafold.py' \
#     AF3_MODEL_DIR=... AF3_DB_DIR=... \
#     bash scripts/run_db5_alphafold3_eval.sh

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

AF3_DATASET="${AF3_DATASET:-DB5}"
AF3_NMAX="${AF3_NMAX:-5}"
AF3_MODEL_SEEDS="${AF3_MODEL_SEEDS:-1,2,3}"
AF3_JSON_DIR="${AF3_JSON_DIR:-$RUN_ROOT/alphafold3_jsons/$AF3_DATASET}"
AF3_OUTPUT_ROOT="${AF3_OUTPUT_ROOT:-$RUN_ROOT/alphafold3_outputs/$AF3_DATASET}"
# AF3 launcher: prefer AF3_CMD (e.g. "python /path/to/run_alphafold.py"),
# else AF3_BIN as a single executable.
AF3_CMD="${AF3_CMD:-${AF3_BIN:-}}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-}"
AF3_DB_DIR="${AF3_DB_DIR:-}"
AF3_EXTRA_ARGS="${AF3_EXTRA_ARGS:-}"
RUN_ALPHAFOLD3="${RUN_ALPHAFOLD3:-0}"

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

echo "=== AlphaFold 3 ${AF3_DATASET} Top${AF3_NMAX} ==="
echo "PPC_ROOT=$PPC_ROOT"
echo "AF3_JSON_DIR=$AF3_JSON_DIR"
echo "AF3_OUTPUT_ROOT=$AF3_OUTPUT_ROOT"
echo "RUN_ALPHAFOLD3=$RUN_ALPHAFOLD3"
echo ""

setup_eval_root "$AF3_DATASET"

if [ "$RUN_ALPHAFOLD3" = "1" ]; then
    if [ -z "$AF3_CMD" ]; then
        echo "[error] RUN_ALPHAFOLD3=1 but AF3_CMD/AF3_BIN unset"
        echo "        Example: AF3_CMD='python /path/to/alphafold3/run_alphafold.py'"
        exit 1
    fi
    if [ -z "${AF3_MODEL_DIR:-}" ] || [ ! -d "$AF3_MODEL_DIR" ]; then
        echo "[error] RUN_ALPHAFOLD3=1 but AF3_MODEL_DIR missing: ${AF3_MODEL_DIR:-}"
        exit 1
    fi
elif [ ! -d "$AF3_OUTPUT_ROOT" ] || ! find "$AF3_OUTPUT_ROOT" -name '*_model.cif' 2>/dev/null | grep -q .; then
    echo "[error] RUN_ALPHAFOLD3=0 but no AF3 mmCIF under $AF3_OUTPUT_ROOT"
    echo "        Set RUN_ALPHAFOLD3=1 after installing AF3, or populate AF3_OUTPUT_ROOT."
    exit 1
else
    echo "RUN_ALPHAFOLD3=0, using existing AF3 outputs in $AF3_OUTPUT_ROOT"
fi

AF3_JSON_ARGS=()
if [ -n "${AF3_TARGETS:-}" ]; then
    AF3_JSON_ARGS+=(--targets "$AF3_TARGETS")
fi
if [ -n "${AF3_LIMIT:-}" ]; then
    AF3_JSON_ARGS+=(--limit "$AF3_LIMIT")
fi

python examples/prepare_db5_alphafold3_jsons.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$AF3_DATASET" \
    --out_dir "$AF3_JSON_DIR" \
    --format alphafold3 \
    --model_seeds "$AF3_MODEL_SEEDS" \
    "${AF3_JSON_ARGS[@]}"

if [ "$RUN_ALPHAFOLD3" = "1" ]; then
    mkdir -p "$AF3_OUTPUT_ROOT"
    # Extra flags via AF3_EXTRA_ARGS, e.g. "--num_diffusion_samples 5 --norun_data_pipeline"
    for json_path in "$AF3_JSON_DIR"/*.json; do
        [ -f "$json_path" ] || continue
        target="$(basename "$json_path" .json)"
        echo "=== AF3 run: $target ==="
        # shellcheck disable=SC2086
        $AF3_CMD \
            --json_path="$json_path" \
            --model_dir="$AF3_MODEL_DIR" \
            ${AF3_DB_DIR:+--db_dir="$AF3_DB_DIR"} \
            --output_dir="$AF3_OUTPUT_ROOT" \
            $AF3_EXTRA_ARGS
    done
fi

AF3_SCORES="${OUT_DIR}/af3_${AF3_DATASET}_top${AF3_NMAX}_scores.csv"
AF3_CONVERT_ARGS=(--pose_prefix af3)
if [ -n "${AF3_TARGETS:-}" ]; then
    AF3_CONVERT_ARGS+=(--targets "$AF3_TARGETS")
fi
if [ -n "${AF3_LIMIT:-}" ]; then
    AF3_CONVERT_ARGS+=(--limit "$AF3_LIMIT")
fi

python examples/convert_alphafold3_db5.py \
    --ppc_root "$PPC_ROOT" \
    --dataset "$AF3_DATASET" \
    --dataset_json "$PPC_ROOT/dataset/$AF3_DATASET/$AF3_DATASET.json" \
    --af3_root "$AF3_OUTPUT_ROOT" \
    --results_root "$RESULTS_ROOT" \
    --scores_csv "$AF3_SCORES" \
    --max_models "$AF3_NMAX" \
    "${AF3_CONVERT_ARGS[@]}"

AF3_POSES="$(pose_models af3 "$AF3_NMAX")"
AF3_DETAIL="${OUT_DIR}/tradock_${AF3_DATASET}_af3_top${AF3_NMAX}.csv"
eval_with_tradock "$AF3_DATASET" "$AF3_POSES" "$AF3_DETAIL" "$MIN_TARGETS" "${AF3_EVAL_LIMIT:-${AF3_LIMIT:-}}"

python examples/compare_af_tradock.py \
    --af_scores "$AF3_SCORES" \
    --tradock_detail "$AF3_DETAIL" \
    --merged_out "${OUT_DIR}/af3_vs_tradock_${AF3_DATASET}_top${AF3_NMAX}.merged.csv" \
    --summary_out "${OUT_DIR}/af3_vs_tradock_${AF3_DATASET}_top${AF3_NMAX}.summary.csv" \
    --aggregate_out "${OUT_DIR}/af3_vs_tradock_${AF3_DATASET}_top${AF3_NMAX}.aggregate.csv"

echo "=== AlphaFold 3 evaluation finished ==="
echo "scores:  $AF3_SCORES"
echo "detail:  $AF3_DETAIL"
