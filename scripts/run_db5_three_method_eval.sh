#!/usr/bin/env bash
# Paper-level DB5 evaluation for HDOCK, AlphaFold/ColabFold, and LightDock poses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRADOCK_ENV_FILE="${TRADOCK_ENV_FILE:-$PROJECT_ROOT/environment}"
if [ -f "$TRADOCK_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TRADOCK_ENV_FILE"
    PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"
fi
TRADOCK_LOCAL_ENV_FILE="${TRADOCK_LOCAL_ENV_FILE:-$PROJECT_ROOT/environment.local}"
if [ -f "$TRADOCK_LOCAL_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$TRADOCK_LOCAL_ENV_FILE"
    PROJECT_ROOT="${TRADOCK_DIR:-$PROJECT_ROOT}"
fi
_DATA_ROOT="${TRADOCK_DATA_ROOT:-$HOME/tradock_data}"
PPC_ROOT="${PPC_ROOT:-${PAPER_ROOT:-$_DATA_ROOT/PPCBench}}"
RUN_ROOT="${DB5_EVAL_RUN_ROOT:-$_DATA_ROOT/db5_three_method_eval}"
RESULTS_ROOT="${DB5_EVAL_RESULTS_ROOT:-${RUN_ROOT}/results}"
PAPER_EVAL_ROOT="${DB5_EVAL_PAPER_ROOT:-${RUN_ROOT}/PPCBench_eval}"
OUT_DIR="${DB5_EVAL_OUT_DIR:-${PROJECT_ROOT}/results}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/Trained_models/pretrain_with_sasa/TransformerDock_best.chk}"
METHODS="${METHODS:-hdock alphafold lightdock}"
MIN_TARGETS="${MIN_TARGETS:-218}"

HDOCK_DATASET="${HDOCK_DATASET:-DB5-u}"
HDOCK_NMAX="${HDOCK_NMAX:-100}"
HDOCK_RUN_ROOT="${HDOCK_RUN_ROOT:-${RUN_ROOT}/hdock}"
HDOCK_WORK_ROOT="${HDOCK_WORK_ROOT:-${HDOCK_RUN_ROOT}/work}"
HDOCK_OUT_ROOT="${HDOCK_OUT_ROOT:-${RESULTS_ROOT}/${HDOCK_DATASET}}"
if [ -z "${HDOCK_BIN:-}" ]; then
    if [ -x "$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/hdock" ]; then
        HDOCK_BIN="$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/hdock"
    elif [ -x /root/autodl-tmp/tools/hdocklite_full/hdock ]; then
        HDOCK_BIN=/root/autodl-tmp/tools/hdocklite_full/hdock
    else
        HDOCK_BIN=hdock
    fi
fi
if [ -z "${CREATEPL_BIN:-}" ]; then
    if [ -x "$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/createpl" ]; then
        CREATEPL_BIN="$_DATA_ROOT/autodl-tmp/tools/hdocklite_full/createpl"
    elif [ -x /root/autodl-tmp/tools/hdocklite_full/createpl ]; then
        CREATEPL_BIN=/root/autodl-tmp/tools/hdocklite_full/createpl
    else
        CREATEPL_BIN=createpl_linux
    fi
fi

AF_DATASET="${AF_DATASET:-DB5}"
AF_NMAX="${AF_NMAX:-5}"
AF_NUM_SEEDS="${AF_NUM_SEEDS:-3}"
AF_MODELS_PER_SEED="${AF_MODELS_PER_SEED:-5}"
AF_RANDOM_SEED_FLAG="${AF_RANDOM_SEED_FLAG:---random-seed}"
AF_FASTA_DIR="${AF_FASTA_DIR:-${RUN_ROOT}/colabfold_fastas/${AF_DATASET}}"
AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-${RUN_ROOT}/colabfold_outputs/${AF_DATASET}}"
if [ -z "${COLABFOLD_BIN:-}" ]; then
    if [ -x "$HOME/localcolabfold/.pixi/envs/default/bin/colabfold_batch" ]; then
        COLABFOLD_BIN="$HOME/localcolabfold/.pixi/envs/default/bin/colabfold_batch"
    else
        COLABFOLD_BIN=colabfold_batch
    fi
fi
COLABFOLD_DATA="${COLABFOLD_DATA:-$_DATA_ROOT/colabfold_params_v3}"
AF_NUM_RECYCLE="${AF_NUM_RECYCLE:-3}"
RUN_COLABFOLD="${RUN_COLABFOLD:-0}"
unset _DATA_ROOT

LIGHTDOCK_DATASET="${LIGHTDOCK_DATASET:-DB5-u}"
LIGHTDOCK_NMAX="${LIGHTDOCK_NMAX:-100}"
LIGHTDOCK_INPUT_DIR="${LIGHTDOCK_INPUT_DIR:-${RUN_ROOT}/lightdock_inputs/${LIGHTDOCK_DATASET}}"
LIGHTDOCK_OUTPUT_ROOT="${LIGHTDOCK_OUTPUT_ROOT:-${RUN_ROOT}/lightdock_outputs/${LIGHTDOCK_DATASET}}"
LIGHTDOCK_SWARMS="${LIGHTDOCK_SWARMS:-40}"
LIGHTDOCK_GLOWWORMS="${LIGHTDOCK_GLOWWORMS:-200}"
LIGHTDOCK_STEPS="${LIGHTDOCK_STEPS:-100}"
LIGHTDOCK_WORKERS="${LIGHTDOCK_WORKERS:-4}"
RUN_LIGHTDOCK="${RUN_LIGHTDOCK:-1}"
unset SCRIPT_DIR

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
    EVAL_ARGS=()
    if [ -n "$eval_limit" ]; then
        EVAL_ARGS+=(--limit "$eval_limit")
    fi
    python -u examples/eval_db5_paper_tradock.py \
        --paper_root "$PAPER_EVAL_ROOT" \
        --dataset "$dataset" \
        --pose_models "$pose_model_csv" \
        --checkpoint "$CHECKPOINT" \
        --out "$detail_out" \
        --score_type "${SCORE_TYPE:-mdn}" \
        --min_targets "$min_targets" \
        "${EVAL_ARGS[@]}"
}

summarize_detail() {
    local detail_out="$1"
    local method_name="$2"
    local stem="$3"
    python examples/summarize_paper_pose_scores.py \
        --detail "$detail_out" \
        --target_out "${OUT_DIR}/${stem}.targets.csv" \
        --summary_out "${OUT_DIR}/${stem}.summary.csv" \
        --method_name "$method_name"
}

echo "=== DB5 three-method paper-level evaluation ==="
echo "PPC_ROOT=$PPC_ROOT"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "PAPER_EVAL_ROOT=$PAPER_EVAL_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "METHODS=$METHODS"
echo ""

if [ ! -d "$PPC_ROOT/dataset" ] || [ ! -d "$PPC_ROOT/evaluate" ]; then
    echo "[error] missing PPCBench dataset/evaluate under $PPC_ROOT"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "[error] missing TraDock checkpoint: $CHECKPOINT"
    exit 1
fi

if has_method hdock; then
    echo "=== HDOCKlite ${HDOCK_DATASET} Top${HDOCK_NMAX} ==="
    setup_eval_root "$HDOCK_DATASET"
    mkdir -p "$HDOCK_OUT_ROOT" "$HDOCK_WORK_ROOT"
    HDOCK_ARGS=()
    if [ -n "${HDOCK_TARGETS:-}" ]; then
        HDOCK_ARGS+=(--targets "$HDOCK_TARGETS")
    fi
    if [ -n "${HDOCK_LIMIT:-}" ]; then
        HDOCK_ARGS+=(--limit "$HDOCK_LIMIT")
    fi
    if [ -n "${HDOCK_OVERWRITE:-}" ]; then
        HDOCK_ARGS+=(--overwrite)
    fi
    python -u examples/generate_db5_hdocklite_candidates.py \
        --paper_root "$PPC_ROOT" \
        --dataset "$HDOCK_DATASET" \
        --out_root "$HDOCK_OUT_ROOT" \
        --work_root "$HDOCK_WORK_ROOT" \
        --hdock_bin "$HDOCK_BIN" \
        --createpl_bin "$CREATEPL_BIN" \
        --nmax "$HDOCK_NMAX" \
        "${HDOCK_ARGS[@]}"

    HDOCK_POSES="$(pose_models hdock "$HDOCK_NMAX")"
    HDOCK_DETAIL="${OUT_DIR}/tradock_${HDOCK_DATASET}_hdock_top${HDOCK_NMAX}.csv"
    eval_with_tradock "$HDOCK_DATASET" "$HDOCK_POSES" "$HDOCK_DETAIL" "$MIN_TARGETS" "${HDOCK_EVAL_LIMIT:-${HDOCK_LIMIT:-}}"
    summarize_detail "$HDOCK_DETAIL" hdock "hdock_${HDOCK_DATASET}_top${HDOCK_NMAX}_compare"
fi

if has_method alphafold; then
    echo "=== AlphaFold/ColabFold ${AF_DATASET} ${AF_NUM_SEEDS} seeds x ${AF_MODELS_PER_SEED} models -> Top${AF_NMAX} ==="
    setup_eval_root "$AF_DATASET"
    if [ "$RUN_COLABFOLD" = "1" ]; then
        if ! command -v "$COLABFOLD_BIN" >/dev/null 2>&1 && [ ! -x "$COLABFOLD_BIN" ]; then
            echo "[error] RUN_COLABFOLD=1 but ColabFold binary missing: $COLABFOLD_BIN"
            echo "        Install LocalColabFold, then: bash scripts/setup_colabfold_paths.sh"
            exit 1
        fi
        if [ ! -d "${COLABFOLD_DATA:-}" ]; then
            echo "[error] RUN_COLABFOLD=1 but COLABFOLD_DATA missing: ${COLABFOLD_DATA:-}"
            echo "        Place AF2 params there or run: bash scripts/setup_colabfold_paths.sh"
            exit 1
        fi
    else
        if [ ! -d "$AF_OUTPUT_ROOT" ] || ! find "$AF_OUTPUT_ROOT" -name '*.pdb' 2>/dev/null | grep -q .; then
            echo "[error] RUN_COLABFOLD=0 but no ColabFold PDBs under $AF_OUTPUT_ROOT"
            echo "        Set RUN_COLABFOLD=1 after installing ColabFold, or populate AF_OUTPUT_ROOT."
            exit 1
        fi
        echo "RUN_COLABFOLD=0, using existing ColabFold outputs in $AF_OUTPUT_ROOT"
    fi
    AF_FASTA_ARGS=()
    if [ -n "${AF_TARGETS:-}" ]; then
        AF_FASTA_ARGS+=(--targets "$AF_TARGETS")
    fi
    if [ -n "${AF_LIMIT:-}" ]; then
        AF_FASTA_ARGS+=(--limit "$AF_LIMIT")
    fi
    python examples/prepare_db5_colabfold_fastas.py \
        --ppc_root "$PPC_ROOT" \
        --dataset "$AF_DATASET" \
        --out_dir "$AF_FASTA_DIR" \
        "${AF_FASTA_ARGS[@]}"

    if [ "$RUN_COLABFOLD" = "1" ]; then
        mkdir -p "$AF_OUTPUT_ROOT"
        COLABFOLD_ARGS=(
            --model-type alphafold2_multimer_v3
            --num-models "$AF_MODELS_PER_SEED"
            --num-recycle "$AF_NUM_RECYCLE"
            --rank multimer
        )
        if [ -n "${COLABFOLD_DATA:-}" ]; then
            COLABFOLD_ARGS+=(--data "$COLABFOLD_DATA")
        fi
        if [ -z "${AF_SEEDS:-}" ]; then
            AF_SEEDS="$(seq 1 "$AF_NUM_SEEDS")"
        fi
        for fasta in "$AF_FASTA_DIR"/*.fasta; do
            target="$(basename "$fasta" .fasta)"
            for seed in $AF_SEEDS; do
                seed_out="$AF_OUTPUT_ROOT/$target/seed_$seed"
                mkdir -p "$seed_out"
                SEED_ARGS=()
                if [ -n "$AF_RANDOM_SEED_FLAG" ]; then
                    SEED_ARGS+=("$AF_RANDOM_SEED_FLAG" "$seed")
                fi
                "$COLABFOLD_BIN" "${COLABFOLD_ARGS[@]}" "${SEED_ARGS[@]}" "$fasta" "$seed_out"
            done
        done
    fi

    AF_SCORES="${OUT_DIR}/af2m_${AF_DATASET}_top${AF_NMAX}_scores.csv"
    AF_CONVERT_ARGS=()
    if [ -n "${AF_TARGETS:-}" ]; then
        AF_CONVERT_ARGS+=(--targets "$AF_TARGETS")
    fi
    if [ -n "${AF_LIMIT:-}" ]; then
        AF_CONVERT_ARGS+=(--limit "$AF_LIMIT")
    fi
    python examples/convert_colabfold_db5.py \
        --ppc_root "$PPC_ROOT" \
        --dataset "$AF_DATASET" \
        --dataset_json "$PPC_ROOT/dataset/$AF_DATASET/$AF_DATASET.json" \
        --colabfold_root "$AF_OUTPUT_ROOT" \
        --results_root "$RESULTS_ROOT" \
        --scores_csv "$AF_SCORES" \
        --max_models "$AF_NMAX" \
        "${AF_CONVERT_ARGS[@]}"

    AF_POSES="$(pose_models af2m "$AF_NMAX")"
    AF_DETAIL="${OUT_DIR}/tradock_${AF_DATASET}_af2m_top${AF_NMAX}.csv"
    eval_with_tradock "$AF_DATASET" "$AF_POSES" "$AF_DETAIL" "$MIN_TARGETS" "${AF_EVAL_LIMIT:-${AF_LIMIT:-}}"
    python examples/compare_af_tradock.py \
        --af_scores "$AF_SCORES" \
        --tradock_detail "$AF_DETAIL" \
        --merged_out "${OUT_DIR}/af_vs_tradock_${AF_DATASET}_top${AF_NMAX}.merged.csv" \
        --summary_out "${OUT_DIR}/af_vs_tradock_${AF_DATASET}_top${AF_NMAX}.summary.csv" \
        --aggregate_out "${OUT_DIR}/af_vs_tradock_${AF_DATASET}_top${AF_NMAX}.aggregate.csv"
fi

if has_method lightdock; then
    echo "=== LightDock ${LIGHTDOCK_DATASET} Top${LIGHTDOCK_NMAX} ==="
    setup_eval_root "$LIGHTDOCK_DATASET"
    if [ "$RUN_LIGHTDOCK" = "1" ]; then
        for cmd in lightdock3_setup.py lightdock3.py lgd_generate_conformations.py; do
            if ! command -v "$cmd" >/dev/null 2>&1; then
                echo "[error] missing LightDock command: $cmd"
                echo "        Install with: bash scripts/install_lightdock.sh"
                exit 1
            fi
        done
    else
        if [ ! -d "$LIGHTDOCK_OUTPUT_ROOT" ]; then
            echo "[error] RUN_LIGHTDOCK=0 but missing outputs: $LIGHTDOCK_OUTPUT_ROOT"
            exit 1
        fi
        echo "RUN_LIGHTDOCK=0, using existing LightDock outputs in $LIGHTDOCK_OUTPUT_ROOT"
    fi
    LIGHTDOCK_PREP_ARGS=()
    if [ -n "${LIGHTDOCK_TARGETS:-}" ]; then
        LIGHTDOCK_PREP_ARGS+=(--targets "$LIGHTDOCK_TARGETS")
    fi
    if [ -n "${LIGHTDOCK_LIMIT:-}" ]; then
        LIGHTDOCK_PREP_ARGS+=(--limit "$LIGHTDOCK_LIMIT")
    fi
    python examples/prepare_db5_lightdock_inputs.py \
        --ppc_root "$PPC_ROOT" \
        --dataset "$LIGHTDOCK_DATASET" \
        --out_dir "$LIGHTDOCK_INPUT_DIR" \
        "${LIGHTDOCK_PREP_ARGS[@]}"

    if [ "$RUN_LIGHTDOCK" = "1" ]; then
        LIGHTDOCK_RUN_ARGS=()
        if [ -n "${LIGHTDOCK_LIMIT:-}" ]; then
            LIGHTDOCK_RUN_ARGS+=(--limit "$LIGHTDOCK_LIMIT")
        fi
        python -u examples/run_lightdock.py \
            --pdb_dir "$LIGHTDOCK_INPUT_DIR" \
            --out_root "$LIGHTDOCK_OUTPUT_ROOT" \
            --swarms "$LIGHTDOCK_SWARMS" \
            --glowworms "$LIGHTDOCK_GLOWWORMS" \
            --steps "$LIGHTDOCK_STEPS" \
            --workers "$LIGHTDOCK_WORKERS" \
            "${LIGHTDOCK_RUN_ARGS[@]}"
    else
        echo "RUN_LIGHTDOCK=0, using existing LightDock outputs in $LIGHTDOCK_OUTPUT_ROOT"
    fi

    LIGHTDOCK_EXPORT_ARGS=()
    if [ -n "${LIGHTDOCK_TARGETS:-}" ]; then
        LIGHTDOCK_EXPORT_ARGS+=(--targets "$LIGHTDOCK_TARGETS")
    fi
    if [ -n "${LIGHTDOCK_LIMIT:-}" ]; then
        LIGHTDOCK_EXPORT_ARGS+=(--limit "$LIGHTDOCK_LIMIT")
    fi
    python examples/export_lightdock_to_paper_candidates.py \
        --ld_root "$LIGHTDOCK_OUTPUT_ROOT" \
        --out_root "$RESULTS_ROOT/$LIGHTDOCK_DATASET" \
        --dataset "$LIGHTDOCK_DATASET" \
        --prefix lightdock \
        --nmax "$LIGHTDOCK_NMAX" \
        "${LIGHTDOCK_EXPORT_ARGS[@]}"

    LIGHTDOCK_POSES="$(pose_models lightdock "$LIGHTDOCK_NMAX")"
    LIGHTDOCK_DETAIL="${OUT_DIR}/tradock_${LIGHTDOCK_DATASET}_lightdock_top${LIGHTDOCK_NMAX}.csv"
    eval_with_tradock "$LIGHTDOCK_DATASET" "$LIGHTDOCK_POSES" "$LIGHTDOCK_DETAIL" "$MIN_TARGETS" "${LIGHTDOCK_EVAL_LIMIT:-${LIGHTDOCK_LIMIT:-}}"
    summarize_detail "$LIGHTDOCK_DETAIL" lightdock "lightdock_${LIGHTDOCK_DATASET}_top${LIGHTDOCK_NMAX}_compare"
fi

echo "=== Done ==="
ls -1 "$OUT_DIR"/*top*.csv "$OUT_DIR"/*compare*.csv 2>/dev/null || true
