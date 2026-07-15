#!/usr/bin/env bash
# Restore the DB5 evaluation repo, portable data, and optional packed env assets.

set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-nan20251/tradock-db5-evaluation}"
RELEASE_TAG="${RELEASE_TAG:-db5-eval-20260715}"
REPO_URL="${REPO_URL:-https://github.com/${GITHUB_REPO}.git}"
PROJECT_ROOT="${TRADOCK_DIR:-/root/TraDock}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/root/autodl-tmp/tradock_full_eval_release}"
INSTALL_ROOT="${INSTALL_ROOT:-/}"
TRADOCK_CONDA_PREFIX="${TRADOCK_CONDA_PREFIX:-/root/autodl-tmp/conda_envs/tradock}"
COLABFOLD_CONDA_PREFIX="${COLABFOLD_CONDA_PREFIX:-/root/autodl-tmp/conda_envs/colabfold}"
COLABFOLD_DATA_ROOT="${COLABFOLD_DATA_ROOT:-/root/autodl-tmp}"
METHODS="${METHODS:-hdock alphafold lightdock}"

download_release() {
    mkdir -p "$DOWNLOAD_DIR"
    if command -v gh >/dev/null 2>&1; then
        gh release download "$RELEASE_TAG" \
            --repo "$GITHUB_REPO" \
            --dir "$DOWNLOAD_DIR" \
            --clobber
    else
        echo "[error] GitHub CLI 'gh' is required for private release downloads."
        echo "Install gh or download release assets manually into: $DOWNLOAD_DIR"
        exit 1
    fi
}

restore_repo() {
    if [ -d "$PROJECT_ROOT/.git" ]; then
        git -C "$PROJECT_ROOT" fetch --all --tags
        git -C "$PROJECT_ROOT" checkout main
        git -C "$PROJECT_ROOT" pull --ff-only
    else
        mkdir -p "$(dirname "$PROJECT_ROOT")"
        git clone "$REPO_URL" "$PROJECT_ROOT"
    fi
}

verify_checksums() {
    local manifest
    manifest="$(find "$DOWNLOAD_DIR" -maxdepth 1 -name '*.sha256' -print -quit)"
    if [ -n "$manifest" ]; then
        (cd "$DOWNLOAD_DIR" && shasum -a 256 -c "$(basename "$manifest")")
    else
        echo "[warn] no sha256 manifest found in $DOWNLOAD_DIR"
    fi
}

restore_portable_data() {
    local outer
    local inner
    outer="$(find "$DOWNLOAD_DIR" -maxdepth 1 -name 'tradock_db5_eval_pack_*_portable.tar.gz' -print -quit)"
    if [ -z "$outer" ]; then
        echo "[error] missing tradock_db5_eval_pack_*_portable.tar.gz in $DOWNLOAD_DIR"
        exit 1
    fi

    mkdir -p "$DOWNLOAD_DIR/extracted"
    tar -xzf "$outer" -C "$DOWNLOAD_DIR/extracted"
    inner="$(find "$DOWNLOAD_DIR/extracted" -path '*/portable_data/tradock_db5_portable_data_latest.tar.gz' -print -quit)"
    if [ -z "$inner" ]; then
        echo "[error] portable data archive not found inside $outer"
        exit 1
    fi
    tar -xzf "$inner" -C "$INSTALL_ROOT"
}

restore_conda_pack() {
    local archive="$1"
    local prefix="$2"
    local label="$3"
    if [ ! -f "$archive" ]; then
        echo "[skip] $label conda-pack archive not found: $archive"
        return
    fi
    mkdir -p "$prefix"
    tar -xzf "$archive" -C "$prefix"
    if [ -x "$prefix/bin/conda-unpack" ]; then
        "$prefix/bin/conda-unpack"
    fi
    echo "[ok] restored $label env to $prefix"
}

restore_colabfold_params() {
    local joined="$DOWNLOAD_DIR/colabfold_params_v3.tar.gz"
    if compgen -G "$DOWNLOAD_DIR/colabfold_params_v3.tar.gz.part-*" >/dev/null; then
        cat "$DOWNLOAD_DIR"/colabfold_params_v3.tar.gz.part-* > "$joined"
    fi
    if [ -f "$joined" ]; then
        tar -xzf "$joined" -C "$COLABFOLD_DATA_ROOT"
        echo "[ok] restored ColabFold params under $COLABFOLD_DATA_ROOT"
    else
        echo "[skip] ColabFold params archive not found in $DOWNLOAD_DIR"
    fi
}

main() {
    echo "=== Download release assets ==="
    download_release

    echo "=== Verify checksums ==="
    verify_checksums

    echo "=== Restore repo ==="
    restore_repo

    echo "=== Restore DB5/PPCBench/HDOCK/checkpoint portable data ==="
    restore_portable_data

    echo "=== Restore optional packed environments ==="
    restore_conda_pack "$DOWNLOAD_DIR/tradock_conda_env.tar.gz" "$TRADOCK_CONDA_PREFIX" "TraDock"
    restore_conda_pack "$DOWNLOAD_DIR/colabfold_conda_env.tar.gz" "$COLABFOLD_CONDA_PREFIX" "ColabFold"
    restore_colabfold_params

    echo "=== Verify install ==="
    METHODS="$METHODS" TRADOCK_DIR="$PROJECT_ROOT" \
        PYTHON_BIN="${PYTHON_BIN:-$TRADOCK_CONDA_PREFIX/bin/python}" \
        COLABFOLD_BIN="${COLABFOLD_BIN:-$COLABFOLD_CONDA_PREFIX/bin/colabfold_batch}" \
        COLABFOLD_DATA="${COLABFOLD_DATA:-$COLABFOLD_DATA_ROOT/colabfold_params_v3}" \
        bash "$PROJECT_ROOT/scripts/verify_full_eval.sh"

    echo "=== Ready ==="
    echo "Run HDOCK only:"
    echo "  cd $PROJECT_ROOT && METHODS=hdock bash scripts/run_db5_three_method_eval.sh"
    echo "Run all methods after AlphaFold/LightDock verification passes:"
    echo "  cd $PROJECT_ROOT && bash scripts/run_db5_three_method_eval.sh"
}

main "$@"
