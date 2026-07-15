#!/usr/bin/env bash
# Shared path helpers for TraDock DB5 evaluation scripts.
# Source from repo root environment or other scripts:
#   source "$PROJECT_ROOT/scripts/tradock_path_lib.sh"

tradock_home() {
    printf '%s\n' "${HOME:-/root}"
}

tradock_data_root() {
    printf '%s\n' "${TRADOCK_DATA_ROOT:-$(tradock_home)/tradock_data}"
}

tradock_first_existing() {
    local kind="$1"
    shift
    local path
    for path in "$@"; do
        [ -n "$path" ] || continue
        case "$kind" in
            file)
                [ -f "$path" ] || continue
                ;;
            dir)
                [ -d "$path" ] || continue
                ;;
            exe)
                [ -x "$path" ] || continue
                ;;
            *)
                return 1
                ;;
        esac
        printf '%s\n' "$path"
        return 0
    done
    return 1
}

tradock_default_ppc_root() {
    tradock_first_existing dir \
        "$(tradock_data_root)/PPCBench" \
        /root/PPCBench \
        || printf '%s\n' "$(tradock_data_root)/PPCBench"
}

tradock_default_checkpoint() {
    local project_root="${1:-}"
    local found
    local candidates=(
        "$(tradock_data_root)/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk"
    )
    if [ -n "$project_root" ]; then
        candidates+=("$project_root/Trained_models/pretrain_with_sasa/TransformerDock_best.chk")
    fi
    if found="$(tradock_first_existing file "${candidates[@]}")"; then
        printf '%s\n' "$found"
        return 0
    fi
    printf '%s\n' "${project_root}/Trained_models/pretrain_with_sasa/TransformerDock_best.chk"
}

tradock_default_hdock_bin() {
    tradock_first_existing exe \
        "$(tradock_data_root)/autodl-tmp/tools/hdocklite_full/hdock" \
        /root/autodl-tmp/tools/hdocklite_full/hdock \
        || printf 'hdock\n'
}

tradock_default_createpl_bin() {
    tradock_first_existing exe \
        "$(tradock_data_root)/autodl-tmp/tools/hdocklite_full/createpl" \
        /root/autodl-tmp/tools/hdocklite_full/createpl \
        || printf 'createpl_linux\n'
}

tradock_default_colabfold_bin() {
    local home
    home="$(tradock_home)"
    tradock_first_existing exe \
        "$home/localcolabfold/.pixi/envs/default/bin/colabfold_batch" \
        "$home/localcolabfold/colabfold-conda/bin/colabfold_batch" \
        "$home/conda_envs/colabfold/bin/colabfold_batch" \
        "$(tradock_data_root)/conda_envs/colabfold/bin/colabfold_batch" \
        || printf 'colabfold_batch\n'
}

tradock_default_colabfold_data() {
    local home
    home="$(tradock_home)"
    tradock_first_existing dir \
        "$(tradock_data_root)/colabfold_params_v3" \
        "$home/autodl-tmp/colabfold_params_v3" \
        "$home/.cache/colabfold" \
        /root/autodl-tmp/colabfold_params_v3 \
        || printf '%s\n' "$(tradock_data_root)/colabfold_params_v3"
}

tradock_source_env_files() {
    local project_root="$1"
    local env_file="${TRADOCK_ENV_FILE:-$project_root/environment}"
    local local_file="${TRADOCK_LOCAL_ENV_FILE:-$project_root/environment.local}"
    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi
    if [ -f "$local_file" ]; then
        # shellcheck disable=SC1090
        source "$local_file"
    fi
}
