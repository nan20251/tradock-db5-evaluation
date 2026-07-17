#!/usr/bin/env bash
# Install OpenFold3 (Apache-2.0) for TraDock DB5 evaluation.
#
# Official docs: https://openfold-3.readthedocs.io/en/latest/Installation.html
#
# After install:
#   RUN_OPENFOLD3=1 OF3_LIMIT=1 bash scripts/run_db5_openfold3_eval.sh

set -euo pipefail

OF3_ENV_NAME="${OF3_ENV_NAME:-openfold3}"
OF3_PYTHON="${OF3_PYTHON:-3.12}"
NONINTERACTIVE="${NONINTERACTIVE:-1}"

echo "=== OpenFold3 install ==="
echo "Recommended: dedicated conda env (do not mix into tradock)."
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found; install Miniconda/Mamba first."
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$OF3_ENV_NAME"; then
    echo "[ok] conda env exists: $OF3_ENV_NAME"
else
    echo "[info] creating conda env: $OF3_ENV_NAME (python=$OF3_PYTHON)"
    conda create -y -n "$OF3_ENV_NAME" "python=$OF3_PYTHON"
fi

conda activate "$OF3_ENV_NAME"
python -m pip install -U pip
python -m pip install openfold3

if [ "$NONINTERACTIVE" = "1" ]; then
    setup_openfold --non-interactive
else
    setup_openfold
fi

echo ""
echo "[ok] OpenFold3 installed in conda env: $OF3_ENV_NAME"
echo "Verify:"
echo "  conda activate $OF3_ENV_NAME"
echo "  which run_openfold"
echo "  run_openfold predict --help | head"
echo ""
echo "Smoke eval (1 DB5 target, ColabFold MSA server):"
echo "  conda activate $OF3_ENV_NAME"
echo "  cd ~/tradock-db5-evaluation"
echo "  RUN_OPENFOLD3=1 OF3_LIMIT=1 bash scripts/run_db5_openfold3_eval.sh"
echo ""
echo "Notes:"
echo "  - MSA uses public ColabFold server by default (needs network)."
echo "  - Docs recommend >=32GB GPU memory; 3080 Ti 12GB may OOM on large targets."
echo "  - Weights (~2GB) go under ~/.openfold3 by default (no DeepMind AF3 form)."
