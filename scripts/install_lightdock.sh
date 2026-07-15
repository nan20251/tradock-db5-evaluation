#!/usr/bin/env bash
# Install LightDock CLIs into the active Python/conda environment.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== Install LightDock ==="
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -U lightdock

missing=0
for cmd in lightdock3_setup.py lightdock3.py lgd_generate_conformations.py; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "[ok] $cmd -> $(command -v "$cmd")"
    else
        # pip scripts may live next to the python binary
        bin_dir="$(dirname "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')")"
        if [ -x "$bin_dir/$cmd" ]; then
            echo "[ok] $cmd -> $bin_dir/$cmd"
            echo "      add to PATH: export PATH=\"$bin_dir:\$PATH\""
        else
            echo "[missing] $cmd"
            missing=1
        fi
    fi
done

if [ "$missing" -ne 0 ]; then
    echo "[error] LightDock install incomplete"
    exit 1
fi

echo "[ok] LightDock ready"
echo "Verify with:"
echo "  METHODS=lightdock bash scripts/verify_full_eval.sh"
