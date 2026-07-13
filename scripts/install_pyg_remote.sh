#!/usr/bin/env bash
set -euo pipefail

# Remote PyG installer for detected Torch/CUDA
# Detected: TORCH=2.4.1, CUDA=cu121
PYTHON_BIN=/root/miniconda3/bin/python

TORCH_TAG=2.4.1
CUDA_TAG=cu121
WHEEL_INDEX=https://data.pyg.org/whl/torch-${TORCH_TAG}+${CUDA_TAG}.html

echo "Using Python: $PYTHON_BIN"
echo "Installing PyG native wheels from: $WHEEL_INDEX"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --no-cache-dir torch-scatter -f "$WHEEL_INDEX"
"$PYTHON_BIN" -m pip install --no-cache-dir torch-sparse -f "$WHEEL_INDEX"
"$PYTHON_BIN" -m pip install --no-cache-dir torch-cluster -f "$WHEEL_INDEX"
"$PYTHON_BIN" -m pip install --no-cache-dir torch-spline-conv -f "$WHEEL_INDEX"
"$PYTHON_BIN" -m pip install --no-cache-dir torch-geometric

echo "Done. Verify by running: $PYTHON_BIN -c 'import torch_geometric; print(torch_geometric.__version__)'"
