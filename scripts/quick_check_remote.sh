#!/bin/bash
# Quick sanity check for remote tradock_pyg environment

PYTHON=/root/miniconda3/envs/tradock_pyg/bin/python

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: tradock_pyg environment not found at $PYTHON"
    exit 1
fi

echo "Python environment:"
echo "  executable: $PYTHON"
$PYTHON --version

echo -e "\nCore imports:"
$PYTHON - <<'PY'
import sys, importlib
for pkg in ['torch', 'numpy', 'scipy', 'pandas']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'N/A')
        print(f'  ✓ {pkg:25s} {v}')
    except Exception as e:
        print(f'  ✗ {pkg:25s} {e!r}')
PY

echo -e "\nPyTorch Geometric packages:"
$PYTHON - <<'PY'
import sys, importlib
for pkg in ['torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'N/A')
        print(f'  ✓ {pkg:25s} {v}')
    except Exception as e:
        print(f'  ✗ {pkg:25s} {e!r}')
PY

echo -e "\nOther packages:"
$PYTHON - <<'PY'
import sys, importlib
for pkg in ['freesasa', 'biopython']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'N/A')
        print(f'  ✓ {pkg:25s} {v}')
    except Exception as e:
        print(f'  ✗ {pkg:25s} {e!r}')
PY
