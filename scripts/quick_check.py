#!/usr/bin/env python
"""Quick sanity check: verify torch, pyg, and freesasa imports."""
import sys
import importlib

def check_import(name):
    try:
        m = importlib.import_module(name)
        v = getattr(m, '__version__', 'N/A')
        print(f'  ✓ {name:25s} {v}')
        return True
    except Exception as e:
        print(f'  ✗ {name:25s} {e!r}')
        return False

print('Python environment:')
print(f'  version: {sys.version.split()[0]}')
print(f'  executable: {sys.executable}')

print('\nCore imports:')
all_ok = True
for pkg in ['torch', 'numpy', 'scipy', 'pandas']:
    all_ok &= check_import(pkg)

print('\nPyTorch/CUDA info:')
try:
    import torch
    print(f'  version: {torch.__version__}')
    print(f'  cuda: {torch.version.cuda}')
    print(f'  cuda_available: {torch.cuda.is_available()}')
except Exception as e:
    print(f'  error: {e!r}')

print('\nPyTorch Geometric:')
for pkg in ['torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']:
    check_import(pkg)

print('\nOther packages:')
for pkg in ['freesasa', 'biopython']:
    check_import(pkg)

sys.exit(0 if all_ok else 1)
