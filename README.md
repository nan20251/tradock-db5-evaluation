# TraDock - quick setup

ppi评分模型

Brief notes to get the project running locally using conda.

1) Create conda env

```bash
conda env create -f environment.yml
conda activate tradock
```

2) Install PyTorch and GPU support

Follow https://pytorch.org/ to install the correct `pytorch` and `cudatoolkit` for your GPU.

3) Install PyTorch Geometric (PyG) native extensions

After matching PyTorch/CUDA, install PyG wheels from the official index. Example placeholder:

```bash
# replace TORCH_TAG and CUDA_TAG with detected values (see run_step0_step1.sh)
pip install torch-scatter -f https://data.pyg.org/whl/torch-TORCH_TAG+CUDA_TAG.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-TORCH_TAG+CUDA_TAG.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-TORCH_TAG+CUDA_TAG.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-TORCH_TAG+CUDA_TAG.html
pip install torch-geometric
```

4) Sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import freesasa; print('FreeSASA OK')"
python -c "import torch_geometric; print('PyG', torch_geometric.__version__)"
```

Notes
- This repo includes helper scripts `run_step0_env_4090.sh` and `run_step0_step1.sh` to detect and guide installs.
- If you see undefined-symbol errors for PyG native libs, reinstall the PyG wheels matching your exact `torch`+`cuda`.
