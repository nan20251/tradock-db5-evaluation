# TraDock 常见问题与解决方案

## 环境与依赖

### PyG 原生扩展导入失败（undefined symbol）

**问题**：运行脚本时报错 `undefined symbol: _ZN3c1017RegisterOperatorsD1Ev`，来自 `torch-sparse`、`torch-cluster`、`torch-spline-conv`。

**原因**：PyG 原生轮子与安装的 PyTorch/CUDA 版本不匹配（ABI 不兼容）。

**解决方案**：
1. 检查当前 PyTorch 版本：
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
2. 根据输出的 `torch` 版本和 CUDA tag，从 PyG 官方轮子索引重新安装：
   ```bash
   # 例如: torch=2.4.1, cuda=cu121
   pip install --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   pip install --force-reinstall torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   pip install --force-reinstall torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   pip install --force-reinstall torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   ```
3. 验证导入：
   ```bash
   python scripts/quick_check.py
   ```

### FreeSASA 导入失败

**问题**：`ImportError: No module named 'freesasa'`

**解决方案**：
```bash
# 使用 conda
conda install -c conda-forge freesasa

# 或使用 pip
pip install freesasa
```

## 数据与 PDB 处理

### FreeSASA 报警告：`unknown atom` 或 `guessing element`

**问题**：运行日志中充满 `FreeSASA: warning: atom 'ILE  CD ' unknown` 等信息。

**原因**：PDB 文件缺少标准的元素列（PDB 格式第 77-78 列）。

**解决方案**：使用 `scripts/fix_pdb_elements.py` 修复 PDB 文件：
```bash
python scripts/fix_pdb_elements.py input.pdb -o fixed.pdb
```

然后在代码中使用修复后的 PDB 文件重新运行计算。

## 模型训练与推理

### NaN 或 Inf 出现在模型输出中

**诊断步骤**：
1. 检查输入数据范围是否合理（使用 `check_data_range.py` 等脚本）。
2. 确认 PyG 原生扩展正确加载（运行 `scripts/quick_check.py`）。
3. **检查 SASA 特征计算是否成功**：
   ```bash
   python scripts/diagnose_sasa_nan.py /path/to/protein.pdb /path/to/surface.ply
   ```

**常见原因与修复**：

#### 原因 1: SASA 特征为全零或计算失败

**症状**：
- 训练日志中大量 FreeSASA 警告（`unknown atom`, `guessing element`）
- SASA 特征全为零，导致梯度为零
- 模型输出 NaN 或梯度爆炸

**修复步骤**：
1. 诊断 PDB 质量：
   ```bash
   python scripts/diagnose_sasa_nan.py protein.pdb
   ```
   如果输出显示 "元素列问题", 继续步骤 2

2. 修复 PDB 元素列：
   ```bash
   python scripts/fix_pdb_elements.py protein.pdb -o protein_fixed.pdb
   ```

3. 重新生成表面特征：
   ```bash
   python examples/surface_gen.py protein_fixed.pdb
   ```

4. 验证 SASA 计算：
   ```bash
   python scripts/diagnose_sasa_nan.py protein_fixed.pdb surface.ply
   ```
   确保 rSASA 不全为零，且 NaN 计数为 0

#### 原因 2: PyG 原生扩展加载失败

- 图操作（聚合、稀疏矩阵运算）依赖 PyG 原生扩展，若扩展加载失败会导致零或 NaN 输出。
- 参考上面 "PyG 原生扩展导入失败" 部分修复。

#### 原因 3: 数据异常（极值、NaN）

- 使用 `examples/check_data_range.py` 或 `examples/check_nan_samples.py` 检查输入数据。

## 快速验证

快速运行环境自检：
```bash
# 本地检查
python scripts/quick_check.py

# 或查看仓库内的诊断脚本
python examples/eval_capri_fast.py  # 快速评估示例
```

## 环境信息

项目默认配置：
- **Python**: 3.8
- **PyTorch**: 2.4.1 + CUDA 12.1
- **PyG**: 2.6.1
- **FreeSASA**: latest

如需其他 PyTorch 版本，编辑 `environment.yml` 或按上述 PyG 轮子索引重新安装。
