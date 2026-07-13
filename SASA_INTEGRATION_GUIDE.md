# TraDock SASA / NaN 修复说明

当前代码使用 11 维模型输入特征：

```text
nx, ny, nz,
charge, hydrophobicity,
hbond_donor, hbond_acceptor,
curvature, shape_index, aa_polar, rSASA
```

注意：原始 PLY 的 `vertex` 字段通常是 14 个，其中 `x/y/z` 是坐标，剩余 11 个才是模型输入特征。

## 已修复内容

- `examples/surface_gen.py`
  - 填补 PDB 元素列，避免 FreeSASA 把 `CA/CD` 误判成钙/镉。
  - FreeSASA 失败时使用 `scripts/robust_sasa_compute.py` 的非零回退。
  - 对法向量、曲率、shape index、rSASA 做 NaN/Inf 清理和范围裁剪。

- `transformerdock/utils/data.py`
  - `read_ply()` 统一读取 11 维特征。
  - 缺字段时补 0，旧/不完整 PLY 不会直接崩溃。
  - 无 face 或空 edge 时自动构建 kNN 边。

- `transformerdock/models.py`
  - 默认 `in_channels=11`。
  - Cross-attention、MDN 输出和能量输出增加 NaN/Inf 防护。
  - 空 interface 时返回可处理的空张量和零能量，避免训练/验证误报。

- `examples/train.py`
  - 训练和验证会跳过 NaN/Inf batch 与无 interface batch。
  - 验证集没有有效 batch 时返回 `inf`，不再返回 0。

## 路径约定

脚本默认从自身位置定位项目根目录，以下路径可用环境变量覆盖：

```bash
export TRADOCK_DIR=/root/TraDock
export DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full
export CAPRI_DIR=/root/TraDock/data/database
export CHECKPOINT=/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

本地仓库不再保留不完整的 `data/database`。CAPRI 完整数据应放在 AutoDL，并通过 `CAPRI_DIR` 指定。

## 验证命令

```bash
cd "$TRADOCK_DIR"
bash scripts/verify_sasa_deployment.sh

DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_ply_fields.py
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_ply_dimensions.py
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_data_range.py
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_nan_samples.py
python scripts/audit_dips_surfaces.py "$DIPS_SURFACES" --samples 1yk0_A_B 1u0c_A_B
```

模型抽样检查：

```bash
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full \
python scripts/check_model_quality.py \
  --checkpoint Trained_models/pretrain_with_sasa/TransformerDock_best.chk
```

## 训练

```bash
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full bash scripts/run_step2_pretrain.sh
```

等价手动命令：

```bash
python examples/train.py \
  --data_dir "$DIPS_SURFACES" \
  --save_dir Trained_models/pretrain_with_sasa \
  --epochs 30 \
  --batch_size 2 \
  --lr 1e-4 \
  --contrast_weight 0.0
```

## CAPRI 评估

```bash
CAPRI_DIR=/root/TraDock/data/database \
CHECKPOINT=/root/TraDock/Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
bash scripts/run_step7_eval.sh
```

## 常见问题

### SASA 出现 NaN 或全 0

先检查 PDB 元素列：

```bash
python scripts/fix_pdb_elements.py input.pdb -o fixed.pdb
```

再生成 surface。当前 `surface_gen.py` 已经会自动补元素列并做非零回退。

### 验证 loss 为 0

当前训练脚本不会再把“全跳过 batch”当成 0。若验证输出 `inf`，说明验证集没有有效 interface 或数据仍有异常，应先运行：

```bash
python scripts/check_data_range.py "$DIPS_SURFACES"
python scripts/check_nan_samples.py --data_dir "$DIPS_SURFACES"
```

### 特征维度混淆

`check_ply_fields.py` 看的是原始 PLY 字段，应该通常是 14 个 vertex 字段。  
`check_ply_dimensions.py` 看的是模型读入特征，应该是 11 维。
