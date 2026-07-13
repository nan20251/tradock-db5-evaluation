# TraDock 项目参考

## 当前结构

- `transformerdock/`: 核心模型和数据加载
- `examples/`: 训练、评估、数据准备脚本
- `scripts/`: 检查、修复、部署和流程脚本
- `Trained_models/`: 训练产物
- `data/`: 轻量元数据和排除列表
- `results/`: 评估结果

## 当前约定

- 模型输入特征为 11 维
- 原始 PLY `vertex` 字段通常为 14 个
- DIPS 完整表面数据放在 `DIPS_SURFACES`
- CAPRI 完整数据放在 `CAPRI_DIR`

## 推荐检查顺序

```bash
cd "$TRADOCK_DIR"
bash scripts/verify_sasa_deployment.sh
python scripts/check_ply_fields.py "$DIPS_SURFACES"
python scripts/check_ply_dimensions.py "$DIPS_SURFACES"
python scripts/check_data_range.py "$DIPS_SURFACES"
python scripts/audit_dips_surfaces.py "$DIPS_SURFACES" --samples 1yk0_A_B 1u0c_A_B
python scripts/check_model_quality.py --data_dir "$DIPS_SURFACES" \
  --checkpoint "$CHECKPOINT"
```

## 推荐训练和评估

```bash
bash scripts/run_step2_pretrain.sh
bash scripts/run_step7_eval.sh
```

## 说明

旧的 AutoDL 部署长文档、恢复指南和归档流程已移除，避免继续引用不存在的测试脚本和旧路径。
