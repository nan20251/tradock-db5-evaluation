#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package current TraDock fixes for deployment on AutoDL."""

from datetime import datetime
from pathlib import Path
import shutil
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / 'tradock_sasa_patch'
OUTPUT_TAR = PROJECT_ROOT / 'tradock_sasa_patch.tar.gz'

FILES_TO_PACK = [
    'examples/surface_gen.py',
    'examples/train.py',
    'examples/eval_capri_fast.py',
    'examples/train_native_vs_decoy_v2.py',
    'transformerdock/utils/data.py',
    'transformerdock/models.py',
    'scripts/robust_sasa_compute.py',
    'scripts/fix_pdb_elements.py',
    'scripts/verify_sasa_deployment.sh',
    'scripts/run_step2_pretrain.sh',
    'scripts/run_step7_eval.sh',
    'scripts/run_all.sh',
    'scripts/audit_dips_surfaces.py',
    'scripts/filter_bad_dips_pairs.py',
    'scripts/check_data_range.py',
    'scripts/check_nan_samples.py',
    'scripts/check_ply_dimensions.py',
    'scripts/check_ply_fields.py',
    'scripts/check_model_quality.py',
    'requirements.txt',
    'environment.yml',
    'SASA_INTEGRATION_GUIDE.md',
]


def copy_file(src_rel):
    src = PROJECT_ROOT / src_rel
    if not src.exists():
        print(f"  [跳过] 文件不存在: {src_rel}")
        return False
    dst = TEMP_DIR / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [OK] {src_rel}")
    return True


def create_deploy_script():
    return """#!/bin/bash
set -euo pipefail

TRADOCK_DIR="${TRADOCK_DIR:-}"
if [ -z "$TRADOCK_DIR" ]; then
    if [ -d "/root/TraDock" ]; then
        TRADOCK_DIR="/root/TraDock"
    elif [ -d "/root/autodl-tmp/TraDock" ]; then
        TRADOCK_DIR="/root/autodl-tmp/TraDock"
    elif [ -d "$HOME/TraDock" ]; then
        TRADOCK_DIR="$HOME/TraDock"
    else
        echo "[错误] 找不到 TraDock 目录，请先设置 TRADOCK_DIR=/path/to/TraDock" >&2
        exit 1
    fi
fi

BACKUP_DIR="$TRADOCK_DIR/backup_before_patch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

FILES=(
    "examples/surface_gen.py"
    "examples/train.py"
    "examples/eval_capri_fast.py"
    "examples/train_native_vs_decoy_v2.py"
    "transformerdock/utils/data.py"
    "transformerdock/models.py"
    "scripts/robust_sasa_compute.py"
    "scripts/fix_pdb_elements.py"
    "scripts/verify_sasa_deployment.sh"
    "scripts/run_step2_pretrain.sh"
    "scripts/run_step7_eval.sh"
    "scripts/run_all.sh"
    "scripts/audit_dips_surfaces.py"
    "scripts/filter_bad_dips_pairs.py"
    "scripts/check_data_range.py"
    "scripts/check_nan_samples.py"
    "scripts/check_ply_dimensions.py"
    "scripts/check_ply_fields.py"
    "scripts/check_model_quality.py"
    "SASA_INTEGRATION_GUIDE.md"
)

echo "TraDock 目录: $TRADOCK_DIR"
echo "备份目录: $BACKUP_DIR"

for file in "${FILES[@]}"; do
    if [ -f "$TRADOCK_DIR/$file" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        cp "$TRADOCK_DIR/$file" "$BACKUP_DIR/$file"
    fi
done

for file in "${FILES[@]}"; do
    if [ -f "./$file" ]; then
        mkdir -p "$TRADOCK_DIR/$(dirname "$file")"
        cp "./$file" "$TRADOCK_DIR/$file"
        echo "  [OK] 更新 $file"
    else
        echo "  [跳过] 补丁中不存在 $file"
    fi
done

chmod +x "$TRADOCK_DIR"/scripts/*.sh 2>/dev/null || true

cd "$TRADOCK_DIR"
echo ""
echo "运行部署验证:"
bash scripts/verify_sasa_deployment.sh

echo ""
echo "部署完成。可配置路径:"
echo "  TRADOCK_DIR=$TRADOCK_DIR"
echo "  DIPS_SURFACES=${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
echo "  CAPRI_DIR=${CAPRI_DIR:-$TRADOCK_DIR/data/database}"
"""


def create_readme():
    return f"""TraDock 修复补丁包
===================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

约定
----
- 模型输入特征: 11 维
- 原始 PLY vertex 字段通常是 14 个
- DIPS surfaces: DIPS_SURFACES
- CAPRI: CAPRI_DIR

部署
----
```bash
tar -xzf tradock_sasa_patch.tar.gz
cd tradock_sasa_patch
TRADOCK_DIR=/path/to/TraDock bash deploy_on_autodl.sh
```

验证
----
```bash
cd "$TRADOCK_DIR"
bash scripts/verify_sasa_deployment.sh
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_data_range.py
DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full python scripts/check_model_quality.py \\
  --checkpoint "$TRADOCK_DIR/Trained_models/pretrain_with_sasa/TransformerDock_best.chk"
```

评估
----
```bash
CAPRI_DIR="$TRADOCK_DIR/data/database" \\
CHECKPOINT="$TRADOCK_DIR/Trained_models/pretrain_with_sasa/TransformerDock_best.chk" \\
bash scripts/run_step7_eval.sh
```
"""


def create_patch_package():
    print("=" * 60)
    print("TraDock 修复补丁打包工具")
    print("=" * 60)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()

    print("\n正在打包文件...")
    copied = 0
    for src_rel in FILES_TO_PACK:
        copied += int(copy_file(src_rel))

    deploy_script = TEMP_DIR / 'deploy_on_autodl.sh'
    deploy_script.write_text(create_deploy_script(), encoding='utf-8')
    deploy_script.chmod(0o755)
    print("  [OK] deploy_on_autodl.sh")

    readme = TEMP_DIR / 'README_DEPLOYMENT.txt'
    readme.write_text(create_readme(), encoding='utf-8')
    print("  [OK] README_DEPLOYMENT.txt")

    if OUTPUT_TAR.exists():
        OUTPUT_TAR.unlink()
    with tarfile.open(OUTPUT_TAR, 'w:gz') as tar:
        tar.add(TEMP_DIR, arcname=TEMP_DIR.name)
    shutil.rmtree(TEMP_DIR)

    print("\n[OK] 打包完成")
    print(f"  文件: {OUTPUT_TAR}")
    print(f"  文件数: {copied}")
    print(f"  大小: {OUTPUT_TAR.stat().st_size / 1024:.2f} KB")
    print(f"  scp {OUTPUT_TAR.name} root@<your-autodl-ip>:/root/")


if __name__ == '__main__':
    create_patch_package()
