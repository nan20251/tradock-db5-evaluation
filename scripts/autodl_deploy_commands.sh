#!/bin/bash
# 在 AutoDL 上执行的完整部署命令

set -euo pipefail

TRADOCK_DIR="${TRADOCK_DIR:-/root/TraDock}"

# 1. 回到root目录
cd /root

# 2. 检查文件是否上传成功
echo "检查上传的文件..."
if [ -f "tradock_sasa_patch.tar.gz" ]; then
    echo "✓ 找到 tradock_sasa_patch.tar.gz"
    ls -lh tradock_sasa_patch.tar.gz
else
    echo "✗ 未找到 tradock_sasa_patch.tar.gz"
    echo "请先从本地上传文件："
    echo "  scp tradock_sasa_patch.tar.gz root@<your-ip>:/root/"
    exit 1
fi

# 3. 解压
echo ""
echo "解压补丁包..."
tar -xzf tradock_sasa_patch.tar.gz
cd tradock_sasa_patch

# 4. 显示内容
echo ""
echo "补丁包内容:"
ls -lh

# 5. 运行部署脚本
echo ""
echo "开始部署..."
bash deploy_on_autodl.sh

echo ""
echo "================================================================"
echo "部署完成！"
echo "================================================================"
echo ""
echo "下一步："
echo "  1. 查看使用指南："
echo "     cat $TRADOCK_DIR/README.md"
echo ""
echo "  2. 快速测试（50个样本）："
echo "     cd $TRADOCK_DIR"
echo "     python examples/prep_dips.py \\"
echo "         --metadata data/dips/metadata.csv \\"
echo "         --pdb_dir /root/autodl-tmp/dips/pdbs \\"
echo "         --split_dir /root/autodl-tmp/dips/split_pdbs \\"
echo "         --out_dir /root/autodl-tmp/dips_with_sasa_test \\"
echo "         --voxel_size 3.5 \\"
echo "         --limit 50"
echo ""
echo "  3. 查看特征维度："
echo "     python -c \"from transformerdock.utils.data import read_ply; \\"
echo "     d=read_ply('/root/autodl-tmp/dips_with_sasa_test/<first_file>_receptor.ply'); \\"
echo "     print(f'特征维度: {d.x.shape[1]}'); \\"
echo "     print(f'rSASA: {d.x[:, -1]}')\""
echo ""
echo "  4. 路径对应验证："
echo "     TRADOCK_DIR=$TRADOCK_DIR DIPS_SURFACES=/root/autodl-tmp/dips_with_sasa_full \\"
echo "     CAPRI_DIR=$TRADOCK_DIR/data/database bash scripts/verify_sasa_deployment.sh"
echo "================================================================"
