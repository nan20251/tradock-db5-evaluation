#!/bin/bash
# SASA功能快速验证脚本（无需导入torch_scatter）

echo "=========================================="
echo "TraDock SASA特征部署验证"
echo "=========================================="

PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DIPS_SURFACES="${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
CAPRI_DIR="${CAPRI_DIR:-$PROJECT_ROOT/data/database}"
cd "$PROJECT_ROOT" || exit 1
echo "项目目录: $PROJECT_ROOT"
echo "DIPS 表面目录: $DIPS_SURFACES"
echo "CAPRI 目录: $CAPRI_DIR"

# 验证1: 检查FreeSASA
echo ""
echo "1. FreeSASA安装检查"
echo "--------------------"
python -c "import freesasa; print('  ✓ FreeSASA可用')" 2>/dev/null || echo "  ✗ FreeSASA不可用"

# 验证2: 检查代码修改
echo ""
echo "2. 代码文件检查"
echo "--------------------"

# surface_gen.py
if grep -q "compute_sasa_features" examples/surface_gen.py; then
    echo "  ✓ surface_gen.py: 包含compute_sasa_features函数"
else
    echo "  ✗ surface_gen.py: 缺少SASA函数"
fi

if grep -q "MAX_SASA" examples/surface_gen.py; then
    echo "  ✓ surface_gen.py: 包含MAX_SASA字典"
else
    echo "  ✗ surface_gen.py: 缺少MAX_SASA"
fi

if grep -q "import freesasa" examples/surface_gen.py; then
    echo "  ✓ surface_gen.py: 导入freesasa模块"
else
    echo "  ✗ surface_gen.py: 未导入freesasa"
fi

if grep -q "rSASA" examples/surface_gen.py; then
    echo "  ✓ surface_gen.py: 输出rSASA字段"
else
    echo "  ✗ surface_gen.py: 未输出rSASA"
fi

# data.py
echo ""
if grep -q "rSASA" transformerdock/utils/data.py; then
    echo "  ✓ data.py: 包含rSASA读取代码"
else
    echo "  ✗ data.py: 缺少rSASA读取"
fi

if grep -q "11维" transformerdock/utils/data.py; then
    echo "  ✓ data.py: 文档更新为11维"
else
    echo "  ⚠ data.py: 文档可能未更新（不影响功能）"
fi

# models.py
echo ""
if grep -q "in_channels=11" transformerdock/models.py; then
    echo "  ✓ models.py: 默认in_channels=11"
else
    echo "  ✗ models.py: 默认值未更新"
fi

# 验证3: 数据目录检查
echo ""
echo "3. 数据目录检查"
echo "--------------------"
if [ -f "$DIPS_SURFACES/pairs.csv" ]; then
    n_pairs=$(($(wc -l < "$DIPS_SURFACES/pairs.csv") - 1))
    echo "  ✓ DIPS surfaces: $DIPS_SURFACES ($n_pairs pairs)"
else
    echo "  ⚠ DIPS surfaces 不在本地: $DIPS_SURFACES"
    echo "    AutoDL 上准备完整数据，或设置 DIPS_SURFACES=/path/to/surfaces"
fi

if [ -d "$CAPRI_DIR" ]; then
    n_pdb=$(find "$CAPRI_DIR" -maxdepth 1 -name 'S-T*.pdb' | wc -l | tr -d ' ')
    n_csv=$(find "$CAPRI_DIR" -maxdepth 1 -name 'S-T*.csv' | wc -l | tr -d ' ')
    echo "  CAPRI files: pdb=$n_pdb csv=$n_csv"
    if [ "$n_pdb" -gt 0 ] && [ "$n_csv" -ge "$n_pdb" ]; then
        echo "  ✓ CAPRI 目录有可评估 PDB/CSV"
    else
        echo "  ⚠ CAPRI 本地数据不完整；完整数据应放 AutoDL 或设置 CAPRI_DIR"
    fi
else
    echo "  ⚠ CAPRI_DIR 不存在: $CAPRI_DIR"
fi

# 验证4: 辅助文件
echo ""
echo "4. 辅助文件检查"
echo "--------------------"
[ -f "SASA_INTEGRATION_GUIDE.md" ] && echo "  ✓ SASA_INTEGRATION_GUIDE.md" || echo "  ✗ SASA_INTEGRATION_GUIDE.md"
[ -f "scripts/robust_sasa_compute.py" ] && echo "  ✓ scripts/robust_sasa_compute.py" || echo "  ✗ scripts/robust_sasa_compute.py"

# 验证5: 代码详细检查
echo ""
echo "5. 关键代码片段检查"
echo "--------------------"

# 检查compute_sasa_features函数的行数
sasa_lines=$(grep -c "compute_sasa_features\|atom_sasa\|atom_rSASA" examples/surface_gen.py)
echo "  surface_gen.py SASA相关代码行数: $sasa_lines"

# 检查data.py中rSASA的读取
rsasa_read=$(grep -c "rSASA.*verts\|feat_list.*rSASA" transformerdock/utils/data.py)
echo "  data.py rSASA读取代码行数: $rsasa_read"

# 检查models.py的修改
model_changes=$(grep -c "in_channels=11\|11维" transformerdock/models.py)
echo "  models.py 维度更新行数: $model_changes"

echo ""
echo "=========================================="
echo "部署状态总结"
echo "=========================================="

# 统计通过的检查项
checks_passed=0
checks_total=8

grep -q "compute_sasa_features" examples/surface_gen.py && ((checks_passed++))
grep -q "MAX_SASA" examples/surface_gen.py && ((checks_passed++))
grep -q "import freesasa" examples/surface_gen.py && ((checks_passed++))
grep -q "rSASA" examples/surface_gen.py && ((checks_passed++))
grep -q "rSASA" transformerdock/utils/data.py && ((checks_passed++))
grep -q "in_channels=11" transformerdock/models.py && ((checks_passed++))
[ -f "scripts/robust_sasa_compute.py" ] && ((checks_passed++))
python -c "import freesasa" 2>/dev/null && ((checks_passed++))

echo "检查项通过: $checks_passed / $checks_total"
echo ""

if [ $checks_passed -eq $checks_total ]; then
    echo "✓ 所有检查通过！SASA特征部署成功！"
    echo ""
    echo "=========================================="
    echo "下一步: 测试生成表面数据"
    echo "=========================================="
    echo ""
    echo "快速测试（1个样本）："
    echo "  cd $PROJECT_ROOT"
    echo "  python -c \""
    echo "from examples.surface_gen import pdb_to_surface_ply"
    echo "import tempfile, os"
    echo "# 找一个现有的PDB文件"
    echo "test_pdb = '/root/autodl-tmp/dips/pdbs/1A2K.pdb'  # 替换为 AutoDL 上实际文件"
    echo "if os.path.exists(test_pdb):"
    echo "    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:"
    echo "        tmp_ply = f.name"
    echo "    success = pdb_to_surface_ply(test_pdb, tmp_ply, voxel_size=3.5, verbose=True)"
    echo "    if success:"
    echo "        # 检查PLY文件"
    echo "        with open(tmp_ply) as f:"
    echo "            content = f.read()"
    echo "            if 'property float rSASA' in content:"
    echo "                print('✓ PLY文件包含rSASA字段！')"
    echo "            else:"
    echo "                print('✗ PLY文件缺少rSASA字段')"
    echo "        os.remove(tmp_ply)"
    echo "    else:"
    echo "        print('表面生成失败')"
    echo "else:"
    echo "    print(f'PDB文件不存在: {test_pdb}')"
    echo "\""
    echo ""
    echo "完整测试（50个样本）："
    echo "  python examples/prep_dips.py \\"
    echo "      --metadata data/dips/metadata.csv \\"
    echo "      --pdb_dir /root/autodl-tmp/dips/pdbs \\"
    echo "      --split_dir /root/autodl-tmp/dips/split_pdbs \\"
    echo "      --out_dir /root/autodl-tmp/dips_with_sasa_test \\"
    echo "      --voxel_size 3.5 \\"
    echo "      --limit 50"
elif [ $checks_passed -ge 6 ]; then
    echo "⚠ 大部分检查通过（$checks_passed/$checks_total），但有些项目失败"
    echo "可以继续测试，但请注意检查失败的项目"
else
    echo "✗ 多个检查失败（仅$checks_passed/$checks_total通过）"
    echo "请检查部署是否正确执行"
fi

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="
