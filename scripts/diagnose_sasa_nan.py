#!/usr/bin/env python
"""
诊断脚本：检测 SASA 计算中的 NaN 和零特征问题

使用方式：
  python scripts/diagnose_sasa_nan.py /path/to/pdb /path/to/ply
"""
import sys
import os
import numpy as np
import warnings

try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False
    print("[警告] FreeSASA 未安装，跳过 SASA 诊断")

# 残基最大暴露面积（参考值）
MAX_SASA = {
    'ALA': 107.0, 'ARG': 241.0, 'ASN': 151.0, 'ASP': 154.0, 'CYS': 135.0,
    'GLN': 171.0, 'GLU': 177.0, 'GLY': 75.0, 'HIS': 194.0, 'ILE': 175.0,
    'LEU': 170.0, 'LYS': 205.0, 'MET': 185.0, 'PHE': 210.0, 'PRO': 145.0,
    'SER': 115.0, 'THR': 140.0, 'TRP': 255.0, 'TYR': 230.0, 'VAL': 155.0,
}

def diagnose_pdb(pdb_path):
    """诊断 PDB 文件质量和 FreeSASA 计算"""
    print(f"\n[PDB 诊断] {pdb_path}")
    print("=" * 60)
    
    if not os.path.exists(pdb_path):
        print(f"错误: 文件不存在")
        return False
    
    # 检查 PDB 格式
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    atom_lines = [l for l in lines if l.startswith(('ATOM  ', 'HETATM'))]
    print(f"ATOM/HETATM 记录数: {len(atom_lines)}")
    
    # 检查元素列
    missing_element = 0
    element_col_issues = []
    for i, line in enumerate(atom_lines[:10]):  # 检查前10行
        if len(line) < 78:
            element_col_issues.append(f"  行 {i+1}: 长度不足 78 字符 (实际 {len(line)})")
            missing_element += 1
        else:
            element = line[76:78].strip()
            if not element:
                element_col_issues.append(f"  行 {i+1}: 元素列为空")
                missing_element += 1
    
    if element_col_issues:
        print(f"\n✗ 元素列问题（样本）:")
        for issue in element_col_issues[:3]:
            print(issue)
        print(f"  ... (共 {len(atom_lines)} 行中约 {missing_element} 行有问题)")
    else:
        print("✓ 元素列完整")
    
    # 如果有 FreeSASA，尝试计算
    if not HAS_FREESASA:
        print("\n⚠ FreeSASA 未安装，无法测试计算")
        return missing_element == 0
    
    print("\n[FreeSASA 计算]")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            structure = freesasa.Structure(pdb_path)
            result = freesasa.calc(structure)
        
        n_atoms = structure.nAtoms()
        atom_sasa = np.array([result.atomArea(i) for i in range(n_atoms)], dtype=np.float32)
        
        print(f"计算成功: {n_atoms} 原子")
        print(f"  SASA 范围: [{atom_sasa.min():.2f}, {atom_sasa.max():.2f}]")
        print(f"  SASA 均值: {atom_sasa.mean():.2f}")
        print(f"  全零计数: {(atom_sasa == 0).sum()}")
        
        if w:
            print(f"\nFreeSASA 警告数: {len(w)}")
            for i, warn in enumerate(w[:3]):
                print(f"  {warn.message}")
            if len(w) > 3:
                print(f"  ... (还有 {len(w)-3} 条警告)")
        
        return (atom_sasa == 0).sum() == 0
    except Exception as e:
        print(f"✗ FreeSASA 计算失败: {e}")
        return False

def diagnose_ply(ply_path):
    """诊断 PLY 文件中的 SASA 特征"""
    print(f"\n[PLY 诊断] {ply_path}")
    print("=" * 60)
    
    if not os.path.exists(ply_path):
        print(f"错误: 文件不存在")
        return
    
    try:
        import ply
        mesh = ply.read(ply_path)
    except Exception as e:
        print(f"错误: 无法读取 PLY 文件: {e}")
        return
    
    # 检查顶点特性
    print(f"顶点数: {len(mesh['vertex'])}")
    dtype_names = mesh['vertex'].dtype.names or []
    print(f"特性: {', '.join(dtype_names)}")
    
    if 'rSASA' in dtype_names:
        rsasa = mesh['vertex']['rSASA']
        print(f"\nrSASA 统计:")
        print(f"  范围: [{rsasa.min():.4f}, {rsasa.max():.4f}]")
        print(f"  均值: {rsasa.mean():.4f}")
        print(f"  全零计数: {(rsasa == 0).sum()}")
        print(f"  NaN 计数: {np.isnan(rsasa).sum()}")
        if np.isnan(rsasa).sum() > 0:
            print("  ✗ 检测到 NaN 值！")
        if (rsasa == 0).sum() > len(rsasa) * 0.5:
            print("  ✗ 超过 50% 的 rSASA 为零，可能计算失败")
    elif 'sasa' in dtype_names:
        sasa = mesh['vertex']['sasa']
        print(f"\nSASA 统计:")
        print(f"  范围: [{sasa.min():.4f}, {sasa.max():.4f}]")
        print(f"  均值: {sasa.mean():.4f}")
        print(f"  全零计数: {(sasa == 0).sum()}")
        if (sasa == 0).sum() > len(sasa) * 0.5:
            print("  ✗ 超过 50% 的 SASA 为零，可能计算失败")
    else:
        print("\n⚠ 未找到 rSASA 或 sasa 特性")

def main():
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  python scripts/diagnose_sasa_nan.py <pdb_path> [ply_path]")
        print("\n示例：")
        print("  python scripts/diagnose_sasa_nan.py data/proteins/1a2b.pdb")
        print("  python scripts/diagnose_sasa_nan.py data/proteins/1a2b.pdb data/surfaces/1a2b.ply")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    pdb_ok = diagnose_pdb(pdb_path)
    
    if len(sys.argv) > 2:
        ply_path = sys.argv[2]
        diagnose_ply(ply_path)
    
    print("\n" + "=" * 60)
    if pdb_ok:
        print("✓ PDB 文件质量良好，SASA 计算应该可以正常进行")
    else:
        print("✗ 检测到问题，建议：")
        print("  1. 用 scripts/fix_pdb_elements.py 修复 PDB 元素列")
        print("  2. 重新生成表面特征")
        print("  3. 查看 FAQ.md 了解更多信息")

if __name__ == '__main__':
    main()
