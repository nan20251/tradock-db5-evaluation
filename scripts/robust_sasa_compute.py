#!/usr/bin/env python
"""
改进的 SASA 计算脚本：增强容错能力和默认值策略

问题：如果 FreeSASA 计算失败（比如 PDB 元素列缺失），原来直接返回全零。
      全零特征导致：
      1. 梯度为零，模型无法学习
      2. 如果 SASA 特征很重要，模型输出可能 NaN

解决方案：
1. 检测 PDB 质量，必要时自动修复元素列
2. 如果 FreeSASA 失败，用更合理的默认值（如均匀值或按氨基酸类型）
3. 验证输出，检测 NaN 和极端值
"""
import sys
import numpy as np
import warnings
import os
import tempfile
import shutil
import contextlib
import re

try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False

# 残基最大暴露面积
MAX_SASA = {
    'ALA': 107.0, 'ARG': 241.0, 'ASN': 151.0, 'ASP': 154.0, 'CYS': 135.0,
    'GLN': 171.0, 'GLU': 177.0, 'GLY': 75.0, 'HIS': 194.0, 'ILE': 175.0,
    'LEU': 170.0, 'LYS': 205.0, 'MET': 185.0, 'PHE': 210.0, 'PRO': 145.0,
    'SER': 115.0, 'THR': 140.0, 'TRP': 255.0, 'TYR': 230.0, 'VAL': 155.0,
}

TWO_LETTER_ELEMENTS = {
    'CL', 'BR', 'NA', 'MG', 'AL', 'SI', 'FE', 'ZN', 'CA', 'MN', 'CO', 'CU',
    'NI', 'CD', 'HG', 'SE', 'LI', 'BE', 'NE', 'AR', 'KR', 'XE',
}


def infer_element(atom_name, record_name='ATOM'):
    if record_name == 'ATOM' and atom_name[:1] == ' ':
        for ch in atom_name:
            if ch.isalpha():
                return ch.upper()
    name = re.sub(r"^[0-9]+", "", atom_name.strip())
    letters = ''.join(ch for ch in name if ch.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in TWO_LETTER_ELEMENTS:
        return letters[:2].title()
    if letters:
        return letters[0]
    return 'C'

def fix_pdb_elements_inplace(pdb_path):
    """在 PDB 文件中填充缺失的元素列"""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    fixed = []
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            if len(line) < 78:
                line = line.rstrip('\n').ljust(78) + '\n'
            element = line[76:78].strip()
            if not element:
                elem = infer_element(line[12:16], line[:6].strip())
                # 保持 PDB 格式：cols 77-78
                new_line = line[:76] + elem.rjust(2) + line[78:]
                fixed.append(new_line)
            else:
                fixed.append(line)
        else:
            fixed.append(line)
    
    with open(pdb_path, 'w') as f:
        f.writelines(fixed)

@contextlib.contextmanager
def _suppress_c_stderr(enabled=True):
    if not enabled:
        yield
        return
    fd = 2
    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


def _fit_length(atom_sasa, atom_rSASA, res_names, strategy='mean'):
    n = len(res_names)
    atom_sasa = np.nan_to_num(
        np.asarray(atom_sasa, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    atom_rSASA = np.nan_to_num(
        np.asarray(atom_rSASA, dtype=np.float32), nan=0.5, posinf=0.5, neginf=0.5
    )
    atom_sasa = np.clip(atom_sasa, 0.0, None)
    atom_rSASA = np.clip(atom_rSASA, 0.0, 2.0)
    if len(atom_rSASA) == n:
        return atom_sasa, atom_rSASA

    fallback_sasa, fallback_rsasa, _ = _compute_sasa_fallback(res_names, strategy)
    m = min(n, len(atom_rSASA))
    if m > 0:
        fallback_sasa[:m] = atom_sasa[:m]
        fallback_rsasa[:m] = atom_rSASA[:m]
    return fallback_sasa, fallback_rsasa


def compute_sasa_robust(pdb_path, coords, res_names, strategy='mean', quiet=False):
    """
    增强版 SASA 计算：失败时提供合理的默认值
    
    Args:
        pdb_path: PDB 文件路径
        coords: 原子坐标 [N, 3]
        res_names: 残基名称列表 [N]
        strategy: 失败策略
            'zero': 返回全零（不推荐）
            'mean': 按氨基酸类型的平均值（推荐）
            'max': 按氨基酸类型的最大值
    
    Returns:
        atom_sasa: 原子 SASA 值 [N]
        atom_rSASA: 相对 SASA 值 [N]
        success: 是否成功计算（True 表示用 FreeSASA，False 表示用默认值）
    """
    if not HAS_FREESASA:
        if not quiet:
            print(f"[警告] FreeSASA 未安装，使用默认值")
        return _compute_sasa_fallback(res_names, strategy)
    
    # 创建临时副本并修复
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
            tmp_pdb = tmp.name
        
        shutil.copy(pdb_path, tmp_pdb)
        fix_pdb_elements_inplace(tmp_pdb)
        
        # 尝试 FreeSASA 计算
        with warnings.catch_warnings(record=True), _suppress_c_stderr(quiet):
            warnings.simplefilter("always")
            structure = freesasa.Structure(tmp_pdb)
            result = freesasa.calc(structure)
        
        n_atoms = structure.nAtoms()
        atom_sasa = np.zeros(n_atoms, dtype=np.float32)
        atom_resnames = []
        
        for i in range(n_atoms):
            atom_sasa[i] = result.atomArea(i)
            atom_resnames.append(structure.residueName(i).strip())
        
        # 计算相对 SASA
        atom_rSASA = np.array([
            sasa / MAX_SASA.get(resname, 150.0)
            for sasa, resname in zip(atom_sasa, atom_resnames)
        ], dtype=np.float32)
        
        # 裁剪异常值
        atom_rSASA = np.clip(atom_rSASA, 0, 2.0)
        
        atom_sasa, atom_rSASA = _fit_length(atom_sasa, atom_rSASA, res_names, strategy)

        os.remove(tmp_pdb)
        return atom_sasa, atom_rSASA, True
    
    except Exception as e:
        if not quiet:
            print(f"[FreeSASA 失败] {pdb_path}: {e}")
            print(f"[回退] 使用 {strategy} 策略生成默认值")
        if 'tmp_pdb' in locals() and os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)
        return _compute_sasa_fallback(res_names, strategy)

def _compute_sasa_fallback(res_names, strategy='mean'):
    """生成默认 SASA 值（当计算失败时）"""
    n = len(res_names)
    
    if strategy == 'zero':
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), False
    
    # 按残基类型的统计值
    atom_rSASA = np.zeros(n, dtype=np.float32)
    for i, resname in enumerate(res_names):
        max_sasa = MAX_SASA.get(resname, 150.0)
        if strategy == 'mean':
            # 用最大值的 0.5（平均暴露度）
            atom_rSASA[i] = 0.5
        elif strategy == 'max':
            atom_rSASA[i] = 1.0
    
    atom_sasa = atom_rSASA * np.array([MAX_SASA.get(r, 150.0) for r in res_names], dtype=np.float32)
    
    return atom_sasa, atom_rSASA, False

def main():
    """测试脚本：演示改进版 SASA 计算"""
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  python scripts/robust_sasa_compute.py <pdb_path>")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    
    # 模拟测试
    res_names = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE'] * 10
    coords = np.random.randn(50, 3)
    
    print(f"测试计算: {pdb_path}")
    print(f"  残基: {len(res_names)}")
    
    atom_sasa, atom_rSASA, success = compute_sasa_robust(pdb_path, coords, res_names, strategy='mean')
    
    print(f"\n计算结果:")
    print(f"  成功: {success}")
    print(f"  SASA 范围: [{atom_sasa.min():.2f}, {atom_sasa.max():.2f}]")
    print(f"  rSASA 范围: [{atom_rSASA.min():.4f}, {atom_rSASA.max():.4f}]")
    print(f"  NaN 计数: {np.isnan(atom_rSASA).sum()}")
    print(f"  全零计数: {(atom_rSASA == 0).sum()}")

if __name__ == '__main__':
    main()
