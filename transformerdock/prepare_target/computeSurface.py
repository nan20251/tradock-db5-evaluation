"""
蛋白质表面网格生成。
从 PDB 文件生成 .ply 格式的表面网格，用于 TransformerDock 输入。

依赖：
    - MSMS（表面生成）或 PyMesh / Open3D（备用）
    - PDB2PQR（静电势计算，可选）
    - APBS（静电势计算，可选）

如果没有安装这些工具，提供基于 alpha-carbon 的简化表面作为备用。
"""

import os
import subprocess
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge


# ─────────────────────────────────────────────────────────────
# 主入口：从 PDB 生成表面网格
# ─────────────────────────────────────────────────────────────

def compute_surface(pdb_file, output_ply=None, ligand_pdb=None,
                    dist_threshold=10.0, method='msms'):
    """
    从 PDB 文件生成蛋白质表面网格（.ply 格式）。

    参数：
        pdb_file       : 输入 PDB 文件路径
        output_ply     : 输出 .ply 文件路径（默认与 pdb_file 同名）
        ligand_pdb     : 配体 PDB 文件（用于裁剪结合位点附近的表面）
        dist_threshold : 结合位点距离阈值（Å），只保留配体周围的表面
        method         : 'msms'（推荐）或 'simple'（备用，基于 Cα）

    返回：
        output_ply 文件路径
    """
    if output_ply is None:
        output_ply = pdb_file.replace('.pdb', '.ply')

    if method == 'msms' and _check_msms():
        _compute_msms_surface(pdb_file, output_ply, ligand_pdb, dist_threshold)
    else:
        print(f"[警告] MSMS 未找到，使用简化 Cα 表面（精度较低）")
        _compute_simple_surface(pdb_file, output_ply, ligand_pdb, dist_threshold)

    return output_ply


def _check_msms():
    """检查 MSMS 是否可用。"""
    try:
        result = subprocess.run(['msms', '-h'], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─────────────────────────────────────────────────────────────
# MSMS 表面生成
# ─────────────────────────────────────────────────────────────

def _compute_msms_surface(pdb_file, output_ply, ligand_pdb=None, dist_threshold=10.0):
    """
    使用 MSMS 生成分子表面。
    步骤：PDB → xyzr → MSMS → .vert/.face → .ply
    """
    base = output_ply.replace('.ply', '')
    xyzr_file = base + '.xyzr'
    vert_file = base + '.vert'
    face_file = base + '.face'

    # 1. PDB → xyzr（原子坐标 + 范德华半径）
    _pdb_to_xyzr(pdb_file, xyzr_file)

    # 2. 运行 MSMS
    cmd = ['msms', '-if', xyzr_file, '-of', base, '-probe_radius', '1.5',
           '-density', '3.0', '-no_header']
    subprocess.run(cmd, check=True, capture_output=True)

    # 3. 读取 .vert 和 .face，写入 .ply
    vertices, normals = _read_msms_vert(vert_file)
    faces = _read_msms_face(face_file)

    # 4. 如果提供了配体，裁剪结合位点附近的表面
    if ligand_pdb is not None:
        lig_coords = _read_pdb_coords(ligand_pdb)
        mask = _get_binding_site_mask(vertices, lig_coords, dist_threshold)
        vertices, normals, faces = _crop_surface(vertices, normals, faces, mask)

    _write_ply(output_ply, vertices, normals, faces)

    # 清理临时文件
    for f in [xyzr_file, vert_file, face_file]:
        if os.path.exists(f):
            os.remove(f)


def _pdb_to_xyzr(pdb_file, xyzr_file):
    """将 PDB 文件转换为 MSMS 需要的 xyzr 格式（x y z radius）。"""
    # 范德华半径（简化版）
    vdw_radii = {
        'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
        'P': 1.8, 'H': 1.2, 'F': 1.47, 'CL': 1.75,
        'BR': 1.85, 'I': 1.98,
    }
    default_radius = 1.7

    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip().upper() if len(line) > 76 else line[12:14].strip()[0].upper()
                r = vdw_radii.get(element, default_radius)
                lines.append(f'{x:.3f} {y:.3f} {z:.3f} {r:.3f}\n')

    with open(xyzr_file, 'w') as f:
        f.writelines(lines)


def _read_msms_vert(vert_file):
    """读取 MSMS .vert 文件，返回顶点坐标和法向量。"""
    vertices, normals = [], []
    with open(vert_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
    return np.array(vertices), np.array(normals)


def _read_msms_face(face_file):
    """读取 MSMS .face 文件，返回三角面片索引（0-indexed）。"""
    faces = []
    with open(face_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                faces.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])
    return np.array(faces)


# ─────────────────────────────────────────────────────────────
# 简化表面（备用，基于 Cα 坐标）
# ─────────────────────────────────────────────────────────────

def _compute_simple_surface(pdb_file, output_ply, ligand_pdb=None, dist_threshold=10.0):
    """
    简化表面：直接用 Cα 坐标作为表面节点（无真实网格）。
    精度低，仅用于测试和调试。
    """
    coords = _read_pdb_ca_coords(pdb_file)

    if ligand_pdb is not None:
        lig_coords = _read_pdb_coords(ligand_pdb)
        mask = _get_binding_site_mask(coords, lig_coords, dist_threshold)
        coords = coords[mask]

    # 用 KNN 构建简单连接（每个节点连接最近的 8 个邻居）
    n = len(coords)
    normals = np.zeros((n, 3))
    normals[:, 2] = 1.0  # 占位法向量

    # 无面片（简化模式）
    faces = np.zeros((0, 3), dtype=int)
    _write_ply(output_ply, coords, normals, faces)


def _read_pdb_ca_coords(pdb_file):
    """读取 PDB 文件中所有 Cα 原子坐标。"""
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if (line.startswith('ATOM') and line[12:16].strip() == 'CA'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else np.zeros((1, 3))


def _read_pdb_coords(pdb_file):
    """读取 PDB 文件中所有重原子坐标。"""
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else np.zeros((1, 3))


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────

def _get_binding_site_mask(surface_coords, ligand_coords, threshold):
    """返回距配体 threshold Å 以内的表面节点 mask。"""
    diff = surface_coords[:, None, :] - ligand_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))   # [N_surf, N_lig]
    return dist.min(axis=1) < threshold


def _crop_surface(vertices, normals, faces, mask):
    """根据 mask 裁剪表面，更新面片索引。"""
    new_idx = np.full(len(vertices), -1, dtype=int)
    new_idx[mask] = np.arange(mask.sum())

    new_verts = vertices[mask]
    new_norms = normals[mask]

    # 只保留三个顶点都在 mask 内的面片
    valid_faces = []
    for face in faces:
        if all(mask[v] for v in face):
            valid_faces.append([new_idx[v] for v in face])
    new_faces = np.array(valid_faces) if valid_faces else np.zeros((0, 3), dtype=int)

    return new_verts, new_norms, new_faces


def _write_ply(ply_path, vertices, normals, faces):
    """将顶点、法向量和面片写入 .ply 文件（ASCII 格式）。"""
    n_verts = len(vertices)
    n_faces = len(faces)

    with open(ply_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n_verts}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write(f'element face {n_faces}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        for v, n in zip(vertices, normals):
            f.write(f'{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} '
                    f'{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n')

        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
