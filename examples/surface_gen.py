"""
真实分子表面生成（Open3D + biopython，无 MSMS 依赖）

输出 .ply 与 `transformerdock/utils/data.py:read_ply` 字段兼容：
  vertex: x y z nx ny nz charge hydrophobicity hbond_donor hbond_acceptor
          curvature shape_index aa_polar
  face:   三角面（ball-pivoting 重建，失败则退化为 k-NN 边）

与旧函数对比升级点：
  - 全原子 VdW 表面采样，而非只取 Cα
  - 真实几何法向量（Open3D estimate_normals）
  - 真实曲率 / 形状指数（局部 PCA）
  - 残基级物化特征保留
"""

import os
import contextlib
import sys
import numpy as np
import open3d as o3d
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# FreeSASA for solvent accessibility
try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False
    print("[WARNING] freesasa not installed. SASA features will be zeros. Install: pip install freesasa")

try:
    from scripts.robust_sasa_compute import compute_sasa_robust
except ImportError:
    compute_sasa_robust = None


# ─────────────────────────────────────────────────────────────
# VdW 半径 & 残基物化性质
# ─────────────────────────────────────────────────────────────

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
    'FE': 1.80, 'ZN': 1.39, 'MG': 1.73, 'CA': 2.31, 'NA': 2.27, 'K': 2.75,
}

AA_PROPS = {
    'ALA': (0.62, 0, 0, 0), 'ARG': (-2.53, 5, 2, 1),
    'ASN': (-0.78, 2, 2, 1), 'ASP': (-0.90, 1, 4, 1),
    'CYS': (0.29, 1, 1, 0), 'GLN': (-0.85, 2, 2, 1),
    'GLU': (-0.74, 1, 4, 1), 'GLY': (0.48, 0, 0, 0),
    'HIS': (-0.40, 2, 2, 1), 'ILE': (1.38, 0, 0, 0),
    'LEU': (1.06, 0, 0, 0), 'LYS': (-1.50, 3, 1, 1),
    'MET': (0.64, 0, 1, 0), 'PHE': (1.19, 0, 0, 0),
    'PRO': (0.12, 0, 0, 0), 'SER': (-0.18, 2, 2, 1),
    'THR': (-0.05, 2, 2, 1), 'TRP': (0.81, 1, 1, 0),
    'TYR': (0.26, 2, 2, 1), 'VAL': (1.08, 0, 0, 0),
}

# 残基形式电荷（近似，用于 charge 通道）
RES_CHARGE = {
    'ARG':  1.0, 'LYS':  1.0, 'HIS': 0.5,
    'ASP': -1.0, 'GLU': -1.0,
}

# 残基最大SASA（完全暴露状态，单位Å²）用于归一化
MAX_SASA = {
    'ALA': 129, 'CYS': 167, 'ASP': 193, 'GLU': 223, 'PHE': 240,
    'GLY': 104, 'HIS': 224, 'ILE': 197, 'LYS': 236, 'LEU': 201,
    'MET': 224, 'ASN': 195, 'PRO': 159, 'GLN': 225, 'ARG': 274,
    'SER': 155, 'THR': 172, 'VAL': 174, 'TRP': 285, 'TYR': 263,
}

TWO_LETTER_ELEMENTS = {
    'CL', 'BR', 'NA', 'MG', 'AL', 'SI', 'FE', 'ZN', 'CA', 'MN', 'CO', 'CU',
    'NI', 'CD', 'HG', 'SE', 'LI', 'BE', 'NE', 'AR', 'KR', 'XE',
}


@contextlib.contextmanager
def _suppress_c_stderr(enabled=True):
    """Suppress noisy C-library stderr output such as repeated FreeSASA guesses."""
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


def _guess_pdb_element(atom_name, record_name='ATOM'):
    if record_name == 'ATOM' and atom_name[:1] == ' ':
        for ch in atom_name:
            if ch.isalpha():
                return ch.upper()
    atom = ''.join(ch for ch in atom_name.strip() if ch.isalpha()).upper()
    if not atom:
        return 'C'
    if len(atom) >= 2 and atom[:2] in TWO_LETTER_ELEMENTS:
        return atom[:2].title()
    return atom[0].upper()


def _write_pdb_with_elements(src_path, dst_path):
    """Write a FreeSASA-friendly PDB copy with cols 77-78 populated."""
    with open(src_path, 'r', errors='ignore') as src, open(dst_path, 'w') as dst:
        for line in src:
            if line.startswith(('ATOM  ', 'HETATM')):
                line = line.rstrip('\n')
                if len(line) < 78:
                    line = line.ljust(78)
                element = line[76:78].strip()
                if not element:
                    element = _guess_pdb_element(line[12:16], line[:6].strip())
                    line = line[:76] + element.rjust(2) + line[78:]
                dst.write(line + '\n')
            else:
                dst.write(line)


def _finite_array(values, fill=0.0, lo=None, hi=None):
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    if lo is not None or hi is not None:
        arr = np.clip(arr, -np.inf if lo is None else lo,
                      np.inf if hi is None else hi)
    return arr


# ─────────────────────────────────────────────────────────────
# FreeSASA 计算
# ─────────────────────────────────────────────────────────────

def compute_sasa_features(pdb_path, coords, res_names, quiet=True):
    """
    使用FreeSASA计算原子级SASA，返回每个原子的SASA和归一化rSASA。

    Args:
        pdb_path: PDB文件路径
        coords: 原子坐标 [N, 3]
        res_names: 残基名称列表 [N]

    Returns:
        atom_sasa: 原子SASA值 [N]
        atom_rSASA: 归一化相对SASA [N]
    """
    if compute_sasa_robust is not None:
        atom_sasa, atom_rSASA, _ = compute_sasa_robust(
            pdb_path, coords, res_names, strategy='mean', quiet=quiet
        )
        return _fit_sasa_length(atom_sasa, atom_rSASA, len(coords), res_names)

    if not HAS_FREESASA:
        return _fallback_sasa(res_names)

    try:
        import tempfile
        # 1. 用FreeSASA计算
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
            tmp_pdb = tmp.name
        _write_pdb_with_elements(pdb_path, tmp_pdb)
        with _suppress_c_stderr(quiet):
            structure = freesasa.Structure(tmp_pdb)
            result = freesasa.calc(structure)

        # 2. 提取每个原子的SASA
        n_atoms = structure.nAtoms()
        atom_sasa = np.zeros(n_atoms, dtype=np.float32)
        atom_resnames = []

        for i in range(n_atoms):
            atom_sasa[i] = result.atomArea(i)
            atom_resnames.append(structure.residueName(i).strip())

        # 3. 计算相对SASA (归一化到残基最大暴露面积)
        atom_rSASA = np.array([
            sasa / MAX_SASA.get(resname, 150.0)
            for sasa, resname in zip(atom_sasa, atom_resnames)
        ], dtype=np.float32)

        # 4. 裁剪异常值（有时会超过理论最大值）
        atom_rSASA = np.clip(atom_rSASA, 0, 2.0)

        try:
            os.remove(tmp_pdb)
        except OSError:
            pass
        return _fit_sasa_length(atom_sasa, atom_rSASA, len(coords), res_names)

    except Exception as e:
        try:
            if 'tmp_pdb' in locals() and os.path.exists(tmp_pdb):
                os.remove(tmp_pdb)
        except OSError:
            pass
        if not quiet:
            print(f"[SASA计算失败] {pdb_path}: {e}")
        return _fallback_sasa(res_names)


def _fallback_sasa(res_names, value=0.5):
    n = len(res_names)
    rsasa = np.full(n, value, dtype=np.float32)
    sasa = np.array(
        [MAX_SASA.get(str(r), 150.0) * value for r in res_names],
        dtype=np.float32,
    )
    return sasa, rsasa


def _fit_sasa_length(atom_sasa, atom_rSASA, n_atoms, res_names):
    atom_sasa = _finite_array(atom_sasa, fill=0.0, lo=0.0)
    atom_rSASA = _finite_array(atom_rSASA, fill=0.5, lo=0.0, hi=2.0)
    if len(atom_rSASA) == n_atoms:
        return atom_sasa, atom_rSASA

    fallback_sasa, fallback_rsasa = _fallback_sasa(res_names)
    out_sasa = fallback_sasa.copy()
    out_rsasa = fallback_rsasa.copy()
    m = min(n_atoms, len(atom_rSASA))
    if m > 0:
        out_sasa[:m] = atom_sasa[:m]
        out_rsasa[:m] = atom_rSASA[:m]
    return out_sasa, out_rsasa


# ─────────────────────────────────────────────────────────────
# 几何工具
# ─────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    """在单位球面上均匀采样 n 个点。"""
    i = np.arange(n, dtype=np.float64)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def compute_curvature_and_shape(points, normals, knn=20):
    """
    局部 PCA 估计曲率（最小特征值占比）和形状指数。
    shape_index ∈ [-1, 1]，参考 MaSIF 定义。
    """
    tree = cKDTree(points)
    _, idx = tree.query(points, k=knn + 1)
    idx = idx[:, 1:]  # 去掉自己

    curvatures = np.zeros(len(points))
    shape_idx = np.zeros(len(points))

    for i in range(len(points)):
        neigh = points[idx[i]]
        centered = neigh - points[i]
        # 局部协方差
        cov = centered.T @ centered / len(neigh)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total > 1e-8:
            curvatures[i] = eigvals[0] / total
        # 形状指数：用两个主曲率方向上法向量投影差
        # 近似：对 k 个邻居在切平面内做二次拟合
        n_i = normals[i]
        diff_n = normals[idx[i]] - n_i
        # 将法向量差投影到点差上，估两个主曲率
        proj = np.einsum('ij,ij->i', diff_n, centered)
        dist2 = np.einsum('ij,ij->i', centered, centered) + 1e-8
        kappa = proj / dist2
        k1 = kappa.max()
        k2 = kappa.min()
        if abs(k1) + abs(k2) > 1e-6:
            shape_idx[i] = (2 / np.pi) * np.arctan2(k1 + k2, k1 - k2 + 1e-8)

    return (
        _finite_array(curvatures, fill=0.0, lo=0.0, hi=1.0),
        _finite_array(shape_idx, fill=0.0, lo=-1.0, hi=1.0),
    )


# ─────────────────────────────────────────────────────────────
# PDB → 原子列表
# ─────────────────────────────────────────────────────────────

def parse_atoms(pdb_path):
    """
    返回 (coords[N,3], radii[N], res_names[N], res_ids[N])
    忽略氢原子和非标准溶剂/离子（HOH、缓冲剂等）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('_', pdb_path)

    coords, radii, res_names, res_ids = [], [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                rn = res.get_resname().strip()
                if rn in ('HOH', 'WAT', 'TIP3'):
                    continue
                rid = (chain.id, res.get_id()[1])
                for atom in res:
                    elem = atom.element.strip().upper() if atom.element else atom.get_name().strip()[0]
                    if elem == 'H':
                        continue
                    coords.append(atom.coord)
                    radii.append(VDW_RADII.get(elem, 1.70))
                    res_names.append(rn)
                    res_ids.append(rid)
        break  # 只取第一个 MODEL

    return (
        np.array(coords, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(res_names),
        res_ids,
    )


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def pdb_to_surface_ply(pdb_path, ply_path,
                       probe_radius=1.4,
                       points_per_atom=60,
                       voxel_size=2.0,
                       knn=20,
                       verbose=False):
    """
    从 PDB 生成真实分子溶剂可及表面 (.ply)。
    兼容 `transformerdock/utils/data.py:read_ply` 的字段约定。
    返回 True/False。
    """
    try:
        coords, radii, res_names, res_ids = parse_atoms(pdb_path)
    except Exception as e:
        if verbose:
            print(f'[parse_atoms 失败] {pdb_path}: {e}')
        return False

    if len(coords) < 10:
        return False

    # 1. 每个原子的 VdW 球面采样，半径扩展为 VdW + probe（SAS 近似）
    sphere = fibonacci_sphere(points_per_atom)
    sas_radii = radii + probe_radius
    all_pts = coords[:, None, :] + sphere[None, :, :] * sas_radii[:, None, None]
    all_pts = all_pts.reshape(-1, 3)
    atom_idx = np.repeat(np.arange(len(coords)), points_per_atom)

    # 2. 丢弃落在其他原子 SAS 球内的点（近似溶剂可及表面）
    tree = cKDTree(coords)
    max_r = sas_radii.max()
    nn_lists = tree.query_ball_point(all_pts, r=max_r)

    keep = np.ones(len(all_pts), dtype=bool)
    for i, nbrs in enumerate(nn_lists):
        own = atom_idx[i]
        for j in nbrs:
            if j == own:
                continue
            d2 = ((all_pts[i] - coords[j]) ** 2).sum()
            if d2 < (sas_radii[j] - 1e-3) ** 2:
                keep[i] = False
                break
    pts = all_pts[keep]
    pt_atom = atom_idx[keep]

    if len(pts) < 50:
        if verbose:
            print(f'[表面点太少] {pdb_path}: {len(pts)}')
        return False

    # 3. Open3D 点云 + voxel 下采样 + 法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # 记录下采样前每个点对应的原子索引，下采样后要重算最近原子
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    verts = np.asarray(pcd.points, dtype=np.float32)
    normals = _finite_array(np.asarray(pcd.normals, dtype=np.float32),
                            fill=0.0, lo=-1.0, hi=1.0)
    n_v = len(verts)
    if n_v < 3:
        return False

    # 4. 计算SASA特征（原子级）
    atom_sasa, atom_rSASA = compute_sasa_features(pdb_path, coords, res_names)

    # 5. 每个表面点 → 最近原子 → 所属残基 → 残基特征 + SASA
    d, nn_idx = tree.query(verts, k=1)
    feat_charge = np.zeros(n_v, dtype=np.float32)
    feat_hphob  = np.zeros(n_v, dtype=np.float32)
    feat_hbd    = np.zeros(n_v, dtype=np.float32)
    feat_hba    = np.zeros(n_v, dtype=np.float32)
    feat_polar  = np.zeros(n_v, dtype=np.float32)
    feat_sasa   = np.zeros(n_v, dtype=np.float32)
    feat_rSASA  = np.zeros(n_v, dtype=np.float32)

    for i in range(n_v):
        nearest_atom = nn_idx[i]
        rn = res_names[nearest_atom]
        props = AA_PROPS.get(rn, (0.0, 0, 0, 0))
        feat_hphob[i] = props[0]
        feat_hbd[i]   = float(props[1])
        feat_hba[i]   = float(props[2])
        feat_polar[i] = float(props[3])
        feat_charge[i] = RES_CHARGE.get(rn, 0.0)

        # SASA特征：从最近原子映射到表面顶点
        if nearest_atom < len(atom_sasa):
            feat_sasa[i] = atom_sasa[nearest_atom]
            feat_rSASA[i] = atom_rSASA[nearest_atom]

    # 6. 曲率 + 形状指数
    curvature, shape_index = compute_curvature_and_shape(verts, normals, knn=knn)
    feat_charge = _finite_array(feat_charge, fill=0.0, lo=-5.0, hi=5.0)
    feat_hphob = _finite_array(feat_hphob, fill=0.0, lo=-5.0, hi=5.0)
    feat_hbd = _finite_array(feat_hbd, fill=0.0, lo=0.0, hi=8.0)
    feat_hba = _finite_array(feat_hba, fill=0.0, lo=0.0, hi=8.0)
    feat_polar = _finite_array(feat_polar, fill=0.0, lo=0.0, hi=1.0)
    feat_rSASA = _finite_array(feat_rSASA, fill=0.5, lo=0.0, hi=2.0)

    # 7. 三角化（ball-pivoting），失败则用 k-NN 边
    faces = None
    try:
        dists = pcd.compute_nearest_neighbor_distance()
        avg_d = float(np.mean(dists))
        radii_bp = o3d.utility.DoubleVector([avg_d * r for r in (1.0, 1.5, 2.0, 2.5)])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_bp)
        if len(mesh.triangles) > 0:
            faces = np.asarray(mesh.triangles, dtype=np.int32)
    except Exception as e:
        if verbose:
            print(f'[ball-pivoting 失败] {e}')

    # 8. 写 .ply（包含SASA特征）
    return _write_ply(
        ply_path, verts, normals,
        feat_charge, feat_hphob, feat_hbd, feat_hba,
        curvature, shape_index, feat_polar,
        feat_rSASA,  # 新增：相对SASA
        faces
    )


def _write_ply(path, verts, normals,
               charge, hphob, hbd, hba,
               curvature, shape_index, polar,
               rSASA,
               faces):
    n_v = len(verts)
    n_f = 0 if faces is None else len(faces)
    try:
        with open(path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {n_v}\n')
            for name in ('x', 'y', 'z', 'nx', 'ny', 'nz',
                         'charge', 'hydrophobicity',
                         'hbond_donor', 'hbond_acceptor',
                         'curvature', 'shape_index', 'aa_polar',
                         'rSASA'):  # 新增
                f.write(f'property float {name}\n')
            f.write(f'element face {n_f}\n')
            f.write('property list uchar int vertex_indices\n')
            f.write('end_header\n')

            for i in range(n_v):
                f.write(
                    f'{verts[i,0]:.3f} {verts[i,1]:.3f} {verts[i,2]:.3f} '
                    f'{normals[i,0]:.3f} {normals[i,1]:.3f} {normals[i,2]:.3f} '
                    f'{charge[i]:.2f} {hphob[i]:.3f} '
                    f'{hbd[i]:.1f} {hba[i]:.1f} '
                    f'{curvature[i]:.4f} {shape_index[i]:.3f} {polar[i]:.1f} '
                    f'{rSASA[i]:.4f}\n'  # 新增
                )
            if faces is not None:
                for a, b, c in faces:
                    f.write(f'3 {a} {b} {c}\n')
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# 命令行单样本测试
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument('--pdb', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--voxel_size', type=float, default=2.0)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    t0 = time.time()
    ok = pdb_to_surface_ply(args.pdb, args.out,
                            voxel_size=args.voxel_size,
                            verbose=args.verbose)
    dt = time.time() - t0
    print(f'{"OK" if ok else "FAIL"}  {args.pdb} -> {args.out}  ({dt:.2f}s)')
