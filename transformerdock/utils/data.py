"""
数据加载和预处理工具。
支持从 .ply 表面网格文件构建 PyG Data 对象。
节点特征（11维）：法向量(3) + 静电势(1) + 疏水性(1) + 氢键供体/受体(2)
              + 曲率(1) + 形状指数(1) + 氨基酸极性(1) + rSASA(1)
"""

import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

try:
    from plyfile import PlyData
    HAS_PLY = True
except ImportError:
    HAS_PLY = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


FEATURE_NAMES = [
    'nx', 'ny', 'nz',
    'charge', 'hydrophobicity',
    'hbond_donor', 'hbond_acceptor',
    'curvature', 'shape_index', 'aa_polar', 'rSASA',
]

PAIR_AWARE_FEATURE_NAMES = [
    'min_partner_dist_norm',
    'contact_density_5A',
    'contact_density_8A',
    'clash_depth',
    'normal_facing',
    'normal_complementarity',
    'electrostatic_partner',
    'hydrophobic_partner',
]

PAIR_AWARE_IN_CHANNELS = len(FEATURE_NAMES) + len(PAIR_AWARE_FEATURE_NAMES)

# Conservative clamps for the current surface_gen.py feature contract. The main
# goal is to prevent a single malformed PLY/SASA value from poisoning BatchNorm.
FEATURE_CLAMPS = [
    (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
    (-5.0, 5.0), (-5.0, 5.0),
    (0.0, 8.0), (0.0, 8.0),
    (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 2.0),
]

PAIR_AWARE_FEATURE_CLAMPS = [
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 2.0),
    (0.0, 5.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
]


def _vertex_names(verts):
    return set(verts.data.dtype.names or [])


def _field_or_zeros(verts, field, n, dtype=np.float32):
    names = _vertex_names(verts)
    if field in names:
        values = np.asarray(verts[field], dtype=dtype).reshape(n, 1)
    else:
        values = np.zeros((n, 1), dtype=dtype)
    return values


def _sanitize_feature_array(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        x = x.reshape(len(x), -1)
    n_clamp = min(x.shape[1], len(FEATURE_CLAMPS))
    for i in range(n_clamp):
        lo, hi = FEATURE_CLAMPS[i]
        x[:, i] = np.clip(x[:, i], lo, hi)
    return x


def _sanitize_pos_array(pos):
    pos = np.asarray(pos, dtype=np.float32)
    return np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)


def _knn_edge_index(pos, k=8):
    """Build a directed kNN edge_index without relying on PyG native extensions."""
    n = int(pos.size(0))
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    k = min(k, n - 1)
    pos_np = pos.detach().cpu().numpy()
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(pos_np).query(pos_np, k=k + 1)
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[:, None]
        dst = idx[:, 1:].reshape(-1)
        src = np.repeat(np.arange(n), k)
    except Exception:
        dist = torch.cdist(pos, pos)
        idx = dist.topk(k + 1, largest=False).indices[:, 1:]
        src = torch.arange(n, device=pos.device).repeat_interleave(k)
        dst = idx.reshape(-1)
        return torch.stack([src.cpu(), dst.cpu()], dim=0).long()
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


def _ensure_edges(data, k=8):
    if getattr(data, 'edge_index', None) is None or data.edge_index.numel() == 0:
        data.edge_index = _knn_edge_index(data.pos, k=k)
    return data


def match_feature_dim(data, in_channels):
    """Trim or zero-pad node features to match a checkpoint/model input width."""
    if in_channels is None:
        return data
    cur = int(data.x.size(1))
    if cur == in_channels:
        return data
    if cur > in_channels:
        data.x = data.x[:, :in_channels]
    else:
        pad = data.x.new_zeros((data.x.size(0), in_channels - cur))
        data.x = torch.cat([data.x, pad], dim=1)
    return data


def _clamp_columns(x, clamps):
    for i, (lo, hi) in enumerate(clamps):
        x[:, i] = torch.clamp(x[:, i], lo, hi)
    return x


def _normalized_normals(x):
    if x.size(1) < 3:
        return x.new_zeros((x.size(0), 3))
    normals = x[:, :3]
    return normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-6)


def _column_or_zeros(x, idx):
    if x.size(1) <= idx:
        return x.new_zeros((x.size(0),))
    return x[:, idx]


@torch.no_grad()
def _partner_features(source, partner, chunk_size=1024):
    """Pose-aware per-node features from source surface to partner surface."""
    n_src = int(source.pos.size(0))
    n_partner = int(partner.pos.size(0))
    if n_src == 0 or n_partner == 0:
        return source.x.new_zeros((n_src, len(PAIR_AWARE_FEATURE_NAMES)))

    src_pos = source.pos.float()
    partner_pos = partner.pos.float()
    src_norm = _normalized_normals(source.x.float())
    partner_norm = _normalized_normals(partner.x.float())
    src_charge = _column_or_zeros(source.x.float(), 3)
    partner_charge = _column_or_zeros(partner.x.float(), 3)
    src_hydro = _column_or_zeros(source.x.float(), 4)
    partner_hydro = _column_or_zeros(partner.x.float(), 4)

    out = source.x.new_zeros((n_src, len(PAIR_AWARE_FEATURE_NAMES)))
    log32 = math.log1p(32.0)
    log64 = math.log1p(64.0)
    log8 = math.log1p(8.0)

    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)
        dist = torch.cdist(src_pos[start:end], partner_pos)
        min_dist, nearest_idx = dist.min(dim=1)

        nearest_pos = partner_pos[nearest_idx]
        direction = nearest_pos - src_pos[start:end]
        unit_direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        normal_facing = (src_norm[start:end] * unit_direction).sum(dim=1).clamp(-1.0, 1.0)
        normal_complementarity = (
            -(src_norm[start:end] * partner_norm[nearest_idx]).sum(dim=1)
        ).clamp(-1.0, 1.0)

        w5 = torch.exp(-((dist / 3.0) ** 2)) * (dist < 5.0).float()
        w8 = torch.exp(-((dist / 4.0) ** 2)) * (dist < 8.0).float()
        density5 = torch.log1p(w5.sum(dim=1)) / log32
        density8 = torch.log1p(w8.sum(dim=1)) / log64

        clash_depth = torch.log1p(torch.relu(2.0 - dist).pow(2).sum(dim=1)) / log8

        denom8 = w8.sum(dim=1).clamp(min=1e-6)
        avg_partner_charge = (w8 * partner_charge.view(1, -1)).sum(dim=1) / denom8
        avg_partner_hydro = (w8 * partner_hydro.view(1, -1)).sum(dim=1) / denom8
        electrostatic_partner = (-src_charge[start:end] * avg_partner_charge / 5.0).clamp(-1.0, 1.0)
        hydrophobic_partner = (src_hydro[start:end] * avg_partner_hydro / 5.0).clamp(-1.0, 1.0)

        feats = torch.stack([
            (min_dist / 20.0).clamp(0.0, 1.0),
            density5,
            density8,
            clash_depth,
            normal_facing,
            normal_complementarity,
            electrostatic_partner,
            hydrophobic_partner,
        ], dim=1)
        out[start:end] = _clamp_columns(
            torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0),
            PAIR_AWARE_FEATURE_CLAMPS,
        ).to(out.dtype)

    return out


def add_pair_aware_features(rec, lig):
    """
    Append pose-aware interface physics to receptor and ligand node features.

    The original .ply files stay 11-dimensional. These 8 extra channels are
    computed from the current receptor/ligand pose, so they must be added after
    both chains have been loaded.
    """
    base_dim = len(FEATURE_NAMES)
    pair_dim = len(PAIR_AWARE_FEATURE_NAMES)
    if rec.x.size(1) >= base_dim + pair_dim and lig.x.size(1) >= base_dim + pair_dim:
        return rec, lig
    rec.x = torch.cat([rec.x, _partner_features(rec, lig)], dim=1)
    lig.x = torch.cat([lig.x, _partner_features(lig, rec)], dim=1)
    return rec, lig


def maybe_add_pair_aware_features(rec, lig, in_channels):
    if in_channels is not None and int(in_channels) > len(FEATURE_NAMES):
        rec, lig = add_pair_aware_features(rec, lig)
    return rec, lig


# ─────────────────────────────────────────────────────────────
# .ply 表面网格读取
# ─────────────────────────────────────────────────────────────

def read_ply(ply_path):
    """
    读取 .ply 文件，返回 PyG Data 对象。
    节点 = 网格顶点，边 = 三角面片的边（FaceToEdge 变换）。
    节点特征：顶点法向量(3) + 静电势(1) + 疏水性(1) + 氢键供体(1) + 氢键受体(1)
              + 曲率(1) + 形状指数(1) + 氨基酸极性(1) + rSASA(1) = 11维
    """
    assert HAS_PLY, "请安装 plyfile: pip install plyfile"
    ply = PlyData.read(ply_path)
    verts = ply['vertex']
    names = _vertex_names(verts)
    n_verts = len(verts)

    # 坐标
    pos_np = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
    pos = torch.tensor(_sanitize_pos_array(pos_np), dtype=torch.float)

    # 节点特征（按可用字段组装）
    feat_list = []

    # 法向量（3维）
    if {'nx', 'ny', 'nz'}.issubset(names):
        normals = np.stack([verts['nx'], verts['ny'], verts['nz']], axis=1)
    else:
        normals = np.zeros((n_verts, 3))
    feat_list.append(normals)

    # 静电势（1维）
    feat_list.append(_field_or_zeros(verts, 'charge', n_verts))

    # 疏水性（1维）
    feat_list.append(_field_or_zeros(verts, 'hydrophobicity', n_verts))

    # 氢键供体（1维）
    feat_list.append(_field_or_zeros(verts, 'hbond_donor', n_verts))

    # 氢键受体（1维）
    feat_list.append(_field_or_zeros(verts, 'hbond_acceptor', n_verts))

    # 曲率（1维）
    feat_list.append(_field_or_zeros(verts, 'curvature', n_verts))

    # 形状指数（1维）
    feat_list.append(_field_or_zeros(verts, 'shape_index', n_verts))

    # 氨基酸极性（1维）
    if 'aa_polar' in names:
        feat_list.append(verts['aa_polar'].reshape(-1, 1).astype(np.float32))
    elif 'aa_type' in names:
        aa = verts['aa_type'].reshape(-1, 1).astype(np.float32)
        feat_list.append((aa % 2).astype(np.float32))
    else:
        feat_list.append(np.zeros((n_verts, 1), dtype=np.float32))

    # 相对SASA（1维）- 新增
    if 'rSASA' in names:
        feat_list.append(verts['rSASA'].reshape(-1, 1).astype(np.float32))
    elif 'sasa' in names:
        # 如果只有绝对SASA，做简单归一化
        sasa = verts['sasa'].reshape(-1, 1).astype(np.float32)
        sasa = np.nan_to_num(sasa, nan=0.0, posinf=0.0, neginf=0.0)
        sasa_norm = sasa / (np.nanmax(sasa) + 1e-6)
        feat_list.append(sasa_norm)
    else:
        feat_list.append(np.zeros((n_verts, 1), dtype=np.float32))

    x = torch.tensor(_sanitize_feature_array(np.concatenate(feat_list, axis=1)),
                     dtype=torch.float)

    # 面片 → 边
    faces = None
    if 'face' in ply and len(ply['face']) > 0:
        face_data = ply['face']['vertex_indices']
        parsed_faces = []
        for f in face_data:
            arr = np.asarray(f, dtype=np.int64)
            if arr.size == 3 and arr.min() >= 0 and arr.max() < n_verts:
                parsed_faces.append(arr)
        if parsed_faces:
            faces = torch.tensor(np.stack(parsed_faces, axis=0).T, dtype=torch.long)

    data = Data(x=x, pos=pos, face=faces)
    if faces is not None:
        data = FaceToEdge(remove_faces=True)(data)
    data = _ensure_edges(data)

    return data


# ─────────────────────────────────────────────────────────────
# 节点截断（防止大蛋白质 OOM）
# ─────────────────────────────────────────────────────────────

def _subsample_nodes(data, max_nodes):
    """随机采样节点到 max_nodes 以内，同时更新 edge_index 和 face。"""
    n = data.x.size(0)
    if n <= max_nodes:
        return data
    perm = torch.randperm(n)[:max_nodes].sort().values
    data.x = data.x[perm]
    data.pos = data.pos[perm]
    if data.edge_index is not None:
        mask = torch.isin(data.edge_index[0], perm) & torch.isin(data.edge_index[1], perm)
        edge_index = data.edge_index[:, mask]
        node_map = torch.zeros(n, dtype=torch.long)
        node_map[perm] = torch.arange(max_nodes)
        data.edge_index = node_map[edge_index]
    if hasattr(data, 'face') and data.face is not None:
        face = data.face
        mask = torch.isin(face[0], perm) & torch.isin(face[1], perm) & torch.isin(face[2], perm)
        face = face[:, mask]
        node_map = torch.zeros(n, dtype=torch.long)
        node_map[perm] = torch.arange(max_nodes)
        data.face = node_map[face]
    data = _ensure_edges(data)
    return data


# ─────────────────────────────────────────────────────────────
# PPI 复合物数据集
# ─────────────────────────────────────────────────────────────

class PPI_Dataset(Dataset):
    """
    蛋白质-蛋白质对接数据集。
    每个样本 = (receptor_data, ligand_data, label, name)

    支持两种文件命名规范：
        标准风格：<name>_receptor.ply + <name>_ligand.ply
        旧版风格：  <name>_receptor.ply + <name>_binder.ply

    pairs.csv 格式（列名 name 或 pdb_id 均可，label 列可选）：
        name,label
        1abc,1
        2xyz,1
    """
    def __init__(self, data_dir, pairs_csv=None, transform=None, max_nodes=1500,
                 in_channels=11):
        super().__init__()
        self.data_dir = data_dir
        self.max_nodes = max_nodes
        self.in_channels = in_channels
        self.pairs = self._load_pairs(pairs_csv)

    def _load_pairs(self, csv_path):
        if csv_path is None:
            csv_path = os.path.join(self.data_dir, 'pairs.csv')
        if not os.path.exists(csv_path):
            return []
        import csv as _csv
        pairs = []
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                # 兼容 name / pdb_id 两种列名
                name = row.get('name') or row.get('pdb_id', '')
                label = int(float(row.get('label', 1)))
                if name:
                    pairs.append((name, label))
        return pairs

    def _find_ply(self, name, role):
        """
        查找 .ply 文件，兼容多种命名规范。
        role: 'receptor' 或 'ligand'
        """
        candidates = [f'{name}_{role}.ply']
        if role == 'ligand':
            candidates.append(f'{name}_binder.ply')
        elif role == 'receptor':
            candidates.append(f'{name}_receptor.ply')

        for fname in candidates:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'找不到 {name} 的 {role} .ply 文件（尝试了: {candidates}）'
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 支持切片，返回子数据集
        if isinstance(idx, slice):
            sub = PPI_Dataset.__new__(PPI_Dataset)
            sub.data_dir = self.data_dir
            sub.max_nodes = self.max_nodes
            sub.in_channels = self.in_channels
            sub.pairs = self.pairs[idx]
            return sub
        name, label = self.pairs[idx]
        rec_path = self._find_ply(name, 'receptor')
        lig_path = self._find_ply(name, 'ligand')
        rec = read_ply(rec_path)
        lig = read_ply(lig_path)
        rec, lig = maybe_add_pair_aware_features(rec, lig, self.in_channels)
        rec = match_feature_dim(rec, self.in_channels)
        lig = match_feature_dim(lig, self.in_channels)
        if self.max_nodes:
            rec = _subsample_nodes(rec, self.max_nodes)
            lig = _subsample_nodes(lig, self.max_nodes)
        return rec, lig, torch.tensor([label], dtype=torch.float), name


# ─────────────────────────────────────────────────────────────
# 单对复合物准备（推理用）
# ─────────────────────────────────────────────────────────────

def prepare_complex(rec_ply_path, lig_ply_path, in_channels=11):
    """
    从两个 .ply 文件准备一个复合物数据对。
    返回 (rec_data, lig_data)，可直接传入 DeepDock_PPI.forward()。
    """
    rec = read_ply(rec_ply_path)
    lig = read_ply(lig_ply_path)
    rec, lig = maybe_add_pair_aware_features(rec, lig, in_channels)
    rec = match_feature_dim(rec, in_channels)
    lig = match_feature_dim(lig, in_channels)
    # 添加 batch 索引（单样本 batch=0）
    rec.batch = torch.zeros(rec.x.size(0), dtype=torch.long)
    lig.batch = torch.zeros(lig.x.size(0), dtype=torch.long)
    return rec, lig


# ─────────────────────────────────────────────────────────────
# DataLoader collate 函数
# ─────────────────────────────────────────────────────────────

def ppi_collate(batch):
    """
    自定义 collate，处理 (rec, lig, label, pdb_id) 四元组。
    rec/lig 用 PyG 的 Batch 合并，label 堆叠，pdb_id 保持列表。
    """
    from torch_geometric.data import Batch
    recs, ligs, labels, pdb_ids = zip(*batch)
    return (
        Batch.from_data_list(list(recs)),
        Batch.from_data_list(list(ligs)),
        torch.stack(labels),
        list(pdb_ids),
    )
