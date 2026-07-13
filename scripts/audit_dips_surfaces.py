#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit DIPS surface pairs for missing files, bad PLY values, and no-interface samples."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformerdock.utils.data import FEATURE_NAMES, read_ply


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')
EXPECTED_VERTEX_FIELDS = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'charge', 'hydrophobicity',
    'hbond_donor', 'hbond_acceptor',
    'curvature', 'shape_index',
    'aa_polar', 'rSASA',
]
FIELDNAMES = [
    'index', 'name', 'status', 'severity', 'message',
    'rec_vertices', 'lig_vertices', 'rec_dim', 'lig_dim',
    'rec_rsasa_nonzero_pct', 'lig_rsasa_nonzero_pct',
    'min_distance', 'interface_pairs',
]


def parse_args():
    parser = argparse.ArgumentParser(description='全量审计 DIPS surfaces 是否有坏样本')
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=DEFAULT_DATA_DIR,
        help='DIPS surfaces 目录，默认读取 DIPS_SURFACES 或 /root/autodl-tmp/dips_with_sasa_full',
    )
    parser.add_argument('--pairs_csv', default=None, help='默认读取 data_dir/pairs.csv')
    parser.add_argument('--out', default='results/dips_surface_audit.csv')
    parser.add_argument('--limit', type=int, default=None, help='只检查前 N 个样本')
    parser.add_argument('--samples', nargs='*', default=None, help='只检查指定样本名，如 1yk0_A_B 1u0c_A_B')
    parser.add_argument('--dist_threshold', type=float, default=10.0)
    parser.add_argument('--skip_interface', action='store_true', help='跳过 receptor/ligand 近距离 interface 检查')
    parser.add_argument('--progress_every', type=int, default=500)
    return parser.parse_args()


def load_pairs(path):
    pairs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get('name') or row.get('pdb_id') or '').strip()
            if name:
                pairs.append(name)
    return pairs


def _ply_path(data_dir, name, role):
    candidates = [Path(data_dir) / f'{name}_{role}.ply']
    if role == 'ligand':
        candidates.append(Path(data_dir) / f'{name}_binder.ply')
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _finite_stats(values):
    arr = np.asarray(values)
    return bool(np.isfinite(arr).all())


def inspect_raw_ply(path):
    ply = PlyData.read(str(path))
    verts = ply['vertex']
    names = set(verts.data.dtype.names or [])
    missing = [field for field in EXPECTED_VERTEX_FIELDS if field not in names]
    if len(verts) == 0:
        return {'ok': False, 'message': 'empty vertex table', 'n': 0}
    bad_fields = []
    for field in names:
        values = np.asarray(verts[field])
        if values.dtype.kind in {'f', 'i', 'u'} and not _finite_stats(values):
            bad_fields.append(field)
    if missing or bad_fields:
        msg = []
        if missing:
            msg.append('missing=' + ','.join(missing))
        if bad_fields:
            msg.append('nonfinite=' + ','.join(bad_fields))
        return {'ok': False, 'message': '; '.join(msg), 'n': len(verts)}
    return {'ok': True, 'message': '', 'n': len(verts)}


def rsasa_nonzero_pct(data):
    try:
        idx = FEATURE_NAMES.index('rSASA')
    except ValueError:
        return 0.0
    values = data.x[:, idx]
    return float((values > 0).sum().item() * 100.0 / max(1, values.numel()))


def interface_stats(rec_pos, lig_pos, threshold):
    rec_np = rec_pos.detach().cpu().numpy()
    lig_np = lig_pos.detach().cpu().numpy()
    if len(rec_np) == 0 or len(lig_np) == 0:
        return float('inf'), 0
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(lig_np)
        distances, _ = tree.query(rec_np, k=1)
        min_dist = float(np.min(distances))
        pairs = tree.query_ball_point(rec_np, r=threshold, return_length=True)
        return min_dist, int(np.asarray(pairs).sum())
    except Exception:
        min_dist = float('inf')
        pairs = 0
        chunk = 256
        lig_t = lig_pos.detach().cpu()
        for i in range(0, rec_pos.size(0), chunk):
            dist = torch.cdist(rec_pos[i:i + chunk].detach().cpu(), lig_t)
            min_dist = min(min_dist, float(dist.min().item()))
            pairs += int((dist <= threshold).sum().item())
        return min_dist, pairs


def issue_row(index, name, status, severity, message, **extra):
    row = {
        'index': index,
        'name': name,
        'status': status,
        'severity': severity,
        'message': message,
    }
    for field in FIELDNAMES:
        row.setdefault(field, extra.get(field, ''))
    return row


def audit_one(index, name, data_dir, threshold, check_interface):
    rec_path = _ply_path(data_dir, name, 'receptor')
    lig_path = _ply_path(data_dir, name, 'ligand')
    if not rec_path.exists() or not lig_path.exists():
        missing = []
        if not rec_path.exists():
            missing.append(str(rec_path))
        if not lig_path.exists():
            missing.append(str(lig_path))
        return issue_row(index, name, 'missing_file', 'error', '; '.join(missing))

    raw_rec = inspect_raw_ply(rec_path)
    raw_lig = inspect_raw_ply(lig_path)
    if not raw_rec['ok'] or not raw_lig['ok']:
        return issue_row(
            index, name, 'bad_raw_ply', 'error',
            f"receptor: {raw_rec['message']} | ligand: {raw_lig['message']}",
            rec_vertices=raw_rec['n'], lig_vertices=raw_lig['n'],
        )

    rec = read_ply(str(rec_path))
    lig = read_ply(str(lig_path))
    rec_dim = int(rec.x.size(1))
    lig_dim = int(lig.x.size(1))
    base = {
        'rec_vertices': int(rec.x.size(0)),
        'lig_vertices': int(lig.x.size(0)),
        'rec_dim': rec_dim,
        'lig_dim': lig_dim,
        'rec_rsasa_nonzero_pct': f'{rsasa_nonzero_pct(rec):.2f}',
        'lig_rsasa_nonzero_pct': f'{rsasa_nonzero_pct(lig):.2f}',
    }

    if rec_dim != len(FEATURE_NAMES) or lig_dim != len(FEATURE_NAMES):
        return issue_row(index, name, 'bad_feature_dim', 'error', 'model feature dim != 11', **base)
    if not torch.isfinite(rec.x).all() or not torch.isfinite(lig.x).all():
        return issue_row(index, name, 'nonfinite_features', 'error', 'read_ply produced NaN/Inf features', **base)
    if not torch.isfinite(rec.pos).all() or not torch.isfinite(lig.pos).all():
        return issue_row(index, name, 'nonfinite_positions', 'error', 'read_ply produced NaN/Inf positions', **base)

    if check_interface:
        min_dist, pairs = interface_stats(rec.pos, lig.pos, threshold)
        base['min_distance'] = f'{min_dist:.4f}'
        base['interface_pairs'] = pairs
        if pairs == 0:
            return issue_row(index, name, 'no_interface', 'warning',
                             f'no receptor-ligand surface pairs within {threshold}A', **base)

    rec_rsasa = float(base['rec_rsasa_nonzero_pct'])
    lig_rsasa = float(base['lig_rsasa_nonzero_pct'])
    if rec_rsasa == 0.0 or lig_rsasa == 0.0:
        return issue_row(index, name, 'zero_rsasa', 'warning', 'all rSASA values are zero', **base)

    return issue_row(index, name, 'ok', 'ok', '', **base)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else data_dir / 'pairs.csv'
    if not pairs_csv.exists():
        print(f"[错误] 未找到 pairs.csv: {pairs_csv}", file=sys.stderr)
        return 1

    names = load_pairs(pairs_csv)
    if args.samples:
        wanted = set(args.samples)
        names = [name for name in names if name in wanted]
    if args.limit:
        names = names[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    n_issue = 0
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for idx, name in enumerate(names, 1):
            try:
                row = audit_one(idx, name, data_dir, args.dist_threshold, not args.skip_interface)
            except Exception as exc:
                row = issue_row(idx, name, 'exception', 'error', str(exc))
            counts[row['status']] = counts.get(row['status'], 0) + 1
            if row['status'] != 'ok':
                n_issue += 1
                writer.writerow(row)
                f.flush()
            if args.progress_every and idx % args.progress_every == 0:
                print(f"[进度] {idx}/{len(names)}  issues={n_issue}  counts={counts}")

    print("=" * 60)
    print("DIPS surfaces 审计完成")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"样本数: {len(names)}")
    print(f"问题数: {n_issue}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"问题明细 CSV: {out_path}")
    print("=" * 60)
    return 1 if any(k not in {'ok', 'zero_rsasa'} for k in counts) else 0


if __name__ == '__main__':
    raise SystemExit(main())
