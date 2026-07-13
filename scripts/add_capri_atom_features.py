#!/usr/bin/env python3
"""Add atom-level receptor-ligand interface features to a CAPRI detail CSV."""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


ATOM_FIELDS = [
    'atom_contact',
    'atom_clash',
    'atom_hbond',
    'atom_hydrophobic',
    'atom_unsatisfied',
    'atom_interface_atoms',
    'atom_score',
]

POLAR = {'N', 'O', 'S'}
HYDROPHOBIC = {'C', 'S'}


def to_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def split_models(pdb_path):
    models = {}
    current_model = None
    current_chains = {}
    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()
            if rec == 'MODEL':
                current_model = int(line.split()[1])
                current_chains = {}
            elif rec == 'ENDMDL':
                if current_model is not None:
                    models[current_model] = current_chains
                current_model = None
            elif rec in ('ATOM', 'HETATM') and current_model is not None and len(line) > 54:
                chain = line[21]
                current_chains.setdefault(chain, []).append(line)
    if models:
        return models

    chains = {}
    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ('ATOM', 'HETATM') and len(line) > 54:
                chains.setdefault(line[21], []).append(line)
    return {0: chains} if chains else {}


def element_from_line(line):
    elem = line[76:78].strip().upper() if len(line) >= 78 else ''
    if elem:
        return elem[0]
    name = line[12:16].strip()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return ''


def parse_atoms(lines):
    coords = []
    elems = []
    for line in lines:
        elem = element_from_line(line)
        if elem == 'H' or not elem:
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        coords.append(xyz)
        elems.append(elem)
    if not coords:
        return np.zeros((0, 3), dtype=np.float32), np.asarray([], dtype='<U1')
    return np.asarray(coords, dtype=np.float32), np.asarray(elems, dtype='<U1')


def pick_rec_lig_chains(chains):
    chain_ids = list(chains.keys())
    if len(chain_ids) < 2:
        return None, None
    rec = 'A' if 'A' in chains else chain_ids[0]
    lig = 'B' if 'B' in chains else chain_ids[1]
    return rec, lig


def contact_pairs(rec_xyz, lig_xyz, cutoff):
    if len(rec_xyz) == 0 or len(lig_xyz) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(lig_xyz)
        pairs = tree.query_ball_point(rec_xyz, cutoff)
        rec_idx = []
        lig_idx = []
        for i, js in enumerate(pairs):
            for j in js:
                rec_idx.append(i)
                lig_idx.append(j)
        if not rec_idx:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
        rec_idx = np.asarray(rec_idx, dtype=np.int64)
        lig_idx = np.asarray(lig_idx, dtype=np.int64)
        dist = np.linalg.norm(rec_xyz[rec_idx] - lig_xyz[lig_idx], axis=1)
        return rec_idx, lig_idx, dist
    except Exception:
        rec_idx = []
        lig_idx = []
        dist_all = []
        for start in range(0, len(rec_xyz), 256):
            block = rec_xyz[start:start + 256]
            d = np.linalg.norm(block[:, None, :] - lig_xyz[None, :, :], axis=2)
            ii, jj = np.where(d <= cutoff)
            rec_idx.extend((ii + start).tolist())
            lig_idx.extend(jj.tolist())
            dist_all.extend(d[ii, jj].tolist())
        return (
            np.asarray(rec_idx, dtype=np.int64),
            np.asarray(lig_idx, dtype=np.int64),
            np.asarray(dist_all, dtype=np.float32),
        )


def atom_features(rec_lines, lig_lines):
    rec_xyz, rec_elem = parse_atoms(rec_lines)
    lig_xyz, lig_elem = parse_atoms(lig_lines)
    rec_i, lig_i, dist = contact_pairs(rec_xyz, lig_xyz, 5.0)
    if len(dist) == 0:
        return {key: 0 for key in ATOM_FIELDS if key != 'atom_score'}

    rec_e = rec_elem[rec_i]
    lig_e = lig_elem[lig_i]
    polar_pair = np.isin(rec_e, list(POLAR)) & np.isin(lig_e, list(POLAR))
    hyd_pair = np.isin(rec_e, list(HYDROPHOBIC)) & np.isin(lig_e, list(HYDROPHOBIC))
    hbond = polar_pair & (dist >= 2.5) & (dist <= 3.5)
    hydrophobic = hyd_pair & (dist <= 4.5)
    contact = (dist >= 3.0) & (dist <= 5.0)
    clash = dist < 2.0

    rec_interface = set(rec_i.tolist())
    lig_interface = set(lig_i.tolist())
    rec_polar_interface = {i for i in rec_interface if rec_elem[i] in POLAR}
    lig_polar_interface = {i for i in lig_interface if lig_elem[i] in POLAR}
    rec_satisfied = set(rec_i[hbond].tolist())
    lig_satisfied = set(lig_i[hbond].tolist())
    unsatisfied = (
        len(rec_polar_interface - rec_satisfied)
        + len(lig_polar_interface - lig_satisfied)
    )
    return {
        'atom_contact': int(contact.sum()),
        'atom_clash': int(clash.sum()),
        'atom_hbond': int(hbond.sum()),
        'atom_hydrophobic': int(hydrophobic.sum()),
        'atom_unsatisfied': int(unsatisfied),
        'atom_interface_atoms': int(len(rec_interface) + len(lig_interface)),
    }


def compute_atom_score(row, args):
    base = to_float(row.get(args.base_score_column), to_float(row.get('score'), 0.0))
    return (
        base
        + args.contact_weight * math.log1p(to_float(row.get('atom_contact')))
        + args.hbond_weight * math.log1p(to_float(row.get('atom_hbond')))
        + args.hydrophobic_weight * math.log1p(to_float(row.get('atom_hydrophobic')))
        - args.clash_weight * math.log1p(to_float(row.get('atom_clash')))
        - args.unsatisfied_weight * math.log1p(to_float(row.get('atom_unsatisfied')))
    )


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detail_csv', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--top_n', type=int, default=200)
    parser.add_argument('--base_score_column', default='mdn_score')
    parser.add_argument('--contact_weight', type=float, default=0.003)
    parser.add_argument('--hbond_weight', type=float, default=0.006)
    parser.add_argument('--hydrophobic_weight', type=float, default=0.003)
    parser.add_argument('--clash_weight', type=float, default=0.030)
    parser.add_argument('--unsatisfied_weight', type=float, default=0.010)
    args = parser.parse_args()

    rows = read_rows(args.detail_csv)
    by_target = defaultdict(list)
    for row in rows:
        by_target[row.get('target', '')].append(row)

    n_done = 0
    for target, target_rows in sorted(by_target.items()):
        pdb_path = os.path.join(args.data_dir, f'{target}.pdb')
        if not os.path.exists(pdb_path):
            continue
        models = split_models(pdb_path)
        ranked = sorted(
            target_rows,
            key=lambda r: to_float(r.get(args.base_score_column), -1e9),
            reverse=True,
        )
        process_ids = {
            int(to_float(r.get('model_id'), -1))
            for r in ranked[:args.top_n]
        }
        feature_cache = {}
        for mid in process_ids:
            chains = models.get(mid)
            if not chains:
                continue
            rec_ch, lig_ch = pick_rec_lig_chains(chains)
            if rec_ch is None:
                continue
            feats = atom_features(chains[rec_ch], chains[lig_ch])
            feature_cache[mid] = feats
        for row in target_rows:
            mid = int(to_float(row.get('model_id'), -1))
            feats = feature_cache.get(mid)
            if feats is None:
                feats = {key: 0 for key in ATOM_FIELDS if key != 'atom_score'}
            for key, value in feats.items():
                row[key] = value
            row['atom_score'] = compute_atom_score(row, args)
        n_done += 1
        if n_done % 10 == 0:
            print(f'processed targets: {n_done}/{len(by_target)}')

    fieldnames = list(rows[0].keys()) if rows else []
    for key in ATOM_FIELDS:
        if key not in fieldnames:
            fieldnames.append(key)
    write_rows(args.out, rows, fieldnames)
    print(f'Wrote atom features: {args.out}')


if __name__ == '__main__':
    main()
