#!/usr/bin/env python3
"""Create SVG montages for TraDock top1-success structures."""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np


SVG_STYLE = """
text{font-family:Arial,Helvetica,sans-serif}
.title{font-size:24px;font-weight:700}
.subtitle{font-size:14px;fill:#444}
.panel-title{font-size:13px;font-weight:700}
.metric{font-size:11px;fill:#333}
.label{font-size:12px;font-weight:700}
"""


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
    with open(pdb_path, errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()
            if rec == 'MODEL':
                current_model = int(line.split()[1])
                current_chains = {}
            elif rec == 'ENDMDL':
                if current_model is not None:
                    models[current_model] = current_chains
                current_model = None
            elif rec in ('ATOM', 'HETATM', 'TER', 'ANISOU') and current_model is not None:
                if len(line) > 21:
                    current_chains.setdefault(line[21], []).append(line)
    if models:
        return models

    chains = {}
    with open(pdb_path, errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ('ATOM', 'HETATM', 'TER', 'ANISOU') and len(line) > 21:
                chains.setdefault(line[21], []).append(line)
    return {0: chains} if chains else {}


def pick_rec_lig_chains(chains):
    chain_ids = list(chains.keys())
    if len(chain_ids) < 2:
        return None, None
    rec_ch = 'A' if 'A' in chains else chain_ids[0]
    lig_ch = 'B' if 'B' in chains else chain_ids[1]
    return rec_ch, lig_ch


def parse_points(lines):
    ca = []
    atoms = []
    for line in lines:
        rec = line[:6].strip()
        if rec not in ('ATOM', 'HETATM') or len(line) < 54:
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        atoms.append(xyz)
        if line[12:16].strip() == 'CA':
            ca.append(xyz)
    pts = ca if ca else atoms
    return np.asarray(pts, dtype=np.float32)


def write_model_pdb(chains, path):
    with open(path, 'w') as f:
        for chain_id in chains:
            for line in chains[chain_id]:
                if line[:6].strip() in ('ATOM', 'HETATM', 'TER', 'ANISOU'):
                    f.write(line)
        f.write('END\n')


def pca_project(points):
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        return centered[:, :2]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def fit_xy(xy, x0, y0, width, height, pad=22):
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    out = (xy - min_xy) * scale
    out[:, 0] += x0 + (width - span[0] * scale) / 2
    out[:, 1] = y0 + height - ((out[:, 1]) + (height - span[1] * scale) / 2)
    return out


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def svg_escape(text):
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )


def draw_panel(svg, item, x, y, w, h, ligand_color):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#d6d6d6"/>')
    svg.append(
        f'<text x="{x+10}" y="{y+18}" class="panel-title">'
        f'{svg_escape(item["label"])} {svg_escape(item["target"])} model {svg_escape(item["model_id"])}</text>'
    )
    svg.append(
        f'<text x="{x+10}" y="{y+34}" class="metric">'
        f'MDN={item["mdn_score"]:.4f}  Fnat={item["fnat"]:.3f}  DockQ={item["dockq"]:.3f}</text>'
    )

    rec = sample_points(item['rec_points'], 650)
    lig = sample_points(item['lig_points'], 450)
    if len(rec) == 0 or len(lig) == 0:
        svg.append(f'<text x="{x+10}" y="{y+70}" class="metric">No drawable atoms</text>')
        return
    all_pts = np.vstack([rec, lig])
    xy = pca_project(all_pts)
    xy = fit_xy(xy, x + 6, y + 44, w - 12, h - 50)
    rec_xy = xy[:len(rec)]
    lig_xy = xy[len(rec):]
    for px, py in rec_xy:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.15" fill="#8a8a8a" opacity="0.38"/>')
    for px, py in lig_xy:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.85" fill="{ligand_color}" opacity="0.78"/>')


def read_success_rows(summary_csv):
    rows = []
    with open(summary_csv, newline='') as f:
        for row in csv.DictReader(f):
            if int(to_float(row.get('success@1', row.get('success_top1', 0)))) != 1:
                continue
            rows.append({
                'target': row['target'],
                'model_id': str(int(to_float(row.get('top1_model_id'), -1))),
                'mdn_score': to_float(row.get('top1_score')),
                'fnat': to_float(row.get('top1_fnat')),
                'dockq': to_float(row.get('top1_dockq')),
                'classification': row.get('top1_class', ''),
            })
    return rows


def load_structure_items(rows, data_dir, out_struct_dir):
    items = []
    out_struct_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        pdb_path = Path(data_dir) / f'{row["target"]}.pdb'
        if not pdb_path.exists():
            continue
        mid = int(row['model_id'])
        models = split_models(pdb_path)
        chains = models.get(mid)
        if not chains:
            continue
        rec_ch, lig_ch = pick_rec_lig_chains(chains)
        if rec_ch is None:
            continue
        item = dict(row)
        item['rec_chain'] = rec_ch
        item['lig_chain'] = lig_ch
        item['rec_points'] = parse_points(chains[rec_ch])
        item['lig_points'] = parse_points(chains[lig_ch])
        pdb_out = out_struct_dir / f'{row["target"]}_model{mid}_tradock_top1_success.pdb'
        write_model_pdb(chains, pdb_out)
        item['extracted_pdb'] = str(pdb_out)
        items.append(item)
    return items


def make_montage(metric_name, selected, out_svg):
    width = 1680
    height = 760
    margin_x = 28
    panel_w = 318
    panel_h = 305
    gap = 12
    y_top = 100
    y_bottom = 430
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<style>{SVG_STYLE}</style>',
        f'<text x="30" y="34" class="title">TraDock top1-success structures sorted by {svg_escape(metric_name)}</text>',
        '<text x="30" y="58" class="subtitle">Gray = receptor chain A/first chain; colored = ligand chain B/second chain. Row 1 = top five, row 2 = bottom five among TraDock fnat&gt;0.3 top1 successes.</text>',
        '<text x="30" y="88" class="label" fill="#1f77b4">Top five</text>',
        '<text x="30" y="418" class="label" fill="#d95f02">Bottom five</text>',
    ]
    for row_idx, row_items in enumerate([selected[:5], selected[5:]]):
        y = y_top if row_idx == 0 else y_bottom
        color = '#1f77b4' if row_idx == 0 else '#d95f02'
        for i, item in enumerate(row_items):
            item = dict(item)
            item['label'] = f'#{i+1}' if row_idx == 0 else f'#{len(row_items)-i}'
            x = margin_x + i * (panel_w + gap)
            draw_panel(svg, item, x, y, panel_w, panel_h, color)
    svg.append('</svg>')
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text('\n'.join(svg))


def write_selection_csv(path, metric, top_items, bottom_items):
    rows = []
    for group, items in [('top5', top_items), ('bottom5', bottom_items)]:
        for rank, item in enumerate(items, 1):
            rows.append({
                'metric': metric,
                'group': group,
                'rank_in_group': rank,
                'target': item['target'],
                'model_id': item['model_id'],
                'mdn_score': item['mdn_score'],
                'fnat': item['fnat'],
                'dockq': item['dockq'],
                'classification': item['classification'],
                'rec_chain': item['rec_chain'],
                'lig_chain': item['lig_chain'],
                'extracted_pdb': item['extracted_pdb'],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary_csv', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    success_rows = read_success_rows(args.summary_csv)
    items = load_structure_items(success_rows, args.data_dir, out_dir / 'structures')
    if len(items) < 10:
        raise RuntimeError(f'Need at least 10 drawable success structures, got {len(items)}')

    metric_specs = [
        ('mdn_score', 'MDN score', lambda x: x['mdn_score']),
        ('fnat', 'Fnat', lambda x: x['fnat']),
        ('dockq', 'DockQ', lambda x: x['dockq']),
    ]
    for metric_key, title, key_fn in metric_specs:
        ranked = sorted(items, key=key_fn, reverse=True)
        top = ranked[:5]
        bottom = list(reversed(ranked[-5:]))
        make_montage(title, top + bottom, out_dir / f'tradock_top1_success_by_{metric_key}.svg')
        write_selection_csv(out_dir / f'tradock_top1_success_by_{metric_key}.csv', metric_key, top, bottom)
        print(f'{metric_key}: wrote top/bottom montage and CSV')


if __name__ == '__main__':
    main()
