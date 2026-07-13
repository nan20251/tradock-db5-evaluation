#!/usr/bin/env python3
"""Compare TraDock and BioScore CAPRI per-target metrics.

Both inputs must contain one row per CAPRI target. TraDock can be the
`*.summary.csv` produced by examples/eval_capri_fast.py; BioScore can be
`per_target_full.csv` from BioScore-PPI.
"""
import argparse
import csv
import os
from statistics import mean, median


METRICS = [
    'spearman',
    'pearson',
    'top20_spearman',
    'auc_pos',
    'auc_any',
    'auc_med',
    'auc_high',
    'auc_dockq',
    'success@1',
    'success@2',
    'success@5',
    'success@10',
    'success@100',
    'success_any@1',
    'success_any@2',
    'success_any@5',
    'success_any@10',
    'success_any@100',
    'hitrate@1',
    'hitrate@2',
    'hitrate@5',
    'hitrate@10',
    'hitrate@100',
]

ALIASES = {
    'spearman': ['spearman', 'spearman_r', 'Spearman'],
    'pearson': ['pearson', 'Pearson'],
    'auc_pos': ['auc_pos', 'auc', 'AUC'],
    'success@1': ['success@1', 'success_top1', 'Succ@1', 'success_top1'],
    'success@2': ['success@2', 'success_top2', 'success_top2'],
    'success@5': ['success@5', 'success_top5', 'Succ@5', 'success_top5'],
    'success@10': ['success@10', 'success_top10', 'Succ@10', 'success_top10'],
    'success@100': ['success@100', 'success_top100', 'success_top100'],
}


def parse_args():
    p = argparse.ArgumentParser(description='Compare CAPRI per-target metrics.')
    p.add_argument('--tradock', required=True, help='TraDock summary CSV')
    p.add_argument('--bioscore', required=True, help='BioScore per_target_full.csv')
    p.add_argument('--out', default='results/capri_compare_tradock_vs_bioscore.csv')
    p.add_argument('--summary_out', default=None)
    p.add_argument('--only_done', action='store_true',
                   help='For TraDock summary, keep only status=done rows')
    return p.parse_args()


def to_float(v):
    if v in (None, ''):
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() == 'true':
            return 1.0
        if s.lower() == 'false':
            return 0.0
        if s == '':
            return None
        v = s
    try:
        return float(v)
    except ValueError:
        return None


def get_metric(row, name):
    for key in [name] + ALIASES.get(name, []):
        if key in row:
            return to_float(row.get(key))
    return None


def load_rows(path, only_done=False):
    out = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get('target')
            if not target:
                continue
            if only_done and row.get('status') not in (None, '', 'done'):
                continue
            out[target] = row
    return out


def summarize(rows, source):
    out = {'source': source, 'n_targets': len(rows)}
    for metric in METRICS:
        vals = []
        for row in rows.values():
            val = get_metric(row, metric)
            if val is not None:
                vals.append(val)
        if vals:
            out[f'mean_{metric}'] = mean(vals)
            out[f'median_{metric}'] = median(vals)
        else:
            out[f'mean_{metric}'] = ''
            out[f'median_{metric}'] = ''
    return out


def main():
    args = parse_args()
    tradock = load_rows(args.tradock, only_done=args.only_done)
    bioscore = load_rows(args.bioscore)

    targets = sorted(set(tradock) | set(bioscore))
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    fields = ['target', 'in_tradock', 'in_bioscore']
    for metric in METRICS:
        fields.extend([f'tradock_{metric}', f'bioscore_{metric}', f'delta_{metric}'])

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for target in targets:
            trow = tradock.get(target)
            brow = bioscore.get(target)
            row = {
                'target': target,
                'in_tradock': int(trow is not None),
                'in_bioscore': int(brow is not None),
            }
            for metric in METRICS:
                tv = get_metric(trow or {}, metric)
                bv = get_metric(brow or {}, metric)
                row[f'tradock_{metric}'] = '' if tv is None else tv
                row[f'bioscore_{metric}'] = '' if bv is None else bv
                row[f'delta_{metric}'] = '' if tv is None or bv is None else tv - bv
            writer.writerow(row)

    summary_out = args.summary_out
    if summary_out is None:
        summary_out = args.out[:-4] + '.summary.csv' if args.out.endswith('.csv') else args.out + '.summary.csv'
    summary_rows = [summarize(tradock, 'TraDock'), summarize(bioscore, 'BioScore')]
    common_t = {k: v for k, v in tradock.items() if k in bioscore}
    common_b = {k: v for k, v in bioscore.items() if k in tradock}
    summary_rows.extend([summarize(common_t, 'TraDock_common'), summarize(common_b, 'BioScore_common')])

    summary_fields = list(summary_rows[0].keys())
    with open(summary_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f'common_targets={len(set(tradock) & set(bioscore))}')
    print(f'comparison -> {args.out}')
    print(f'summary    -> {summary_out}')


if __name__ == '__main__':
    main()
