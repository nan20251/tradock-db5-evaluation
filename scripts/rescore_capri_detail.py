#!/usr/bin/env python3
"""Recompute top-k CAPRI summaries from an existing detail CSV."""
import argparse
import csv
import os
from collections import defaultdict


GOOD_LEVELS = {'acceptable', 'medium', 'high'}
MID_LEVELS = {'medium', 'high'}
HIGH_LEVELS = {'high'}
TOPKS = (1, 2, 5, 10, 20, 100)


def to_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def is_positive(row, metric, threshold):
    if metric == 'classification':
        return str(row.get('classification', '')).lower() in GOOD_LEVELS
    if metric == 'fnat':
        return to_float(row.get('fnat')) > threshold
    if metric == 'dockq':
        return to_float(row.get('dockq')) >= threshold
    raise ValueError(f'unknown metric: {metric}')


def success(labels, k):
    return int(any(labels[:min(k, len(labels))]))


def read_rows(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, score_column, pos_metric, pos_threshold, dockq_threshold):
    by_target = defaultdict(list)
    for row in rows:
        by_target[row.get('target', '')].append(row)

    detail_rows = []
    summary_rows = []
    for target, target_rows in sorted(by_target.items()):
        ranked = sorted(
            target_rows,
            key=lambda r: to_float(r.get(score_column), -1e9),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            out = dict(row)
            out['rescore_column'] = score_column
            out['rescore_score'] = to_float(row.get(score_column), -1e9)
            out['rescore_rank'] = rank
            detail_rows.append(out)

        pos = [is_positive(r, pos_metric, pos_threshold) for r in ranked]
        dockq_pos = [to_float(r.get('dockq')) >= dockq_threshold for r in ranked]
        any_pos = [str(r.get('classification', '')).lower() in GOOD_LEVELS for r in ranked]
        med_pos = [str(r.get('classification', '')).lower() in MID_LEVELS for r in ranked]
        high_pos = [str(r.get('classification', '')).lower() in HIGH_LEVELS for r in ranked]
        top = ranked[0]
        row = {
            'target': target,
            'score_column': score_column,
            'n_models': len(ranked),
            'n_positive': sum(pos),
            'top1_model_id': top.get('model_id', ''),
            'top1_score': to_float(top.get(score_column), -1e9),
            'top1_fnat': to_float(top.get('fnat')),
            'top1_dockq': to_float(top.get('dockq')),
            'top1_class': top.get('classification', ''),
        }
        for k in TOPKS:
            row[f'success@{k}'] = success(pos, k)
            row[f'success_dockq@{k}'] = success(dockq_pos, k)
            row[f'success_any@{k}'] = success(any_pos, k)
            row[f'success_med@{k}'] = success(med_pos, k)
            row[f'success_high@{k}'] = success(high_pos, k)
        summary_rows.append(row)

    denom = max(1, len(summary_rows))
    aggregate = {
        'score_column': score_column,
        'pos_metric': pos_metric,
        'pos_threshold': pos_threshold,
        'dockq_threshold': dockq_threshold,
        'n_targets': len(summary_rows),
        'n_targets_with_positive': sum(int(r['n_positive']) > 0 for r in summary_rows),
    }
    for k in TOPKS:
        aggregate[f'success@{k}'] = sum(r[f'success@{k}'] for r in summary_rows) / denom
        aggregate[f'success_dockq@{k}'] = sum(r[f'success_dockq@{k}'] for r in summary_rows) / denom
        aggregate[f'success_any@{k}'] = sum(r[f'success_any@{k}'] for r in summary_rows) / denom
        aggregate[f'success_med@{k}'] = sum(r[f'success_med@{k}'] for r in summary_rows) / denom
        aggregate[f'success_high@{k}'] = sum(r[f'success_high@{k}'] for r in summary_rows) / denom
    return detail_rows, summary_rows, aggregate


def main():
    parser = argparse.ArgumentParser(description='Rescore an existing CAPRI detail CSV')
    parser.add_argument('--input', required=True)
    parser.add_argument('--score_column', required=True)
    parser.add_argument('--out_prefix', required=True)
    parser.add_argument('--pos_metric', choices=['fnat', 'dockq', 'classification'], default='fnat')
    parser.add_argument('--pos_threshold', type=float, default=0.3)
    parser.add_argument('--dockq_threshold', type=float, default=0.23)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f'no rows in {args.input}')
    if args.score_column not in rows[0]:
        raise ValueError(f'{args.score_column} not found in {args.input}')

    detail, summary, aggregate = summarize(
        rows, args.score_column, args.pos_metric,
        args.pos_threshold, args.dockq_threshold,
    )
    detail_path = args.out_prefix + '.csv'
    summary_path = args.out_prefix + '.summary.csv'
    aggregate_path = args.out_prefix + '.summary.aggregate.csv'
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_csv(aggregate_path, [aggregate])
    print('detail:', detail_path)
    print('summary:', summary_path)
    print('aggregate:', aggregate_path)
    print(
        f"Success@1={aggregate['success@1']:.3f} "
        f"Success@5={aggregate['success@5']:.3f} "
        f"Success@10={aggregate['success@10']:.3f} "
        f"Success@20={aggregate['success@20']:.3f}"
    )


if __name__ == '__main__':
    main()
