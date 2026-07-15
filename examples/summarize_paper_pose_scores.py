#!/usr/bin/env python3
"""
Summarize a detail CSV from examples/eval_db5_paper_tradock.py.

It reports TraDock reranking, original input-rank ordering, oracle ordering,
and score-vs-DockQ correlations/classification metrics.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def fval(value):
    if value in (None, ''):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def ival(value, default=999999):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def corr(xs, ys):
    vals = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 2:
        return math.nan
    x = [v[0] for v in vals]
    y = [v[1] for v in vals]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return math.nan
    return num / (den_x * den_y)


def spearman(xs, ys):
    vals = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 2:
        return math.nan
    return corr(rankdata([x for x, _ in vals]), rankdata([y for _, y in vals]))


def roc_auc(scores, labels):
    vals = [
        (score, label) for score, label in zip(scores, labels)
        if math.isfinite(score)
    ]
    pos = [s for s, label in vals if label == 1]
    neg = [s for s, label in vals if label == 0]
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def average_precision(scores, labels):
    vals = [
        (score, label) for score, label in zip(scores, labels)
        if math.isfinite(score)
    ]
    npos = sum(label for _, label in vals)
    if npos == 0:
        return math.nan
    vals = sorted(vals, key=lambda x: x[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(vals, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / npos


def mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def success_at(rows, k, threshold):
    return int(any(row['dockq_f'] >= threshold for row in rows[:min(k, len(rows))]))


def read_done_rows(path):
    rows = []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('status') != 'done':
                continue
            score = fval(row.get('score'))
            dockq = fval(row.get('dockq'))
            if not math.isfinite(score) or not math.isfinite(dockq):
                continue
            row['score_f'] = score
            row['dockq_f'] = dockq
            row['input_rank_i'] = ival(row.get('input_rank'))
            rows.append(row)
    return rows


def summarize_target(target, rows, threshold):
    tradock = sorted(rows, key=lambda r: r['score_f'], reverse=True)
    input_ranked = sorted(rows, key=lambda r: r['input_rank_i'])
    oracle = sorted(rows, key=lambda r: r['dockq_f'], reverse=True)
    out = {
        'target': target,
        'n_valid': len(rows),
        'best_available_dockq': oracle[0]['dockq_f'],
        'tradock_top1_pose': tradock[0]['pose_model'],
        'tradock_top1_dockq': tradock[0]['dockq_f'],
        'input_top1_pose': input_ranked[0]['pose_model'],
        'input_top1_dockq': input_ranked[0]['dockq_f'],
        'oracle_top1_pose': oracle[0]['pose_model'],
        'oracle_top1_dockq': oracle[0]['dockq_f'],
        'spearman_score_dockq': spearman(
            [r['score_f'] for r in rows],
            [r['dockq_f'] for r in rows],
        ),
    }
    for k in (1, 3, 5, 10, 100):
        out[f'tradock_success@{k}'] = success_at(tradock, k, threshold)
        out[f'input_success@{k}'] = success_at(input_ranked, k, threshold)
        out[f'oracle_success@{k}'] = success_at(oracle, k, threshold)
    return out


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detail', required=True)
    parser.add_argument('--target_out', required=True)
    parser.add_argument('--summary_out', required=True)
    parser.add_argument('--method_name', default='input')
    parser.add_argument('--positive_dockq', type=float, default=0.23)
    args = parser.parse_args()

    rows = read_done_rows(args.detail)
    by_target = defaultdict(list)
    for row in rows:
        by_target[row['target']].append(row)
    target_rows = [
        summarize_target(target, target_rows, args.positive_dockq)
        for target, target_rows in sorted(by_target.items())
    ]

    all_scores = [row['score_f'] for row in rows]
    all_dockq = [row['dockq_f'] for row in rows]
    all_labels = [int(v >= args.positive_dockq) for v in all_dockq]
    summary = {
        'method_name': args.method_name,
        'n_targets': len(target_rows),
        'n_valid_decoys': len(rows),
        'positive_dockq': args.positive_dockq,
        'mean_best_available_dockq': mean([r['best_available_dockq'] for r in target_rows]),
        'mean_tradock_top1_dockq': mean([r['tradock_top1_dockq'] for r in target_rows]),
        'mean_input_top1_dockq': mean([r['input_top1_dockq'] for r in target_rows]),
        'mean_oracle_top1_dockq': mean([r['oracle_top1_dockq'] for r in target_rows]),
        'pooled_pearson_score_dockq': corr(all_scores, all_dockq),
        'pooled_spearman_score_dockq': spearman(all_scores, all_dockq),
        'mean_per_target_spearman_score_dockq': mean([
            r['spearman_score_dockq'] for r in target_rows
        ]),
        'auroc_score_for_dockq_ge_threshold': roc_auc(all_scores, all_labels),
        'auprc_score_for_dockq_ge_threshold': average_precision(all_scores, all_labels),
    }
    for k in (1, 3, 5, 10, 100):
        summary[f'tradock_success@{k}'] = mean([r[f'tradock_success@{k}'] for r in target_rows])
        summary[f'{args.method_name}_success@{k}'] = mean([r[f'input_success@{k}'] for r in target_rows])
        summary[f'oracle_success@{k}'] = mean([r[f'oracle_success@{k}'] for r in target_rows])

    write_rows(args.target_out, target_rows)
    with open(args.summary_out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f'target_summary -> {args.target_out}')
    print(f'summary -> {args.summary_out}')
    for key, value in summary.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
