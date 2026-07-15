#!/usr/bin/env python3
"""
Compare AlphaFold confidence scores with TraDock scores on the same AF poses.

Inputs:
  --af_scores       CSV from examples/convert_colabfold_db5.py
  --tradock_detail  detail CSV from examples/eval_db5_paper_tradock.py

Outputs:
  merged CSV, per-target summary CSV, and aggregate CSV.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SUCCESS_DOCKQ = 0.23


def fval(value):
    if value in (None, ''):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def rankdata(values):
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
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


def mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def af_score(row):
    score = fval(row.get('ranking_confidence'))
    if math.isfinite(score):
        return score
    score = fval(row.get('iptm'))
    if math.isfinite(score):
        return score
    return fval(row.get('mean_plddt'))


def sort_af(rows):
    return sorted(
        rows,
        key=lambda r: (
            math.isfinite(af_score(r)),
            af_score(r),
            -int(fval(r.get('af_rank')) if math.isfinite(fval(r.get('af_rank'))) else 999),
        ),
        reverse=True,
    )


def sort_tradock(rows):
    return sorted(rows, key=lambda r: fval(r.get('score')), reverse=True)


def sort_oracle(rows):
    return sorted(rows, key=lambda r: fval(r.get('dockq')), reverse=True)


def success_at(rows, k, threshold):
    return int(any(fval(r.get('dockq')) >= threshold for r in rows[:min(k, len(rows))]))


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
    paired = [
        (score, label) for score, label in zip(scores, labels)
        if math.isfinite(score)
    ]
    npos = sum(label for _, label in paired)
    if npos == 0:
        return math.nan
    paired = sorted(paired, key=lambda x: x[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(paired, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / npos


def summarize_target(target, rows, threshold):
    af_rows = sort_af(rows)
    tradock_rows = sort_tradock(rows)
    oracle_rows = sort_oracle(rows)
    af_top = af_rows[0] if af_rows else {}
    tradock_top = tradock_rows[0] if tradock_rows else {}
    oracle_top = oracle_rows[0] if oracle_rows else {}

    out = {
        'target': target,
        'n_poses': len(rows),
        'af_top_pose': af_top.get('pose_model', ''),
        'af_top_dockq': fval(af_top.get('dockq')),
        'af_top_score': af_score(af_top),
        'tradock_top_pose': tradock_top.get('pose_model', ''),
        'tradock_top_dockq': fval(tradock_top.get('dockq')),
        'tradock_top_score': fval(tradock_top.get('score')),
        'oracle_top_pose': oracle_top.get('pose_model', ''),
        'oracle_top_dockq': fval(oracle_top.get('dockq')),
        'spearman_af_dockq': spearman(
            [af_score(r) for r in rows],
            [fval(r.get('dockq')) for r in rows],
        ),
        'spearman_tradock_dockq': spearman(
            [fval(r.get('score')) for r in rows],
            [fval(r.get('dockq')) for r in rows],
        ),
    }
    for k in (1, 3, 5, 10, 100):
        out[f'af_success@{k}'] = success_at(af_rows, k, threshold)
        out[f'tradock_success@{k}'] = success_at(tradock_rows, k, threshold)
        out[f'oracle_success@{k}'] = success_at(oracle_rows, k, threshold)
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
    parser.add_argument('--af_scores', required=True)
    parser.add_argument('--tradock_detail', required=True)
    parser.add_argument('--merged_out', required=True)
    parser.add_argument('--summary_out', required=True)
    parser.add_argument('--aggregate_out', required=True)
    parser.add_argument('--positive_dockq', type=float, default=DEFAULT_SUCCESS_DOCKQ)
    args = parser.parse_args()

    af_rows = read_csv(args.af_scores)
    tradock_rows = read_csv(args.tradock_detail)

    af_by_key = {(r['target'], r['pose_model']): r for r in af_rows}
    merged = []
    for row in tradock_rows:
        if row.get('status') != 'done':
            continue
        key = (row['target'], row['pose_model'])
        af = af_by_key.get(key, {})
        merged_row = dict(row)
        for col in (
            'af_rank', 'ranking_confidence', 'iptm', 'ptm', 'mean_plddt',
            'source_af_rank', 'receptor_align_rmsd', 'mapping_mode', 'source_pdb',
            'converted_ligand',
        ):
            merged_row[col] = af.get(col, '')
        merged_row['af_score_used'] = af_score(merged_row)
        merged.append(merged_row)

    write_rows(args.merged_out, merged)

    by_target = defaultdict(list)
    for row in merged:
        by_target[row['target']].append(row)

    summary = [
        summarize_target(target, rows, args.positive_dockq)
        for target, rows in sorted(by_target.items())
    ]
    write_rows(args.summary_out, summary)

    all_af_scores = [af_score(r) for r in merged]
    all_tradock_scores = [fval(r.get('score')) for r in merged]
    all_dockq = [fval(r.get('dockq')) for r in merged]
    all_labels = [int(v >= args.positive_dockq) if math.isfinite(v) else 0 for v in all_dockq]

    aggregate = {
        'n_targets': len(summary),
        'n_poses': len(merged),
        'positive_dockq': args.positive_dockq,
        'mean_af_top1_dockq': mean([fval(r['af_top_dockq']) for r in summary]),
        'mean_tradock_top1_dockq': mean([fval(r['tradock_top_dockq']) for r in summary]),
        'mean_oracle_top1_dockq': mean([fval(r['oracle_top_dockq']) for r in summary]),
        'mean_spearman_af_dockq': mean([fval(r['spearman_af_dockq']) for r in summary]),
        'mean_spearman_tradock_dockq': mean([fval(r['spearman_tradock_dockq']) for r in summary]),
        'pooled_pearson_af_dockq': corr(all_af_scores, all_dockq),
        'pooled_pearson_tradock_dockq': corr(all_tradock_scores, all_dockq),
        'pooled_spearman_af_dockq': spearman(all_af_scores, all_dockq),
        'pooled_spearman_tradock_dockq': spearman(all_tradock_scores, all_dockq),
        'auroc_af_for_dockq_ge_threshold': roc_auc(all_af_scores, all_labels),
        'auroc_tradock_for_dockq_ge_threshold': roc_auc(all_tradock_scores, all_labels),
        'auprc_af_for_dockq_ge_threshold': average_precision(all_af_scores, all_labels),
        'auprc_tradock_for_dockq_ge_threshold': average_precision(all_tradock_scores, all_labels),
    }
    for k in (1, 3, 5, 10, 100):
        aggregate[f'af_success@{k}'] = mean([fval(r[f'af_success@{k}']) for r in summary])
        aggregate[f'tradock_success@{k}'] = mean([fval(r[f'tradock_success@{k}']) for r in summary])
        aggregate[f'oracle_success@{k}'] = mean([fval(r[f'oracle_success@{k}']) for r in summary])

    with open(args.aggregate_out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate.keys()))
        writer.writeheader()
        writer.writerow(aggregate)

    print(f'merged   -> {args.merged_out}')
    print(f'summary  -> {args.summary_out}')
    print(f'aggregate-> {args.aggregate_out}')
    if summary:
        print(
            'Top1 DockQ: '
            f"AF={aggregate['mean_af_top1_dockq']:.4f} "
            f"TraDock={aggregate['mean_tradock_top1_dockq']:.4f} "
            f"Oracle={aggregate['mean_oracle_top1_dockq']:.4f}"
        )


if __name__ == '__main__':
    main()
