#!/usr/bin/env python3
"""
Train/apply a lightweight TraDock second-stage reranker.

Training data must come from train/validation decoys. Final test sets such as
CAPRI113 should only be passed with --apply_csv and a saved --model_in.
"""
import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GOOD_LEVELS = {'acceptable', 'medium', 'high'}
MID_LEVELS = {'medium', 'high'}
HIGH_LEVELS = {'high'}
TOPKS = (1, 2, 5, 10, 20, 100)


FEATURE_NAMES = [
    'score',
    'mdn_score',
    'energy_score',
    'enhanced_score',
    'log_n_interface_pairs',
    'log_n_close_pairs',
    'log_n_clash_pairs',
    'contact_bonus',
    'close_contact_bonus',
    'sparse_penalty',
    'clash_penalty',
    'log_atom_contact',
    'log_atom_clash',
    'log_atom_hbond',
    'log_atom_hydrophobic',
    'log_atom_unsatisfied',
    'log_atom_interface_atoms',
]


def _to_float(row, key, default=0.0):
    value = row.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if not np.isfinite(out):
        return default
    return out


def _label(row, metric, threshold):
    if metric == 'classification':
        return str(row.get('classification', '')).lower() in GOOD_LEVELS
    return _to_float(row, metric, 0.0) >= threshold


def _feature_value(row, name):
    score = _to_float(row, 'score', 0.0)
    mdn_score = _to_float(row, 'mdn_score', score)
    energy_score = _to_float(row, 'energy_score', 0.0)
    enhanced_score = _to_float(row, 'enhanced_score', score)
    n_interface = max(0.0, _to_float(row, 'n_interface_pairs', 0.0))
    n_close = max(0.0, _to_float(row, 'n_close_pairs', 0.0))
    n_clash = max(0.0, _to_float(row, 'n_clash_pairs', 0.0))
    values = {
        'score': score,
        'mdn_score': mdn_score,
        'energy_score': energy_score,
        'enhanced_score': enhanced_score,
        'log_n_interface_pairs': float(np.log1p(n_interface)),
        'log_n_close_pairs': float(np.log1p(n_close)),
        'log_n_clash_pairs': float(np.log1p(n_clash)),
        'contact_bonus': _to_float(row, 'contact_bonus', float(np.log1p(n_interface))),
        'close_contact_bonus': _to_float(row, 'close_contact_bonus', float(np.log1p(n_close))),
        'sparse_penalty': _to_float(row, 'sparse_penalty', 0.0),
        'clash_penalty': _to_float(row, 'clash_penalty', float(np.log1p(n_clash))),
        'log_atom_contact': float(np.log1p(max(0.0, _to_float(row, 'atom_contact', 0.0)))),
        'log_atom_clash': float(np.log1p(max(0.0, _to_float(row, 'atom_clash', 0.0)))),
        'log_atom_hbond': float(np.log1p(max(0.0, _to_float(row, 'atom_hbond', 0.0)))),
        'log_atom_hydrophobic': float(np.log1p(max(0.0, _to_float(row, 'atom_hydrophobic', 0.0)))),
        'log_atom_unsatisfied': float(np.log1p(max(0.0, _to_float(row, 'atom_unsatisfied', 0.0)))),
        'log_atom_interface_atoms': float(np.log1p(max(0.0, _to_float(row, 'atom_interface_atoms', 0.0)))),
    }
    return values.get(name, _to_float(row, name, 0.0))


def _feature_row(row, feature_names=None):
    names = feature_names or FEATURE_NAMES
    return [_feature_value(row, name) for name in names]


def _read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if not row.get('target'):
                    row['target'] = row.get('stem') or row.get('pdb_id') or row.get('complex') or ''
                rows.append(row)
    return rows


def _matrix(rows, feature_names=None):
    x = np.asarray([_feature_row(r, feature_names) for r in rows], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _ranked_by(rows, score_key):
    return sorted(rows, key=lambda r: _to_float(r, score_key, -1e9), reverse=True)


def _group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get('target', '')].append(row)
    return dict(grouped)


def _success(labels, k):
    return int(any(labels[:min(k, len(labels))]))


def _eval_rows(rows, score_key, label_metric, label_threshold):
    grouped = _group_rows(rows)
    summary = []
    aggregate = {
        'n_targets': len(grouped),
        'n_targets_with_positive': 0,
        'mean_auc': 0.0,
    }
    aucs = []

    for target, target_rows in sorted(grouped.items()):
        ranked = _ranked_by(target_rows, score_key)
        labels_all = [_label(r, label_metric, label_threshold) for r in target_rows]
        labels_ranked = [_label(r, label_metric, label_threshold) for r in ranked]
        dockq_ranked = [_to_float(r, 'dockq', 0.0) >= 0.23 for r in ranked]
        any_ranked = [str(r.get('classification', '')).lower() in GOOD_LEVELS for r in ranked]
        med_ranked = [str(r.get('classification', '')).lower() in MID_LEVELS for r in ranked]
        high_ranked = [str(r.get('classification', '')).lower() in HIGH_LEVELS for r in ranked]

        y_score = [_to_float(r, score_key, 0.0) for r in target_rows]
        if any(labels_all) and not all(labels_all):
            try:
                aucs.append(float(roc_auc_score(labels_all, y_score)))
            except ValueError:
                pass
        if any(labels_all):
            aggregate['n_targets_with_positive'] += 1

        top = ranked[0]
        out = {
            'target': target,
            'n_models': len(target_rows),
            'n_positive': int(sum(labels_all)),
            'top1_model_id': top.get('model_id', ''),
            'top1_score': _to_float(top, score_key, 0.0),
            'top1_original_score': _to_float(top, 'score', 0.0),
            'top1_fnat': _to_float(top, 'fnat', 0.0),
            'top1_dockq': _to_float(top, 'dockq', 0.0),
            'top1_class': top.get('classification', ''),
        }
        for k in TOPKS:
            out[f'success@{k}'] = _success(labels_ranked, k)
            out[f'success_dockq@{k}'] = _success(dockq_ranked, k)
            out[f'success_any@{k}'] = _success(any_ranked, k)
            out[f'success_med@{k}'] = _success(med_ranked, k)
            out[f'success_high@{k}'] = _success(high_ranked, k)
        summary.append(out)

    denom = max(1, len(summary))
    for k in TOPKS:
        aggregate[f'success@{k}'] = sum(r[f'success@{k}'] for r in summary) / denom
        aggregate[f'success_dockq@{k}'] = sum(r[f'success_dockq@{k}'] for r in summary) / denom
        aggregate[f'success_any@{k}'] = sum(r[f'success_any@{k}'] for r in summary) / denom
    aggregate['mean_auc'] = float(np.mean(aucs)) if aucs else 0.0
    return summary, aggregate


def _write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _fit(train_rows, label_metric, label_threshold, valid_fraction, seed):
    groups = np.asarray([r.get('target', '') for r in train_rows])
    y = np.asarray([_label(r, label_metric, label_threshold) for r in train_rows], dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError('training labels need both positive and negative rows')

    train_idx = np.arange(len(train_rows))
    valid_idx = np.asarray([], dtype=np.int64)
    if valid_fraction > 0 and len(set(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed)
        train_idx, valid_idx = next(splitter.split(train_idx, y, groups))

    x = _matrix(train_rows, FEATURE_NAMES)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=2000, random_state=seed),
    )
    model.fit(x[train_idx], y[train_idx])

    if len(valid_idx) > 0:
        valid_rows = [dict(train_rows[i]) for i in valid_idx]
        pred = model.predict_proba(x[valid_idx])[:, 1]
        for row, score in zip(valid_rows, pred):
            row['rerank_score'] = float(score)
        _, before = _eval_rows(valid_rows, 'score', label_metric, label_threshold)
        _, after = _eval_rows(valid_rows, 'rerank_score', label_metric, label_threshold)
        print('Validation targets:', before['n_targets'])
        for k in (1, 5, 10, 20):
            print(
                f"  Success@{k}: original={before[f'success@{k}']:.3f} "
                f"rerank={after[f'success@{k}']:.3f}"
            )

    return model


def _apply(model_blob, rows):
    model = model_blob['model']
    feature_names = model_blob.get('feature_names', FEATURE_NAMES)
    scores = model.predict_proba(_matrix(rows, feature_names))[:, 1]
    grouped = _group_rows(rows)
    original_rank = {}
    rerank_rank = {}
    for target, target_rows in grouped.items():
        for i, row in enumerate(_ranked_by(target_rows, 'score'), 1):
            original_rank[(target, row.get('model_id', ''), id(row))] = i
        for row, score in zip(rows, scores):
            if row.get('target', '') == target:
                row['rerank_score'] = float(score)
        for i, row in enumerate(_ranked_by(target_rows, 'rerank_score'), 1):
            rerank_rank[(target, row.get('model_id', ''), id(row))] = i

    out = []
    for row in rows:
        key = (row.get('target', ''), row.get('model_id', ''), id(row))
        new_row = dict(row)
        new_row['rerank_score'] = float(new_row['rerank_score'])
        new_row['original_rank'] = original_rank.get(key, '')
        new_row['rerank_rank'] = rerank_rank.get(key, '')
        out.append(new_row)
    return out


def main():
    parser = argparse.ArgumentParser(description='Train/apply TraDock reranker')
    parser.add_argument('--train_csv', nargs='*', default=[])
    parser.add_argument('--model_out')
    parser.add_argument('--model_in')
    parser.add_argument('--apply_csv')
    parser.add_argument('--out')
    parser.add_argument('--label_metric', choices=['dockq', 'fnat', 'classification'], default='dockq')
    parser.add_argument('--label_threshold', type=float, default=0.23)
    parser.add_argument('--valid_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    model_blob = None
    if args.train_csv:
        train_rows = _read_rows(args.train_csv)
        model = _fit(
            train_rows,
            label_metric=args.label_metric,
            label_threshold=args.label_threshold,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )
        model_blob = {
            'model': model,
            'feature_names': FEATURE_NAMES,
            'label_metric': args.label_metric,
            'label_threshold': args.label_threshold,
        }
        if args.model_out:
            os.makedirs(os.path.dirname(args.model_out) if os.path.dirname(args.model_out) else '.', exist_ok=True)
            with open(args.model_out, 'wb') as f:
                pickle.dump(model_blob, f)
            print('Saved model:', args.model_out)

    if args.model_in:
        with open(args.model_in, 'rb') as f:
            model_blob = pickle.load(f)

    if args.apply_csv:
        if model_blob is None:
            raise ValueError('--apply_csv requires --model_in or --train_csv')
        if not args.out:
            raise ValueError('--apply_csv requires --out')
        rows = _read_rows([args.apply_csv])
        reranked = _apply(model_blob, rows)
        fieldnames = list(reranked[0].keys()) if reranked else []
        for key in ('rerank_score', 'original_rank', 'rerank_rank'):
            if key not in fieldnames:
                fieldnames.append(key)
        _write_csv(args.out, reranked, fieldnames)

        metric = model_blob.get('label_metric', args.label_metric)
        threshold = float(model_blob.get('label_threshold', args.label_threshold))
        summary, aggregate = _eval_rows(reranked, 'rerank_score', metric, threshold)
        summary_path = args.out[:-4] + '.summary.csv' if args.out.endswith('.csv') else args.out + '.summary.csv'
        aggregate_path = summary_path[:-4] + '.aggregate.csv'
        _write_csv(summary_path, summary)
        _write_csv(aggregate_path, [aggregate])
        print('Wrote:', args.out)
        print('Summary:', summary_path)
        print('Aggregate:', aggregate_path)
        print(
            f"Rerank Success@1={aggregate['success@1']:.3f} "
            f"Success@5={aggregate['success@5']:.3f} "
            f"Success@10={aggregate['success@10']:.3f} "
            f"Success@20={aggregate['success@20']:.3f}"
        )


if __name__ == '__main__':
    main()
