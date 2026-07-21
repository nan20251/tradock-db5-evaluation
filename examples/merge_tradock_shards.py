#!/usr/bin/env python3
"""Merge TraDock shard CSV outputs from run_db5_eval_multigpu.sh."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_db5_paper_tradock import (  # noqa: E402
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    write_aggregate,
)


def summary_path(out_path: str) -> str:
    if out_path.endswith('.csv'):
        return out_path[:-4] + '.summary.csv'
    return out_path + '.summary.csv'


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True,
                    help='OUT_PREFIX used by run_db5_eval_multigpu.sh')
    ap.add_argument('--n_shards', type=int, required=True)
    ap.add_argument('--pose_models', default='',
                    help='Optional comma list for aggregate metadata')
    args = ap.parse_args()

    prefix = Path(args.prefix)
    detail_rows = []
    summary_rows = []
    for i in range(args.n_shards):
        detail_p = Path(f'{prefix}.shard{i}of{args.n_shards}.csv')
        summary_p = Path(summary_path(str(detail_p)))
        drows = read_csv(detail_p)
        srows = read_csv(summary_p)
        print(f'[merge] {detail_p.name}: detail={len(drows)} summary={len(srows)}')
        detail_rows.extend(drows)
        summary_rows.extend(srows)

    def key_row(r):
        try:
            return int(r.get('target_index') or 0), r.get('target', '')
        except ValueError:
            return 0, r.get('target', '')

    detail_rows.sort(key=key_row)
    summary_rows.sort(key=key_row)

    out_detail = Path(f'{prefix}.csv')
    out_summary = Path(summary_path(str(out_detail)))
    write_csv(out_detail, DETAIL_FIELDS, detail_rows)
    write_csv(out_summary, SUMMARY_FIELDS, summary_rows)
    poses = [x for x in args.pose_models.split(',') if x.strip()] or ['merged']
    agg = write_aggregate(str(out_summary), poses)
    print(f'[ok] detail -> {out_detail}')
    print(f'[ok] summary -> {out_summary}')
    if agg:
        print(f'[ok] aggregate -> {agg[0]}')


if __name__ == '__main__':
    main()
