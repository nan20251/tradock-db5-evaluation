#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter docking decoys by overlap with an oracle (native) receptor binding site.

Oracle site = receptor residues within --native-cutoff A of the native ligand.
For each decoy: site_frac = |site residues contacted by decoy ligand| / |site|.
Keep decoys with site_frac >= --site-frac-min.

Examples
--------
# Annotate an existing TraDock detail CSV and report Success@N on kept poses
python scripts/filter_decoys_by_site.py \\
  --detail-csv results/tradock_DB5-u_hdock_all500.shard0of2.csv \\
  --paper-root /path/to/PPCBench_eval \\
  --dataset DB5-u \\
  --out results/hdock_site_filter.csv \\
  --site-frac-min 0.3

# One target: native monomers + decoy pose dirs under paper results
python scripts/filter_decoys_by_site.py \\
  --paper-root /path/to/PPCBench_eval \\
  --dataset DB5-u \\
  --targets 1ACB,2I9B \\
  --pose-prefix hdock \\
  --max-poses 500 \\
  --out results/site_filter_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformerdock.prepare_target.siteFilter import (  # noqa: E402
    decoy_site_overlap,
    keep_by_site_frac,
    oracle_receptor_site,
)

SUCCESS_KS = (1, 5, 10, 25, 50, 100, 200, 500)


def _import_eval_helpers():
    """Reuse path / candidate discovery from the TraDock eval script."""
    examples = os.path.join(ROOT, 'examples')
    if examples not in sys.path:
        sys.path.insert(0, examples)
    import eval_db5_paper_tradock as ev  # type: ignore
    return ev


def parse_args():
    p = argparse.ArgumentParser(
        description='Filter decoys by native receptor binding-site overlap.',
    )
    p.add_argument('--detail-csv', help='Existing TraDock detail CSV (has pred_rec/pred_lig)')
    p.add_argument('--paper-root', required=True, help='PPCBench / PPCBench_eval root')
    p.add_argument('--dataset', default='DB5-u')
    p.add_argument('--targets', default='', help='Comma-separated PDB ids (optional)')
    p.add_argument('--limit', type=int, default=0, help='Only first N targets from dataset json')
    p.add_argument('--pose-prefix', default='hdock')
    p.add_argument('--max-poses', type=int, default=500)
    p.add_argument('--native-cutoff', type=float, default=5.0,
                   help='Native interface residue cutoff (A)')
    p.add_argument('--contact-cutoff', type=float, default=8.0,
                   help='Decoy ligand–site contact cutoff (A)')
    p.add_argument('--site-frac-min', type=float, default=0.3,
                   help='Keep decoy if site_frac >= this')
    p.add_argument('--match-by-resseq', action='store_true',
                   help='Ignore chain id when matching site residues')
    p.add_argument('--out', required=True, help='Output annotated CSV')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def _target_allow(args) -> Optional[Set[str]]:
    if args.targets.strip():
        return {t.strip() for t in args.targets.split(',') if t.strip()}
    return None


def _load_site_cache(ev, dataset_dir, dataset, pdbid, native_cutoff, cache, verbose):
    if pdbid in cache:
        return cache[pdbid]
    gt_rec, gt_lig, _, _ = ev.official_paths(dataset_dir, dataset, pdbid)
    site = oracle_receptor_site(str(gt_rec), str(gt_lig), cutoff=native_cutoff)
    cache[pdbid] = {
        'site': site,
        'gt_rec': str(gt_rec),
        'gt_lig': str(gt_lig),
        'n_site': len(site),
    }
    if verbose:
        print(f'  {pdbid}: oracle site residues = {len(site)} '
              f'(native cutoff {native_cutoff} A)')
    return cache[pdbid]


def _success_at_k(ranked_success_flags, k):
    return int(any(ranked_success_flags[:k]))


def _is_success(row) -> bool:
    if row.get('success') not in ('', None):
        try:
            return int(float(row['success'])) == 1
        except ValueError:
            pass
    dockq = row.get('dockq')
    if dockq in ('', None):
        return False
    try:
        return float(dockq) >= 0.23
    except ValueError:
        return False


def _summarize(rows_by_target, site_frac_min):
    """Per-target Success@N before/after site filter (TraDock score order)."""
    summary = []
    for pdbid, rows in sorted(rows_by_target.items()):
        valid = [
            r for r in rows
            if r.get('status', 'done') == 'done'
            and r.get('score') not in ('', None)
            and r.get('dockq') not in ('', None)
        ]
        if not valid:
            continue
        all_ranked = sorted(valid, key=lambda r: float(r['score']), reverse=True)
        kept = [r for r in all_ranked if keep_by_site_frac(float(r['site_frac']), site_frac_min)]
        paper_all = sorted(valid, key=lambda r: int(float(r.get('input_rank') or 10**9)))
        paper_kept = [r for r in paper_all if keep_by_site_frac(float(r['site_frac']), site_frac_min)]
        oracle_all = sorted(valid, key=lambda r: -float(r['dockq']))
        oracle_kept = [r for r in oracle_all if keep_by_site_frac(float(r['site_frac']), site_frac_min)]

        def flags(seq):
            return [_is_success(r) for r in seq]

        row = {
            'target': pdbid,
            'n_valid': len(valid),
            'n_kept': len(kept),
            'kept_pct': 100.0 * len(kept) / max(1, len(valid)),
            'n_site': valid[0].get('n_site', ''),
            'n_success_all': sum(flags(valid)),
            'n_success_kept': sum(flags(kept)),
        }
        fa, fk = flags(all_ranked), flags(kept)
        pa, pk = flags(paper_all), flags(paper_kept)
        oa, ok = flags(oracle_all), flags(oracle_kept)
        for k in SUCCESS_KS:
            row[f'tradock_success@{k}'] = _success_at_k(fa, k)
            row[f'tradock_kept_success@{k}'] = _success_at_k(fk, k) if kept else 0
            row[f'paper_success@{k}'] = _success_at_k(pa, k)
            row[f'paper_kept_success@{k}'] = _success_at_k(pk, k) if paper_kept else 0
            row[f'oracle_success@{k}'] = _success_at_k(oa, k)
            row[f'oracle_kept_success@{k}'] = _success_at_k(ok, k) if oracle_kept else 0
        summary.append(row)
    return summary


def _print_aggregate(summary, site_frac_min):
    n = len(summary) or 1

    def mean(key):
        return sum(float(r.get(key) or 0) for r in summary) / n

    print(f'\n=== site filter summary (n_targets={len(summary)}, '
          f'site_frac>={site_frac_min}) ===')
    print(f'mean kept: {mean("kept_pct"):.1f}%  '
          f'({mean("n_kept"):.0f}/{mean("n_valid"):.0f} poses)')
    for k in (1, 10, 100):
        print(
            f'T@{k}: all={100 * mean(f"tradock_success@{k}"):.1f}%  '
            f'kept={100 * mean(f"tradock_kept_success@{k}"):.1f}%  | '
            f'P@{k}: all={100 * mean(f"paper_success@{k}"):.1f}%  '
            f'kept={100 * mean(f"paper_kept_success@{k}"):.1f}%  | '
            f'O@{k}: all={100 * mean(f"oracle_success@{k}"):.1f}%  '
            f'kept={100 * mean(f"oracle_kept_success@{k}"):.1f}%'
        )


def run_from_detail(args, ev):
    paper_root = Path(args.paper_root)
    dataset_dir = paper_root / 'dataset' / args.dataset
    allow = _target_allow(args)
    site_cache = {}
    out_rows = []
    by_target = defaultdict(list)
    seen_targets = []

    with open(args.detail_csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            pdbid = row.get('target') or row.get('pdbid')
            if not pdbid:
                continue
            if allow is not None and pdbid not in allow:
                continue
            if pdbid not in by_target:
                if args.limit and len(seen_targets) >= args.limit:
                    continue
                seen_targets.append(pdbid)

            status = row.get('status', 'done')
            pred_rec = row.get('pred_rec')
            pred_lig = row.get('pred_lig')
            new = dict(row)
            try:
                info = _load_site_cache(
                    ev, dataset_dir, args.dataset, pdbid,
                    args.native_cutoff, site_cache, args.verbose,
                )
                new['n_site'] = info['n_site']
                if status != 'done' or not pred_rec or not pred_lig:
                    raise ValueError(row.get('message') or 'missing pose paths')
                if not Path(pred_rec).exists() or not Path(pred_lig).exists():
                    raise FileNotFoundError(f'missing pose pdb: {pred_rec} / {pred_lig}')
                stats = decoy_site_overlap(
                    pred_rec, pred_lig, info['site'],
                    contact_cutoff=args.contact_cutoff,
                    match_by_resseq=args.match_by_resseq,
                )
                new.update(stats)
                new['keep'] = int(keep_by_site_frac(stats['site_frac'], args.site_frac_min))
                new['site_status'] = 'done'
                new['site_message'] = ''
            except Exception as exc:
                new.setdefault('n_site', '')
                new['n_hit'] = ''
                new['site_frac'] = ''
                new['n_lig_near_site'] = ''
                new['lig_near_frac'] = ''
                new['keep'] = 0
                new['site_status'] = 'error'
                new['site_message'] = str(exc)[:200]

            out_rows.append(new)
            by_target[pdbid].append(new)

    extra = [
        'n_site', 'n_hit', 'site_frac', 'n_lig_near_site', 'lig_near_frac',
        'n_site_on_decoy_rec', 'keep', 'site_status', 'site_message',
    ]
    fields = fieldnames + [c for c in extra if c not in fieldnames]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(out_rows)

    summary = _summarize(by_target, args.site_frac_min)
    summary_path = args.out[:-4] + '.summary.csv' if args.out.endswith('.csv') else args.out + '.summary.csv'
    if summary:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        _print_aggregate(summary, args.site_frac_min)
        print(f'wrote {args.out}')
        print(f'wrote {summary_path}')
    else:
        print(f'wrote {args.out} (no summarizable targets)')
    return 0


def run_from_paper(args, ev):
    paper_root = Path(args.paper_root)
    dataset_dir = paper_root / 'dataset' / args.dataset
    targets = ev.read_targets(dataset_dir, args.dataset)
    allow = _target_allow(args)
    if allow is not None:
        targets = [t for t in targets if t.get('pdb') in allow]
    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    pose_models = ev.discover_pose_models(paper_root, args.dataset, args.pose_prefix)

    site_cache = {}
    out_rows = []
    by_target = defaultdict(list)

    for t in targets:
        pdbid = t['pdb']
        print(f'[{pdbid}] discovering poses...', flush=True)
        info = _load_site_cache(
            ev, dataset_dir, args.dataset, pdbid,
            args.native_cutoff, site_cache, True,
        )
        gt_rec, gt_lig, default_pred_rec, condition = ev.official_paths(
            dataset_dir, args.dataset, pdbid,
        )
        cands = ev.collect_candidates(
            paper_root, args.dataset, pdbid, pose_models, default_pred_rec,
            max_poses=args.max_poses,
        )
        print(f'[{pdbid}] poses={len(cands)}  site_res={info["n_site"]}', flush=True)
        for cand in cands:
            row = {
                'dataset': args.dataset,
                'condition': condition,
                'target': pdbid,
                'pose_model': cand.pose_model,
                'input_rank': cand.input_rank,
                'pred_rec': str(cand.rec_path),
                'pred_lig': str(cand.lig_path),
                'score': '',
                'dockq': '',
                'success': '',
                'status': 'done',
                'n_site': info['n_site'],
            }
            try:
                stats = decoy_site_overlap(
                    str(cand.rec_path), str(cand.lig_path), info['site'],
                    contact_cutoff=args.contact_cutoff,
                    match_by_resseq=args.match_by_resseq,
                )
                row.update(stats)
                row['keep'] = int(keep_by_site_frac(stats['site_frac'], args.site_frac_min))
                row['site_status'] = 'done'
                row['site_message'] = ''
            except Exception as exc:
                row['site_frac'] = ''
                row['keep'] = 0
                row['site_status'] = 'error'
                row['site_message'] = str(exc)[:200]
            out_rows.append(row)
            by_target[pdbid].append(row)

        n_keep = sum(1 for r in by_target[pdbid] if r.get('keep') == 1)
        print(f'  kept {n_keep}/{len(by_target[pdbid])} '
              f'(site_frac>={args.site_frac_min})')

    fields = [
        'dataset', 'condition', 'target', 'pose_model', 'input_rank',
        'pred_rec', 'pred_lig', 'n_site', 'n_hit', 'site_frac',
        'n_lig_near_site', 'lig_near_frac', 'n_site_on_decoy_rec',
        'keep', 'site_status', 'site_message',
        'score', 'dockq', 'success', 'status',
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'wrote {args.out}')
    print('Note: without --detail-csv there is no TraDock score; '
          'use detail mode to compare Success@N.')
    return 0


def main() -> int:
    args = parse_args()
    ev = _import_eval_helpers()
    if args.detail_csv:
        return run_from_detail(args, ev)
    return run_from_paper(args, ev)


if __name__ == '__main__':
    raise SystemExit(main())
