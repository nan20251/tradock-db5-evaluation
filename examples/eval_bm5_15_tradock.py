"""
Score Shirali BM5_15complexes (DeepRank-GNN hold-out) with TraDock.

Expected layout (from BM5_15complexes.zip):
  <data_dir>/PDBs/*.pdb          # e.g. 1PPE-ti5-it0-811.pdb, chains A/B
  <data_dir>/complexes_list.txt  # optional; {stem}_A_B per line

Labels (Shirali):
  BM5_scores&labels.csv columns: PDB,PID,...,CAPRI_quality,label

Usage:
  python examples/eval_bm5_15_tradock.py \\
      --data_dir /path/to/BM5_15complexes \\
      --labels_csv /path/to/BM5_scores&labels.csv \\
      --checkpoint Trained_models/pretrain_with_sasa/TransformerDock_best.chk \\
      --out results/bm5_15_tradock.csv \\
      --n_workers 16
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.surface_gen import pdb_to_surface_ply
from transformerdock.models import DeepDock_PPI, NO_INTERFACE_SCORE, ppi_score
from transformerdock.utils.data import prepare_complex


TOPKS = (1, 5, 10, 25, 100, 200)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True, help='Extracted BM5_15complexes root (contains PDBs/)')
    p.add_argument('--labels_csv', required=True, help='BM5_scores&labels.csv')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='Detail CSV path')
    p.add_argument('--rec_chain', default='A')
    p.add_argument('--lig_chain', default='B')
    p.add_argument('--score_type', default='mdn', choices=['mdn', 'energy', 'fusion', 'enhanced'])
    p.add_argument('--dist_threshold', type=float, default=-1.0,
                   help='Override checkpoint dist_threshold; <0 keeps checkpoint value')
    p.add_argument('--voxel_size', type=float, default=3.5)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit_targets', type=int, default=0, help='Debug: only first N targets')
    p.add_argument('--limit_per_target', type=int, default=0, help='Debug: only first N decoys per target')
    p.add_argument('--workdir', default='', help='Scratch dir for surfaces (default: temp)')
    return p.parse_args()


def load_labels(path):
    """PDB stem -> {pid, label, capri_quality, haddock, ...}"""
    by_stem = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row['PDB'].strip()
            by_stem[stem] = {
                'pid': row.get('PID', stem.split('-')[0]).strip(),
                'label': int(float(row.get('label', 0) or 0)),
                'capri_quality': row.get('CAPRI_quality', '').strip(),
                'haddock': _to_float(row.get('HADDOCK')),
                'piston': _to_float(row.get('PIsToN')),
                'deeprank_gnn': _to_float(row.get('DeepRank-GNN')),
            }
    return by_stem


def _to_float(v, default=float('nan')):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def discover_pdbs(data_dir: Path):
    pdb_dir = data_dir / 'PDBs'
    if not pdb_dir.is_dir():
        # allow passing PDBs/ directly
        pdb_dir = data_dir
    files = sorted(pdb_dir.glob('*.pdb'))
    if not files:
        raise FileNotFoundError(f'no *.pdb under {pdb_dir}')
    return files


def group_by_pid(pdb_files, labels):
    groups = defaultdict(list)
    missing_label = 0
    for path in pdb_files:
        stem = path.stem
        meta = labels.get(stem)
        if meta is None:
            missing_label += 1
            pid = stem.split('-')[0]
            meta = {'pid': pid, 'label': -1, 'capri_quality': 'unknown',
                    'haddock': float('nan'), 'piston': float('nan'), 'deeprank_gnn': float('nan')}
        groups[meta['pid']].append((path, stem, meta))
    return groups, missing_label


def split_ab(pdb_path, out_rec, out_lig, rec_chain='A', lig_chain='B'):
    rec, lig = [], []
    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and len(line) > 21:
                c = line[21]
                if c == rec_chain:
                    rec.append(line)
                elif c == lig_chain:
                    lig.append(line)
    if not rec or not lig:
        return False
    with open(out_rec, 'w') as f:
        f.writelines(rec)
        f.write('END\n')
    with open(out_lig, 'w') as f:
        f.writelines(lig)
        f.write('END\n')
    return True


def _surface_task(args):
    stem, pdb_path, workdir, rec_chain, lig_chain, voxel_size = args
    rec_pdb = os.path.join(workdir, f'{stem}_rec.pdb')
    lig_pdb = os.path.join(workdir, f'{stem}_lig.pdb')
    rec_ply = os.path.join(workdir, f'{stem}_rec.ply')
    lig_ply = os.path.join(workdir, f'{stem}_lig.ply')
    try:
        if not split_ab(pdb_path, rec_pdb, lig_pdb, rec_chain, lig_chain):
            return None
        ok_r = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
        ok_l = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)
        for p in (rec_pdb, lig_pdb):
            try:
                os.remove(p)
            except OSError:
                pass
        if not ok_r or not ok_l:
            return None
        return stem, rec_ply, lig_ply
    except Exception:
        return None


@torch.no_grad()
def score_ply(model, device, rec_ply, lig_ply, dist_threshold, score_type, in_channels):
    rec_data, lig_data = prepare_complex(rec_ply, lig_ply, in_channels=in_channels)
    rec_data = rec_data.to(device)
    lig_data = lig_data.to(device)
    amp_on = str(device).startswith('cuda')
    with torch.cuda.amp.autocast(enabled=amp_on):
        pi, sigma, mu, dist, _, pred_energy = model(rec_data, lig_data)
    pi, sigma, mu, dist = pi.float(), sigma.float(), mu.float(), dist.float()
    mdn = ppi_score(pi, sigma, mu, dist, dist_threshold)
    energy = float(pred_energy.float().item())
    if score_type == 'energy':
        score = energy
    elif score_type == 'fusion':
        score = mdn if mdn <= NO_INTERFACE_SCORE else 0.5 * mdn + 0.5 * energy
    elif score_type == 'enhanced':
        d = dist.squeeze(1)
        n_if = int((d < dist_threshold).sum().item())
        score = mdn if mdn <= NO_INTERFACE_SCORE else mdn + 0.5 * energy + 0.003 * float(np.log1p(n_if))
    else:
        score = mdn
    return float(score), float(mdn), energy


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    in_channels = int(args.get('in_channels', 11))
    model = DeepDock_PPI(
        in_channels=in_channels,
        hidden_dim=args.get('hidden_dim', 128),
        n_gaussians=args.get('n_gaussians', 10),
        n_transformer_blocks=args.get('n_tf_blocks', 6),
        transformer_heads=args.get('tf_heads', 4),
        use_global_attn=True,
        global_attn_layers=2,
        cross_attn_heads=args.get('cross_heads', 8),
        n_cross_attn_layers=args.get('n_cross_layers', 2),
        dist_threshold=args.get('dist_threshold', 10.0),
        dropout_rate=0.0,
    ).to(device)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state)
    model.eval()
    dist_threshold = float(args.get('dist_threshold', 10.0))
    print(f"模型加载: {checkpoint}  epoch={ckpt.get('epoch', '?')}  in_channels={in_channels}")
    return model, dist_threshold, in_channels


def auc_binary(y_true, y_score):
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def success_at_k(rows, k, higher_better=True, label_key='label', score_key='tradock'):
    """rows: list of dicts with label 0/1 and score."""
    valid = [r for r in rows if r.get(label_key, -1) in (0, 1) and np.isfinite(r.get(score_key, np.nan))]
    if not valid:
        return 0
    ranked = sorted(valid, key=lambda r: r[score_key], reverse=higher_better)
    top = ranked[: min(k, len(ranked))]
    return int(any(r[label_key] == 1 for r in top))


def summarize_target(pid, rows):
    labeled = [r for r in rows if r['label'] in (0, 1)]
    n_pos = sum(r['label'] == 1 for r in labeled)
    scores = [r['tradock'] for r in labeled if np.isfinite(r['tradock'])]
    labels = [r['label'] for r in labeled if np.isfinite(r['tradock'])]
    out = {
        'pid': pid,
        'n_models': len(rows),
        'n_labeled': len(labeled),
        'n_positive': n_pos,
        'n_scored': len(scores),
        'auc': auc_binary(labels, scores) if scores else float('nan'),
    }
    for k in TOPKS:
        out[f'success@{k}'] = success_at_k(labeled, k) if n_pos > 0 else 0
        # HADDOCK score is lower-better
        out[f'haddock_success@{k}'] = success_at_k(
            labeled, k, higher_better=False, score_key='haddock'
        ) if n_pos > 0 and any(np.isfinite(r.get('haddock', np.nan)) for r in labeled) else 0
    return out


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    labels = load_labels(args.labels_csv)
    pdb_files = discover_pdbs(data_dir)
    groups, missing_label = group_by_pid(pdb_files, labels)
    pids = sorted(groups.keys())
    if args.limit_targets > 0:
        pids = pids[: args.limit_targets]

    print(f'data_dir={data_dir}')
    print(f'pdbs={len(pdb_files)} targets={len(pids)} labels={len(labels)} missing_label={missing_label}')
    print(f'pids={pids}')

    device = torch.device(args.device)
    model, ckpt_dist, in_channels = load_model(args.checkpoint, device)
    dist_threshold = ckpt_dist if args.dist_threshold < 0 else float(args.dist_threshold)
    print(f'checkpoint={args.checkpoint} device={device} dist_threshold={dist_threshold}')

    work_root = args.workdir or tempfile.mkdtemp(prefix='bm5_15_tradock_')
    os.makedirs(work_root, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix('.summary.csv')

    detail_fields = [
        'pid', 'pdb', 'label', 'capri_quality', 'tradock', 'mdn', 'energy',
        'haddock', 'piston', 'deeprank_gnn', 'status',
    ]
    summary_rows = []
    t0 = time.time()

    with open(out_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=detail_fields)
        writer.writeheader()

        for ti, pid in enumerate(pids, 1):
            items = groups[pid]
            if args.limit_per_target > 0:
                items = items[: args.limit_per_target]
            print(f'[{ti}/{len(pids)}] {pid} n={len(items)}', flush=True)

            tasks = []
            for path, stem, meta in items:
                tasks.append((
                    stem, str(path), work_root,
                    args.rec_chain, args.lig_chain, args.voxel_size,
                ))

            ply_map = {}
            with ProcessPoolExecutor(max_workers=max(1, args.n_workers)) as pool:
                futs = {pool.submit(_surface_task, t): t[0] for t in tasks}
                for fut in as_completed(futs):
                    res = fut.result()
                    if res is None:
                        continue
                    stem, rec_ply, lig_ply = res
                    ply_map[stem] = (rec_ply, lig_ply)

            target_rows = []
            for path, stem, meta in items:
                row = {
                    'pid': pid,
                    'pdb': stem,
                    'label': meta['label'],
                    'capri_quality': meta['capri_quality'],
                    'tradock': float('nan'),
                    'mdn': float('nan'),
                    'energy': float('nan'),
                    'haddock': meta['haddock'],
                    'piston': meta['piston'],
                    'deeprank_gnn': meta['deeprank_gnn'],
                    'status': 'ok',
                }
                if stem not in ply_map:
                    row['status'] = 'surface_fail'
                    writer.writerow(row)
                    target_rows.append(row)
                    continue
                rec_ply, lig_ply = ply_map[stem]
                try:
                    score, mdn, energy = score_ply(
                        model, device, rec_ply, lig_ply,
                        dist_threshold, args.score_type, in_channels,
                    )
                    row['tradock'] = score
                    row['mdn'] = mdn
                    row['energy'] = energy
                except Exception as e:
                    row['status'] = f'score_fail:{type(e).__name__}'
                finally:
                    for p in (rec_ply, lig_ply):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                writer.writerow(row)
                target_rows.append(row)
                fout.flush()

            summary_rows.append(summarize_target(pid, target_rows))
            s = summary_rows[-1]
            print(
                f"  scored={s['n_scored']}/{s['n_models']} pos={s['n_positive']} "
                f"auc={s['auc']:.3f} S@1={s['success@1']} S@10={s['success@10']} S@100={s['success@100']}",
                flush=True,
            )

    # write per-target summary + aggregate (Shirali-style)
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        fields = list(summary_rows[0].keys()) if summary_rows else ['pid']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with_pos = [r for r in summary_rows if r['n_positive'] > 0]
    denom = len(with_pos) or 1
    agg = {
        'n_targets': len(summary_rows),
        'n_targets_with_positive': len(with_pos),
        'mean_auc': float(np.nanmean([r['auc'] for r in with_pos])) if with_pos else float('nan'),
        'elapsed_sec': time.time() - t0,
    }
    for k in TOPKS:
        agg[f'success@{k}'] = sum(r[f'success@{k}'] for r in with_pos) / denom
        agg[f'haddock_success@{k}'] = sum(r[f'haddock_success@{k}'] for r in with_pos) / denom

    agg_path = out_path.with_suffix('.aggregate.csv')
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)

    print('--- aggregate (targets with >=1 positive) ---')
    for k, v in agg.items():
        print(f'  {k}={v}')
    print(f'detail={out_path}')
    print(f'summary={summary_path}')
    print(f'aggregate={agg_path}')


if __name__ == '__main__':
    # Windows spawn safety for ProcessPoolExecutor
    main()
