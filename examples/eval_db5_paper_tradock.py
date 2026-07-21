"""
Evaluate TraDock on the official DB5 apo/holo data from:

  Revisiting Protein-protein Docking: A Systematic Evaluation Framework
  https://github.com/Yukki1777/PPCBench

This script does not generate docking poses. It reranks the paper-provided
candidate poses with TraDock and computes the paper-aligned metrics:
C-RMSD, I-RMSD, DockQ, and Top-N success with DockQ >= 0.23.

Expected paper layout:
  <paper_root>/dataset/DB5/DB5.json
  <paper_root>/dataset/DB5/structures/<PDB>/<PDB>_r_b.pdb
  <paper_root>/dataset/DB5/structures/<PDB>/<PDB>_l_b.pdb
  <paper_root>/dataset/DB5-u/DB5-u.json
  <paper_root>/dataset/DB5-u/structures/<PDB>/<PDB>_r_b_f.pdb
  <paper_root>/dataset/DB5-u/structures/<PDB>/<PDB>_l_b_f.pdb
  <paper_root>/dataset/DB5-u/structures/<PDB>/<PDB>_r_u_f.pdb
  <paper_root>/results/<dataset>/<pose_model>/<PDB>/<PDB>_<pose_model>_id.pdb

For Top-N reranking, pass pose model directories in rank order, e.g.
  --pose_models hdock_1,hdock_2,hdock_3,hdock_4,hdock_5

To rerank ALL decoys on disk for a method:
  --all_decoys --pose_prefix hdock
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from Bio import BiopythonWarning, PDB

# HDock/LightDock PDBs often omit occupancy; Bio.PDB spam is not actionable.
warnings.filterwarnings('ignore', category=BiopythonWarning)
warnings.filterwarnings('ignore', message='.*Missing occupancy.*')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.surface_gen import pdb_to_surface_ply
from transformerdock.models import DeepDock_PPI, ppi_score
from transformerdock.utils.data import prepare_complex


DOCKQ_SUCCESS = 0.23


@dataclass
class Candidate:
    pose_model: str
    input_rank: int
    rec_path: Path
    lig_path: Path


def parse_csv_list(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def classify_dockq(dockq):
    if dockq >= 0.80:
        return 'high'
    if dockq >= 0.49:
        return 'medium'
    if dockq >= 0.23:
        return 'acceptable'
    return 'unacceptable'


def load_paper_eval(paper_root):
    paper_root = Path(paper_root).resolve()
    if not (paper_root / 'evaluate').is_dir():
        raise FileNotFoundError(f'missing official evaluate/ under {paper_root}')
    sys.path.insert(0, str(paper_root))

    from evaluate.dataset import test_complex_process, BaseComplex, CA_INDEX
    from evaluate.dockq import dockQ
    from evaluate.rmsd import compute_crmsd, compute_irmsd, protein_surface_intersection

    dockq_exec = paper_root / 'evaluate' / 'DockQ' / 'DockQ.py'
    if not dockq_exec.exists():
        raise FileNotFoundError(
            f'missing official DockQ v1.0 at {dockq_exec}; '
            'run: cd <paper_root>/evaluate && '
            'git clone --branch v1.0 https://github.com/bjornwallner/DockQ.git'
        )

    return SimpleNamespace(
        test_complex_process=test_complex_process,
        BaseComplex=BaseComplex,
        CA_INDEX=CA_INDEX,
        dockQ=dockQ,
        compute_crmsd=compute_crmsd,
        compute_irmsd=compute_irmsd,
        protein_surface_intersection=protein_surface_intersection,
    )


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
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
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型加载: {checkpoint_path}  epoch={ckpt.get('epoch', '?')}  in_channels={in_channels}")
    return model, args.get('dist_threshold', 10.0), in_channels


def read_targets(dataset_dir, dataset):
    json_path = dataset_dir / f'{dataset}.json'
    if not json_path.exists():
        raise FileNotFoundError(f'missing dataset json: {json_path}')
    targets = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    return targets


def official_paths(dataset_dir, dataset, pdbid):
    struct_dir = dataset_dir / 'structures' / pdbid
    if dataset == 'DB5':
        gt_rec = struct_dir / f'{pdbid}_r_b.pdb'
        gt_lig = struct_dir / f'{pdbid}_l_b.pdb'
        pred_rec = gt_rec
        condition = 'holo'
    elif dataset == 'DB5-u':
        gt_rec = struct_dir / f'{pdbid}_r_b_f.pdb'
        gt_lig = struct_dir / f'{pdbid}_l_b_f.pdb'
        pred_rec = struct_dir / f'{pdbid}_r_u_f.pdb'
        condition = 'apo'
    elif dataset.endswith('-g-u'):
        gt_rec = struct_dir / f'{pdbid}_r_b_g.pdb'
        gt_lig = struct_dir / f'{pdbid}_l_b_g.pdb'
        pred_rec = gt_rec
        condition = 'apo_geodock'
    else:
        gt_rec = struct_dir / f'{pdbid}_r.pdb'
        gt_lig = struct_dir / f'{pdbid}_l.pdb'
        pred_rec = gt_rec
        condition = 'other'

    for path in (gt_rec, gt_lig, pred_rec):
        if not path.exists():
            raise FileNotFoundError(f'missing required structure: {path}')
    return gt_rec, gt_lig, pred_rec, condition


def monomer2complex(monomers, save_path):
    parser = PDB.PDBParser(QUIET=True)
    writer = PDB.PDBIO()
    model = PDB.Model.Model('annoym')
    for monomer_path in monomers:
        structure = parser.get_structure('annoym', str(monomer_path))
        for pdb_model in structure:
            for chain in pdb_model:
                model.add(chain)
    writer.set_structure(model)
    writer.save(str(save_path))


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


SUCCESS_KS = (1, 3, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000)


def pose_model_rank(name):
    """Extract trailing integer rank from names like hdock_12 / lightdock_3."""
    try:
        return int(str(name).rsplit('_', 1)[-1])
    except ValueError:
        return 10 ** 9


def discover_pose_models(paper_root, dataset, prefix):
    """List all pose dirs under results/<dataset>/ matching prefix or prefix_*."""
    root = Path(paper_root) / 'results' / dataset
    if not root.is_dir():
        raise FileNotFoundError(f'missing results directory: {root}')
    prefix = str(prefix).rstrip('_')
    names = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name == prefix or name.startswith(prefix + '_'):
            names.append(name)
    names = sorted(names, key=pose_model_rank)
    if not names:
        raise FileNotFoundError(
            f'no pose dirs matching prefix={prefix!r} under {root}'
        )
    return names


def collect_candidates(paper_root, dataset, pdbid, pose_models, default_pred_rec):
    out = []
    for i, pose_model in enumerate(pose_models, 1):
        target_dir = paper_root / 'results' / dataset / pose_model / pdbid
        lig_path = first_existing([
            target_dir / f'{pdbid}_{pose_model}_id.pdb',
            target_dir / f'{pdbid}_{pose_model}.pdb',
        ])
        if lig_path is None:
            continue
        rec_path = first_existing([
            target_dir / f'{pdbid}_{pose_model}_rec_id.pdb',
            target_dir / f'{pdbid}_{pose_model}_rec.pdb',
        ]) or default_pred_rec
        rank = pose_model_rank(pose_model)
        if rank >= 10 ** 9:
            rank = i
        out.append(Candidate(pose_model, rank, rec_path, lig_path))
    # Keep original docking rank order for Paper@K.
    out.sort(key=lambda c: c.input_rank)
    return out


def _surface_worker(job):
    """Process-pool worker: (pdb_path, out_ply, voxel_size) -> (pdb_path, out_ply, ok)."""
    pdb_path, out_ply, voxel_size = job
    ok = pdb_to_surface_ply(str(pdb_path), str(out_ply), voxel_size=voxel_size)
    return str(pdb_path), str(out_ply), bool(ok and Path(out_ply).exists())


class SurfaceCache:
    def __init__(self, tmpdir, voxel_size, n_workers=0):
        self.tmpdir = Path(tmpdir)
        self.voxel_size = voxel_size
        self.n_workers = max(0, int(n_workers))
        self.cache = {}
        self._idx = 0

    def _next_out(self):
        out = self.tmpdir / f'surf_{self._idx}.ply'
        self._idx += 1
        return out

    def get(self, pdb_path):
        pdb_path = Path(pdb_path).resolve()
        key = str(pdb_path)
        if key in self.cache:
            return self.cache[key]
        out = self._next_out()
        ok = pdb_to_surface_ply(str(pdb_path), str(out), voxel_size=self.voxel_size)
        if not ok or not out.exists():
            raise RuntimeError(f'surface generation failed: {pdb_path}')
        self.cache[key] = out
        return out

    def prefetch(self, pdb_paths):
        """Generate surfaces for unique PDBs, optionally in parallel."""
        jobs = []
        for pdb_path in pdb_paths:
            pdb_path = Path(pdb_path).resolve()
            key = str(pdb_path)
            if key in self.cache:
                continue
            out = self._next_out()
            jobs.append((str(pdb_path), str(out), self.voxel_size))

        if not jobs:
            return

        if self.n_workers <= 1 or len(jobs) == 1:
            for pdb_path, out_ply, voxel_size in jobs:
                ok = pdb_to_surface_ply(pdb_path, out_ply, voxel_size=voxel_size)
                if not ok or not Path(out_ply).exists():
                    raise RuntimeError(f'surface generation failed: {pdb_path}')
                self.cache[pdb_path] = Path(out_ply)
            return

        workers = min(self.n_workers, len(jobs))
        # Use spawn: forking after CUDA init deadlocks / stalls ProcessPool workers.
        ctx = mp.get_context('spawn')
        done = 0
        print(f'  surface prefetch: {len(jobs)} pdbs, workers={workers}', flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_surface_worker, job) for job in jobs]
            for fut in as_completed(futures):
                pdb_path, out_ply, ok = fut.result()
                if not ok:
                    raise RuntimeError(f'surface generation failed: {pdb_path}')
                self.cache[pdb_path] = Path(out_ply)
                done += 1
                if done == 1 or done % 50 == 0 or done == len(jobs):
                    print(f'  surface prefetch: {done}/{len(jobs)}', flush=True)


@torch.no_grad()
def score_pair(model, dist_threshold, device, rec_ply, lig_ply,
               score_type='mdn', fusion_alpha=0.5, in_channels=11, use_amp=True):
    rec_data, lig_data = prepare_complex(str(rec_ply), str(lig_ply), in_channels=in_channels)
    rec_data = rec_data.to(device)
    lig_data = lig_data.to(device)
    amp_on = bool(use_amp) and str(device).startswith('cuda')
    with torch.cuda.amp.autocast(enabled=amp_on):
        pi, sigma, mu, dist, _, pred_energy = model(rec_data, lig_data)
    # Keep scoring math in FP32 for stability.
    mdn_score = ppi_score(pi.float(), sigma.float(), mu.float(), dist.float(), dist_threshold)
    energy_score = float(pred_energy.float().item())
    if score_type == 'energy':
        return energy_score
    if score_type == 'fusion':
        return fusion_alpha * mdn_score + (1 - fusion_alpha) * energy_score
    return mdn_score


def compute_official_metrics(paper_eval, gt, cand, tmpdir, pdbid, rchain_id, lchain_id):
    gt_rec, gt_lig, gt_X, seg = gt
    gt_complex = tmpdir / f'{pdbid}_gt.pdb'
    pred_complex = tmpdir / f'{pdbid}_{cand.pose_model}_predicted.pdb'

    monomer2complex([gt_rec, gt_lig], gt_complex)
    monomer2complex([cand.rec_path, cand.lig_path], pred_complex)
    dock_X = paper_eval.BaseComplex.from_pdb(
        str(pred_complex), str(gt_lig)
    ).ligand_coord()[:, paper_eval.CA_INDEX]

    if dock_X.shape[0] != gt_X.shape[0]:
        raise RuntimeError(
            f'coordinate dimension mismatch: pred={dock_X.shape[0]} gt={gt_X.shape[0]}'
        )

    dock_X_re = torch.tensor(dock_X[seg == 0])
    dock_X_li = torch.tensor(dock_X[seg == 1])
    crmsd = paper_eval.compute_crmsd(dock_X, gt_X, aligned=False)
    irmsd = paper_eval.compute_irmsd(dock_X, gt_X, seg, aligned=False)
    intersection = float(
        paper_eval.protein_surface_intersection(dock_X_re, dock_X_li).relu().mean()
        + paper_eval.protein_surface_intersection(dock_X_li, dock_X_re).relu().mean()
    )
    dockq = paper_eval.dockQ(
        str(pred_complex), str(gt_complex),
        rchain_id=rchain_id, lchain_id=lchain_id,
    )

    for path in (gt_complex, pred_complex):
        try:
            path.unlink()
        except OSError:
            pass

    return {
        'crmsd': crmsd,
        'irmsd': irmsd,
        'dockq': dockq,
        'intersection': intersection,
        'classification': classify_dockq(dockq),
        'success': int(dockq >= DOCKQ_SUCCESS),
    }


DETAIL_FIELDS = [
    'dataset', 'condition', 'target_index', 'target_total', 'target',
    'pose_model', 'input_rank', 'tradock_rank',
    'score', 'dockq', 'irmsd', 'crmsd', 'intersection',
    'classification', 'success', 'status', 'message',
    'pred_rec', 'pred_lig',
]

SUMMARY_FIELDS = [
    'dataset', 'condition', 'target_index', 'target_total', 'target',
    'status', 'message', 'elapsed_sec',
    'n_poses', 'n_valid', 'n_success_available',
    'best_available_dockq',
    'top1_pose_model', 'top1_score', 'top1_dockq', 'top1_irmsd',
    'top1_crmsd', 'top1_classification',
]
for _k in SUCCESS_KS:
    SUMMARY_FIELDS.extend([
        f'success@{_k}',
        f'tradock_success@{_k}',
        f'paper_success@{_k}',
        f'oracle_success@{_k}',
    ])
SUMMARY_FIELDS.extend([
    'paper_top1_pose_model', 'paper_top1_dockq', 'paper_top1_irmsd',
    'paper_top1_crmsd', 'paper_top1_classification',
])


def _summary_path(out_path):
    if out_path.endswith('.csv'):
        return out_path[:-4] + '.summary.csv'
    return out_path + '.summary.csv'


def _aggregate_path(summary_path):
    if summary_path.endswith('.csv'):
        return summary_path[:-4] + '.aggregate.csv'
    return summary_path + '.aggregate.csv'


def write_aggregate(summary_path, pose_models):
    rows = []
    with open(summary_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None

    n_targets = len(rows)
    done = [r for r in rows if r.get('status') == 'done']
    valid_top1 = [
        r for r in done
        if r.get('top1_dockq') not in ('', None)
    ]

    def fval(row, key, default=0.0):
        try:
            return float(row.get(key, default) or default)
        except ValueError:
            return default

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    def median(values):
        return float(np.median(values)) if values else 0.0

    out = {
        'dataset': rows[0].get('dataset', ''),
        'condition': rows[0].get('condition', ''),
        'pose_models': ','.join(pose_models),
        'success_threshold': DOCKQ_SUCCESS,
        'n_targets': n_targets,
        'n_done': len(done),
        'n_missing_or_failed': n_targets - len(done),
        'n_targets_with_success_available': sum(
            int(fval(r, 'n_success_available')) > 0 for r in rows
        ),
        'mean_best_available_dockq': mean([fval(r, 'best_available_dockq') for r in done]),
        'mean_top1_dockq': mean([fval(r, 'top1_dockq') for r in valid_top1]),
        'median_top1_dockq': median([fval(r, 'top1_dockq') for r in valid_top1]),
        'mean_top1_irmsd': mean([fval(r, 'top1_irmsd') for r in valid_top1]),
        'median_top1_irmsd': median([fval(r, 'top1_irmsd') for r in valid_top1]),
        'mean_top1_crmsd': mean([fval(r, 'top1_crmsd') for r in valid_top1]),
        'median_top1_crmsd': median([fval(r, 'top1_crmsd') for r in valid_top1]),
        'top1_unacceptable_rate': mean([
            int(r.get('top1_classification') == 'unacceptable') for r in valid_top1
        ]),
        'top1_acceptable+_rate': mean([
            int(r.get('top1_classification') in {'acceptable', 'medium', 'high'})
            for r in valid_top1
        ]),
        'top1_medium+_rate': mean([
            int(r.get('top1_classification') in {'medium', 'high'})
            for r in valid_top1
        ]),
        'top1_high_rate': mean([
            int(r.get('top1_classification') == 'high') for r in valid_top1
        ]),
    }
    for k in SUCCESS_KS:
        tradock_key = f'tradock_success@{k}'
        paper_key = f'paper_success@{k}'
        oracle_key = f'oracle_success@{k}'
        tradock_rate = sum(int(fval(r, tradock_key, fval(r, f'success@{k}'))) for r in rows) / max(n_targets, 1)
        paper_rate = sum(int(fval(r, paper_key)) for r in rows) / max(n_targets, 1)
        oracle_rate = sum(int(fval(r, oracle_key)) for r in rows) / max(n_targets, 1)
        out[f'success@{k}'] = tradock_rate
        out[tradock_key] = tradock_rate
        out[paper_key] = paper_rate
        out[oracle_key] = oracle_rate
        out[f'delta_tradock_minus_paper@{k}'] = tradock_rate - paper_rate

    agg_path = _aggregate_path(summary_path)
    with open(agg_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        writer.writeheader()
        writer.writerow(out)
    return agg_path, out


def fmt_float(value):
    if value in ('', None):
        return ''
    try:
        return f'{float(value):.6f}'
    except (TypeError, ValueError):
        return value


def evaluate_target(args, paper_eval, model, dist_threshold, in_channels, device,
                    target, index, total, pose_models, tmpdir):
    t0 = time.time()
    pdbid = target['pdb']
    dataset_dir = Path(args.paper_root) / 'dataset' / args.dataset
    gt_rec, gt_lig, default_pred_rec, condition = official_paths(
        dataset_dir, args.dataset, pdbid
    )
    candidates = collect_candidates(
        Path(args.paper_root), args.dataset, pdbid, pose_models, default_pred_rec
    )
    if not candidates:
        return [], {
            'dataset': args.dataset,
            'condition': condition,
            'target_index': index,
            'target_total': total,
            'target': pdbid,
            'status': 'missing_pose',
            'message': 'no candidate pose found in paper results',
            'elapsed_sec': f'{time.time() - t0:.3f}',
        }

    cache = SurfaceCache(tmpdir, args.voxel_size, n_workers=getattr(args, 'n_workers', 0))
    # Prefetch unique surfaces (CPU-heavy); receptor is often shared across poses.
    uniq_pdbs = []
    seen = set()
    for cand in candidates:
        for path in (cand.rec_path, cand.lig_path):
            key = str(Path(path).resolve())
            if key not in seen:
                seen.add(key)
                uniq_pdbs.append(path)
    cache.prefetch(uniq_pdbs)

    gt_data = paper_eval.test_complex_process(str(gt_lig), str(gt_rec))
    gt = (
        gt_rec,
        gt_lig,
        gt_data['X'][:, paper_eval.CA_INDEX].numpy(),
        gt_data['Seg'].numpy(),
    )

    detail_rows = []
    for cand in candidates:
        row = {
            'dataset': args.dataset,
            'condition': condition,
            'target_index': index,
            'target_total': total,
            'target': pdbid,
            'pose_model': cand.pose_model,
            'input_rank': cand.input_rank,
            'pred_rec': str(cand.rec_path),
            'pred_lig': str(cand.lig_path),
            'status': 'done',
            'message': '',
        }
        try:
            rec_ply = cache.get(cand.rec_path)
            lig_ply = cache.get(cand.lig_path)
            row['score'] = score_pair(
                model, dist_threshold, device, rec_ply, lig_ply,
                score_type=args.score_type, fusion_alpha=args.fusion_alpha,
                in_channels=in_channels,
                use_amp=getattr(args, 'amp', True),
            )
            metrics = compute_official_metrics(
                paper_eval, gt, cand, tmpdir, pdbid,
                target['rchain'], target['lchain'],
            )
            row.update(metrics)
        except Exception as exc:
            row['status'] = 'error'
            row['message'] = str(exc)
            row['score'] = ''
        detail_rows.append(row)

    valid = [
        r for r in detail_rows
        if r.get('status') == 'done'
        and np.isfinite(float(r.get('score')))
        and r.get('dockq') not in ('', None)
    ]
    ranked = sorted(valid, key=lambda r: float(r['score']), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row['tradock_rank'] = rank

    if not ranked:
        summary = {
            'dataset': args.dataset,
            'condition': condition,
            'target_index': index,
            'target_total': total,
            'target': pdbid,
            'status': 'no_valid_pose',
            'message': 'all candidate scoring/metric jobs failed',
            'elapsed_sec': f'{time.time() - t0:.3f}',
            'n_poses': len(candidates),
            'n_valid': 0,
        }
        return detail_rows, summary

    paper_ranked = sorted(valid, key=lambda r: int(r['input_rank']))
    oracle_ranked = sorted(valid, key=lambda r: float(r['dockq']), reverse=True)

    def success_at(rows, k):
        return int(any(float(r['dockq']) >= DOCKQ_SUCCESS for r in rows[:min(k, len(rows))]))

    top1 = ranked[0]
    paper_top1 = paper_ranked[0]
    n_success = sum(int(float(r['dockq']) >= DOCKQ_SUCCESS) for r in valid)
    summary = {
        'dataset': args.dataset,
        'condition': condition,
        'target_index': index,
        'target_total': total,
        'target': pdbid,
        'status': 'done',
        'message': '',
        'elapsed_sec': f'{time.time() - t0:.3f}',
        'n_poses': len(candidates),
        'n_valid': len(valid),
        'n_success_available': n_success,
        'best_available_dockq': max(float(r['dockq']) for r in valid),
        'top1_pose_model': top1['pose_model'],
        'top1_score': top1['score'],
        'top1_dockq': top1['dockq'],
        'top1_irmsd': top1['irmsd'],
        'top1_crmsd': top1['crmsd'],
        'top1_classification': top1['classification'],
        'paper_top1_pose_model': paper_top1['pose_model'],
        'paper_top1_dockq': paper_top1['dockq'],
        'paper_top1_irmsd': paper_top1['irmsd'],
        'paper_top1_crmsd': paper_top1['crmsd'],
        'paper_top1_classification': paper_top1['classification'],
    }
    for k in SUCCESS_KS:
        summary[f'tradock_success@{k}'] = success_at(ranked, k)
        summary[f'paper_success@{k}'] = success_at(paper_ranked, k)
        summary[f'oracle_success@{k}'] = success_at(oracle_ranked, k)
        summary[f'success@{k}'] = summary[f'tradock_success@{k}']
    return detail_rows, summary


def main():
    parser = argparse.ArgumentParser(
        description='TraDock reranking on official DB5 apo/holo paper data'
    )
    parser.add_argument('--paper_root', required=True,
                        help='Official PPCBench root containing dataset/ and results/')
    parser.add_argument('--dataset', required=True,
                        help='DB5 for holo, DB5-u for apo')
    parser.add_argument('--pose_models', default=None,
                        help='Comma-separated paper result dirs, e.g. hdock_1,...,hdock_5. '
                             'Use with --all_decoys/--pose_prefix to auto-discover all poses.')
    parser.add_argument('--pose_prefix', default=None,
                        help='Pose method prefix under results/<dataset>/, e.g. hdock or lightdock')
    parser.add_argument('--all_decoys', action='store_true',
                        help='Rerank ALL decoys matching --pose_prefix (ignore Top-N lists)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--score_type', default='mdn', choices=['mdn', 'energy', 'fusion'])
    parser.add_argument('--fusion_alpha', type=float, default=0.5)
    parser.add_argument('--voxel_size', type=float, default=3.5)
    parser.add_argument('--min_targets', type=int, default=218,
                        help='Fail unless full dataset has at least this many targets; ignored with --limit/--start_index/--end_index/--shard')
    parser.add_argument('--limit', type=int, default=None,
                        help='Debug only: evaluate first N targets')
    parser.add_argument('--start_index', type=int, default=0,
                        help='0-based inclusive start index into the target list')
    parser.add_argument('--end_index', type=int, default=None,
                        help='0-based exclusive end index into the target list')
    parser.add_argument('--shard', type=str, default=None,
                        help='Split targets across GPUs, e.g. 0/4 means shard 0 of 4')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='CPU workers for parallel surface generation (0/1 = serial)')
    parser.add_argument('--amp', dest='amp', action='store_true', default=True,
                        help='Use CUDA autocast FP16 for model forward (default: on)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable mixed precision')
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve()
    dataset_dir = paper_root / 'dataset' / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f'missing official dataset directory: {dataset_dir}')

    pose_models = parse_csv_list(args.pose_models) if args.pose_models else []
    if args.all_decoys:
        prefix = args.pose_prefix
        if not prefix:
            # Infer from first explicit model if provided: hdock_1 -> hdock
            if pose_models:
                prefix = pose_models[0].rsplit('_', 1)[0]
            else:
                raise ValueError('--all_decoys requires --pose_prefix (e.g. hdock or lightdock)')
        pose_models = discover_pose_models(paper_root, args.dataset, prefix)
        print(f'all_decoys: prefix={prefix}  discovered={len(pose_models)} '
              f'({pose_models[0]} ... {pose_models[-1]})')
    elif not pose_models:
        raise ValueError('provide --pose_models or --all_decoys --pose_prefix')

    targets = read_targets(dataset_dir, args.dataset)
    full_n = len(targets)
    if args.limit:
        targets = targets[:args.limit]
    if args.shard:
        try:
            shard_i, shard_n = args.shard.split('/', 1)
            shard_i, shard_n = int(shard_i), int(shard_n)
        except Exception as exc:
            raise ValueError('--shard must look like 0/4') from exc
        if shard_n <= 0 or shard_i < 0 or shard_i >= shard_n:
            raise ValueError(f'invalid --shard {args.shard}')
        targets = [t for idx, t in enumerate(targets) if idx % shard_n == shard_i]
    else:
        start = max(0, int(args.start_index))
        end = len(targets) if args.end_index is None else int(args.end_index)
        targets = targets[start:end]

    sliced = bool(args.limit or args.shard or args.start_index or args.end_index is not None)
    if not sliced and full_n < args.min_targets:
        raise RuntimeError(
            f'{args.dataset} has {full_n} targets, expected at least {args.min_targets}; '
            'use the full Zenodo paper data or set --min_targets explicitly'
        )

    paper_eval = load_paper_eval(paper_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}  amp={args.amp and device == "cuda"}  n_workers={args.n_workers}')
    print(f'paper_root: {paper_root}')
    print(f'dataset: {args.dataset}  targets={len(targets)}/{full_n}')
    if len(pose_models) > 12:
        print(f'pose_models: n={len(pose_models)}  '
              f'{pose_models[0]} ... {pose_models[-1]}')
    else:
        print(f'pose_models: {pose_models}')
    if args.shard:
        print(f'shard: {args.shard}')
    model, dist_threshold, in_channels = load_model(args.checkpoint, device)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    summary_path = _summary_path(args.out)
    tmpdir = Path(tempfile.mkdtemp(prefix='tradock_db5_paper_'))

    with open(args.out, 'w', newline='') as detail_f, open(summary_path, 'w', newline='') as summary_f:
        detail_writer = csv.DictWriter(detail_f, fieldnames=DETAIL_FIELDS, extrasaction='ignore')
        summary_writer = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
        detail_writer.writeheader()
        summary_writer.writeheader()
        detail_f.flush()
        summary_f.flush()

        try:
            for i, target in enumerate(targets, 1):
                pdbid = target['pdb']
                print(f'[{i}/{len(targets)}] {pdbid}', flush=True)
                try:
                    detail_rows, summary = evaluate_target(
                        args, paper_eval, model, dist_threshold, in_channels, device,
                        target, i, len(targets), pose_models, tmpdir,
                    )
                except Exception as exc:
                    summary = {
                        'dataset': args.dataset,
                        'condition': '',
                        'target_index': i,
                        'target_total': len(targets),
                        'target': pdbid,
                        'status': 'error',
                        'message': str(exc),
                        'elapsed_sec': '0.000',
                    }
                    detail_rows = []

                for row in detail_rows:
                    for key in ('score', 'dockq', 'irmsd', 'crmsd', 'intersection'):
                        if key in row:
                            row[key] = fmt_float(row[key])
                    detail_writer.writerow(row)
                detail_f.flush()

                for key in ('best_available_dockq', 'top1_score', 'top1_dockq',
                            'top1_irmsd', 'top1_crmsd', 'paper_top1_dockq',
                            'paper_top1_irmsd', 'paper_top1_crmsd'):
                    if key in summary:
                        summary[key] = fmt_float(summary[key])
                summary_writer.writerow(summary)
                summary_f.flush()

                if summary.get('status') == 'done':
                    print(
                        f"  score={float(summary['top1_score']):.3f} "
                        f"DockQ={float(summary['top1_dockq']):.3f} "
                        f"TraDock@1={summary.get('tradock_success@1', 0)} "
                        f"Paper@1={summary.get('paper_success@1', 0)}"
                    )
                else:
                    print(f"  {summary.get('status')}: {summary.get('message', '')}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    agg = write_aggregate(summary_path, pose_models)
    print(f'detail    -> {args.out}')
    print(f'summary   -> {summary_path}')
    if agg:
        agg_path, agg_row = agg
        print(f'aggregate -> {agg_path}')
        print(
            f"TraDock@1={100 * agg_row['tradock_success@1']:.2f}%  "
            f"Paper@1={100 * agg_row['paper_success@1']:.2f}%  "
            f"TraDock@5={100 * agg_row['tradock_success@5']:.2f}%  "
            f"Paper@5={100 * agg_row['paper_success@5']:.2f}%"
        )


if __name__ == '__main__':
    main()
