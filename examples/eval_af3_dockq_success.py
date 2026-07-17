#!/usr/bin/env python3
"""
Report AlphaFold3 / OpenFold3 prediction success rates with DockQ.

Literature-style protocol (e.g. FoldBench / AF3 complex benchmarks):
  1. Collect all seed x sample models for each DB5 target.
  2. Rank by model confidence (ranking_score / sample_ranking_score).
  3. Compute DockQ of each model vs DB5 native complex.
  4. Success = DockQ >= 0.23 (acceptable / CAPRI).
  5. Aggregate:
       - Success@1_ranked : top-ranked model is successful
       - Success@K_ranked : any of top-K ranked models is successful
       - Success_oracle   : any sampled model is successful
       - CAPRI class histogram for the top-ranked model

This is NOT TraDock reranking. Use scripts/run_db5_openfold3_eval.sh for that.

Requires PPCBench evaluate/DockQ under --ppc_root (same as eval_db5_paper_tradock.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


DOCKQ_SUCCESS = 0.23


def parse_csv(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def normalize_chains(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    return list(str(value))


def classify_dockq(dockq):
    if not math.isfinite(dockq):
        return 'unknown'
    if dockq >= 0.80:
        return 'high'
    if dockq >= 0.49:
        return 'medium'
    if dockq >= 0.23:
        return 'acceptable'
    return 'unacceptable'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_targets(path):
    targets = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                targets[row['pdb']] = row
    return targets


def load_paper_eval(ppc_root):
    ppc_root = Path(ppc_root).resolve()
    evaluate = ppc_root / 'evaluate'
    if not evaluate.is_dir():
        raise FileNotFoundError(f'missing evaluate/ under {ppc_root}')
    sys.path.insert(0, str(ppc_root))
    from evaluate.dockq import dockQ

    dockq_exec = evaluate / 'DockQ' / 'DockQ.py'
    if not dockq_exec.exists():
        raise FileNotFoundError(
            f'missing DockQ v1.0 at {dockq_exec}; '
            f'run: cd {evaluate} && '
            'git clone --branch v1.0 https://github.com/bjornwallner/DockQ.git'
        )
    return SimpleNamespace(dockQ=dockQ)


def native_paths(ppc_root, dataset, pdbid):
    struct_dir = Path(ppc_root) / 'dataset' / dataset / 'structures' / pdbid
    if dataset == 'DB5':
        return struct_dir / f'{pdbid}_r_b.pdb', struct_dir / f'{pdbid}_l_b.pdb'
    if dataset == 'DB5-u':
        return struct_dir / f'{pdbid}_r_b_f.pdb', struct_dir / f'{pdbid}_l_b_f.pdb'
    raise ValueError('only DB5 and DB5-u supported')


def target_dirs_for(pred_root, pdbid):
    root = Path(pred_root)
    exact = root / pdbid
    if exact.is_dir():
        return [exact]
    sanitized = root / re.sub(r'[^A-Za-z0-9._-]+', '_', pdbid)
    if sanitized.is_dir():
        return [sanitized]
    return [
        p for p in root.iterdir()
        if p.is_dir() and pdbid.lower() in p.name.lower()
    ]


def find_sample_models(target_dir):
    paths = []
    for pattern in ('*_model.cif', '*_model.pdb'):
        paths.extend(Path(target_dir).rglob(pattern))
    sample_paths = [
        p for p in paths
        if re.search(r'seed[_-]\d+.*sample[_-]\d+', str(p))
    ]
    return sample_paths or paths


def parse_seed_sample(path):
    text = str(path)
    m = re.search(r'seed[_-](\d+).*?sample[_-](\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _first_finite(data, keys, default=math.nan):
    for key in keys:
        if key in data and finite(data[key]):
            return data[key]
    return default


def read_confidence(model_path):
    model_path = Path(model_path)
    stem = model_path.name
    for suffix in ('_model.cif', '_model.pdb', '.cif', '.pdb'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates = [
        model_path.with_name(stem + '_confidences_aggregated.json'),
        model_path.with_name(stem + '_summary_confidences.json'),
        model_path.parent / (stem + '_confidences_aggregated.json'),
        model_path.parent / (stem + '_summary_confidences.json'),
    ]
    scores = {
        'ranking_score': math.nan,
        'iptm': math.nan,
        'ptm': math.nan,
    }
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as handle:
            data = json.load(handle)
        scores['ranking_score'] = _first_finite(
            data, ('sample_ranking_score', 'ranking_score')
        )
        scores['iptm'] = _first_finite(data, ('iptm',))
        scores['ptm'] = _first_finite(data, ('ptm',))
        break
    seed, sample = parse_seed_sample(model_path)
    scores['seed'] = seed if seed is not None else ''
    scores['sample'] = sample if sample is not None else ''
    return scores


def load_model_structure(path):
    path = Path(path)
    if path.suffix.lower() in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return next(parser.get_structure(path.stem, str(path)).get_models())


def write_chain_subset(model, chain_ids, out_path):
    structure = Structure('subset')
    new_model = Model(0)
    structure.add(new_model)
    wanted = set(chain_ids)
    found = []
    for chain in model:
        if chain.id not in wanted:
            continue
        new_chain = Chain(chain.id)
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            new_chain.add(residue.copy())
        if len(list(new_chain.get_residues())) == 0:
            continue
        new_model.add(new_chain)
        found.append(chain.id)
    if set(found) != wanted:
        missing = wanted - set(found)
        raise RuntimeError(f'missing chains in prediction: {sorted(missing)}')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))


def monomer2complex(monomers, save_path):
    parser = PDBParser(QUIET=True)
    model = Model(0)
    for monomer_path in monomers:
        structure = parser.get_structure('x', str(monomer_path))
        for pdb_model in structure:
            for chain in pdb_model:
                model.add(chain)
    structure = Structure('complex')
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(save_path))


def dockq_one(paper_eval, pred_model_path, gt_rec, gt_lig, rec_chains, lig_chains, tmpdir, tag):
    pred_model = load_model_structure(pred_model_path)
    pred_rec = tmpdir / f'{tag}_pred_rec.pdb'
    pred_lig = tmpdir / f'{tag}_pred_lig.pdb'
    pred_complex = tmpdir / f'{tag}_pred.pdb'
    gt_complex = tmpdir / f'{tag}_gt.pdb'

    write_chain_subset(pred_model, rec_chains, pred_rec)
    write_chain_subset(pred_model, lig_chains, pred_lig)
    monomer2complex([pred_rec, pred_lig], pred_complex)
    monomer2complex([gt_rec, gt_lig], gt_complex)

    rchain_id = ''.join(rec_chains)
    lchain_id = ''.join(lig_chains)
    dockq = float(
        paper_eval.dockQ(
            str(pred_complex),
            str(gt_complex),
            rchain_id=rchain_id,
            lchain_id=lchain_id,
        )
    )
    return dockq


def rank_key(row):
    score = row['ranking_score']
    return (
        finite(score),
        score if finite(score) else -math.inf,
        str(row['source_model']),
    )


def success_in_topn(rows, k, threshold=DOCKQ_SUCCESS):
    return int(any(float(r['dockq']) >= threshold for r in rows[: min(k, len(rows))]))


def mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5', choices=['DB5', 'DB5-u'])
    parser.add_argument('--pred_root', required=True,
                        help='OpenFold3/AF3 output root (contains per-target dirs).')
    parser.add_argument('--out_detail', required=True)
    parser.add_argument('--out_targets', required=True)
    parser.add_argument('--out_summary', required=True)
    parser.add_argument('--targets', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dockq_threshold', type=float, default=DOCKQ_SUCCESS)
    parser.add_argument('--topks', default='1,5,10,25')
    args = parser.parse_args()

    ppc_root = Path(args.ppc_root).resolve()
    dataset_json = ppc_root / 'dataset' / args.dataset / f'{args.dataset}.json'
    targets = load_targets(dataset_json)
    selected = [x.strip().upper() for x in parse_csv(args.targets)]
    if not selected:
        selected = sorted(targets)
    if args.limit:
        selected = selected[: args.limit]
    topks = [int(x) for x in parse_csv(args.topks)]
    thr = float(args.dockq_threshold)

    paper_eval = load_paper_eval(ppc_root)
    detail_rows = []
    target_rows = []

    with tempfile.TemporaryDirectory(prefix='of3_dockq_') as tmp:
        tmpdir = Path(tmp)
        for pdbid in selected:
            if pdbid not in targets:
                print(f'skip missing metadata: {pdbid}')
                continue
            target = targets[pdbid]
            rec_chains = normalize_chains(target['rchain'])
            lig_chains = normalize_chains(target['lchain'])
            gt_rec, gt_lig = native_paths(ppc_root, args.dataset, pdbid)
            if not gt_rec.exists() or not gt_lig.exists():
                print(f'skip missing native: {pdbid}')
                continue

            dirs = target_dirs_for(args.pred_root, pdbid)
            models = []
            for d in dirs:
                models.extend(find_sample_models(d))
            if not models:
                print(f'skip no predictions: {pdbid}')
                target_rows.append({
                    'target': pdbid,
                    'n_models': 0,
                    'status': 'no_predictions',
                })
                continue

            per_target = []
            for idx, model_path in enumerate(models):
                conf = read_confidence(model_path)
                try:
                    dockq = dockq_one(
                        paper_eval, model_path, gt_rec, gt_lig,
                        rec_chains, lig_chains, tmpdir, f'{pdbid}_{idx}'
                    )
                    status = 'ok'
                    message = ''
                except Exception as exc:
                    dockq = math.nan
                    status = 'error'
                    message = str(exc)
                    print(f'ERROR {pdbid} {model_path}: {exc}')

                row = {
                    'target': pdbid,
                    'source_model': str(model_path),
                    'seed': conf['seed'],
                    'sample': conf['sample'],
                    'ranking_score': conf['ranking_score'],
                    'iptm': conf['iptm'],
                    'ptm': conf['ptm'],
                    'dockq': dockq,
                    'classification': classify_dockq(dockq),
                    'success': int(finite(dockq) and dockq >= thr),
                    'status': status,
                    'message': message,
                }
                per_target.append(row)
                detail_rows.append(row)

            valid = [r for r in per_target if r['status'] == 'ok' and finite(r['dockq'])]
            if not valid:
                target_rows.append({
                    'target': pdbid,
                    'n_models': len(per_target),
                    'status': 'all_failed',
                })
                continue

            ranked = sorted(valid, key=rank_key, reverse=True)
            for rank, row in enumerate(ranked, 1):
                row['rank_by_confidence'] = rank
            oracle = max(valid, key=lambda r: r['dockq'])
            top1 = ranked[0]
            summary = {
                'target': pdbid,
                'n_models': len(valid),
                'top1_dockq': top1['dockq'],
                'top1_classification': top1['classification'],
                'top1_ranking_score': top1['ranking_score'],
                'top1_source_model': top1['source_model'],
                'oracle_dockq': oracle['dockq'],
                'oracle_classification': oracle['classification'],
                'oracle_source_model': oracle['source_model'],
                'status': 'ok',
            }
            for k in topks:
                summary[f'success@{k}'] = success_in_topn(ranked, k, thr)
            summary['success_oracle'] = int(oracle['dockq'] >= thr)
            target_rows.append(summary)
            print(
                f"{pdbid}: top1_DockQ={top1['dockq']:.3f} ({top1['classification']}) "
                f"oracle={oracle['dockq']:.3f} "
                f"S@1={summary.get('success@1', 0)} "
                f"S_oracle={summary['success_oracle']} n={len(valid)}"
            )

    ok_targets = [r for r in target_rows if r.get('status') == 'ok']
    denom = len(ok_targets)
    aggregate = {
        'dataset': args.dataset,
        'pred_root': str(Path(args.pred_root).resolve()),
        'dockq_threshold': thr,
        'n_targets_ok': denom,
        'n_targets_listed': len(selected),
        'mean_top1_dockq': mean([float(r['top1_dockq']) for r in ok_targets]),
        'mean_oracle_dockq': mean([float(r['oracle_dockq']) for r in ok_targets]),
    }
    for k in topks:
        key = f'success@{k}'
        if denom:
            aggregate[key] = sum(int(r.get(key, 0)) for r in ok_targets) / denom
        else:
            aggregate[key] = math.nan
    if denom:
        aggregate['success_oracle'] = (
            sum(int(r.get('success_oracle', 0)) for r in ok_targets) / denom
        )
        for cls in ('high', 'medium', 'acceptable', 'unacceptable'):
            aggregate[f'top1_frac_{cls}'] = (
                sum(1 for r in ok_targets if r.get('top1_classification') == cls) / denom
            )
    else:
        aggregate['success_oracle'] = math.nan

    for path, rows, preferred in (
        (args.out_detail, detail_rows, None),
        (args.out_targets, target_rows, None),
        (args.out_summary, [aggregate], list(aggregate.keys())),
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text('')
            continue
        fieldnames = preferred or sorted({k for row in rows for k in row.keys()})
        # Keep a stable, readable order for detail/target CSVs.
        if preferred is None:
            priority = [
                'target', 'rank_by_confidence', 'seed', 'sample',
                'ranking_score', 'iptm', 'ptm', 'dockq', 'classification',
                'success', 'status', 'n_models',
                'top1_dockq', 'top1_classification', 'oracle_dockq',
                'success@1', 'success@5', 'success_oracle', 'source_model',
            ]
            ordered = [k for k in priority if k in fieldnames]
            ordered.extend(k for k in fieldnames if k not in ordered)
            fieldnames = ordered
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    print('--- aggregate ---')
    for key in (
        'n_targets_ok', 'mean_top1_dockq', 'mean_oracle_dockq',
        'success@1', 'success@5', 'success_oracle',
    ):
        if key in aggregate:
            print(f'{key}: {aggregate[key]}')
    print(f'detail  -> {args.out_detail}')
    print(f'targets -> {args.out_targets}')
    print(f'summary -> {args.out_summary}')


if __name__ == '__main__':
    main()
