"""
LightDock native complexes -> TransformerDock native surface data.

Input layout:
  <ld_root>/<stem>/
    native.pdb
    rec_chains.txt
    lig_chains.txt

Output:
  <out_dir>/
    {stem}_receptor.ply
    {stem}_ligand.ply
    pairs.csv
"""

import argparse
import csv
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.prep_dockground_decoys import split_chains
from examples.surface_gen import pdb_to_surface_ply


def read_chain_list(path):
    with open(path) as f:
        text = f.read().strip()
    parts = text.split()
    if len(parts) == 1 and len(parts[0]) > 1:
        return list(parts[0])
    return parts


def iter_targets(ld_root, limit=None):
    targets = []
    for stem in sorted(os.listdir(ld_root)):
        target_dir = os.path.join(ld_root, stem)
        if not os.path.isdir(target_dir):
            continue
        native = os.path.join(target_dir, 'native.pdb')
        rec_chains = os.path.join(target_dir, 'rec_chains.txt')
        lig_chains = os.path.join(target_dir, 'lig_chains.txt')
        if all(os.path.exists(p) for p in (native, rec_chains, lig_chains)):
            targets.append((stem, native, rec_chains, lig_chains))
    if limit:
        targets = targets[:limit]
    return targets


def process_target(job):
    stem, native_pdb, rec_chains_path, lig_chains_path, out_dir, voxel_size = job
    rec_ply = os.path.join(out_dir, f'{stem}_receptor.ply')
    lig_ply = os.path.join(out_dir, f'{stem}_ligand.ply')
    if os.path.exists(rec_ply) and os.path.exists(lig_ply):
        return {'ok': True, 'stem': stem, 'cached': True, 'msg': 'cached'}

    rec_chains = read_chain_list(rec_chains_path)
    lig_chains = read_chain_list(lig_chains_path)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_pdb = os.path.join(tmp, 'receptor.pdb')
            lig_pdb = os.path.join(tmp, 'ligand.pdb')
            split_chains(native_pdb, rec_pdb, lig_pdb, set(rec_chains), set(lig_chains))
            ok_rec = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
            ok_lig = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)
        if not (ok_rec and ok_lig):
            for path in (rec_ply, lig_ply):
                if os.path.exists(path):
                    os.remove(path)
            return {'ok': False, 'stem': stem, 'cached': False,
                    'msg': f'ply failed rec={ok_rec} lig={ok_lig}'}
    except Exception as exc:
        for path in (rec_ply, lig_ply):
            if os.path.exists(path):
                os.remove(path)
        return {'ok': False, 'stem': stem, 'cached': False, 'msg': str(exc)}

    return {'ok': True, 'stem': stem, 'cached': False, 'msg': 'ok'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ld_root', required=True,
                        help='LightDock output root with <stem>/native.pdb')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--voxel_size', type=float, default=3.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = iter_targets(args.ld_root, args.limit)
    if not targets:
        sys.exit(f'no LightDock native targets found under {args.ld_root}')

    jobs = [
        (stem, native, rec_chains, lig_chains, args.out_dir, args.voxel_size)
        for stem, native, rec_chains, lig_chains in targets
    ]

    rows = []
    failures = []
    if args.workers == 1:
        for job in jobs:
            result = process_target(job)
            (rows if result['ok'] else failures).append(result)
            print(f"[{result['stem']}] {'OK' if result['ok'] else 'FAIL'} {result['msg']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_target, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                (rows if result['ok'] else failures).append(result)
                print(f"[{result['stem']}] {'OK' if result['ok'] else 'FAIL'} {result['msg']}")

    rows.sort(key=lambda row: row['stem'])
    pairs_csv = os.path.join(args.out_dir, 'pairs.csv')
    with open(pairs_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'label'])
        for row in rows:
            writer.writerow([row['stem'], 1])

    print('---')
    print(f'ok={len(rows)} fail={len(failures)}')
    print(f'pairs_csv={pairs_csv}')

    if failures:
        failures_log = os.path.join(args.out_dir, 'failures.log')
        with open(failures_log, 'w') as f:
            for row in failures:
                f.write(f"{row['stem']}\t{row['msg']}\n")
        print(f'failures={failures_log}')


if __name__ == '__main__':
    main()
