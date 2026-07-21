#!/usr/bin/env python3
"""
Export LightDock complex decoys into PPCBench/TraDock paper-style pose layout.

Input comes from examples/run_lightdock.py:
  <ld_root>/<PDB>/rec_chains.txt
  <ld_root>/<PDB>/lig_chains.txt
  <ld_root>/<PDB>/swarm_*/gso_*.out
  <ld_root>/<PDB>/swarm_*/lightdock_*.pdb

Output layout:
  <out_root>/lightdock_1/<PDB>/<PDB>_lightdock_1_id.pdb
  <out_root>/lightdock_1/<PDB>/<PDB>_lightdock_1_rec_id.pdb
  ...
"""
import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path


MANIFEST_FIELDS = [
    'dataset', 'target', 'status', 'message', 'n_requested', 'n_exported',
    'ranking', 'ld_target_dir', 'out_root',
]

SELECTED_FIELDS = [
    'dataset', 'target', 'rank', 'pose_model', 'score_field', 'score',
    'scoring', 'luciferin', 'swarm', 'glowworm', 'gso_file', 'source_pdb',
]


def parse_csv(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def read_chain_list(path):
    with open(path) as handle:
        content = handle.read().strip()
    parts = content.split()
    if len(parts) == 1 and len(parts[0]) > 1:
        return list(parts[0])
    return parts


def numeric_key(path):
    swarm = 0
    pose = 0
    for part in Path(path).parts:
        match = re.match(r'swarm_(\d+)$', part)
        if match:
            swarm = int(match.group(1))
            break
    match = re.search(r'lightdock_(\d+)\.pdb$', os.path.basename(path))
    if match:
        pose = int(match.group(1))
    return swarm, pose, str(path)


def parse_swarm_id(path):
    match = re.match(r'swarm_(\d+)$', Path(path).name)
    return int(match.group(1)) if match else 0


def parse_gso_step(path):
    match = re.match(r'gso_(\d+)\.out$', Path(path).name)
    return int(match.group(1)) if match else -1


def final_gso_file(swarm_dir):
    files = sorted(Path(swarm_dir).glob('gso_*.out'), key=parse_gso_step)
    return files[-1] if files else None


def parse_float(value):
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out


def parse_gso_line(line, glowworm_id):
    if not line or line.startswith('#') or not line.startswith('('):
        return None
    try:
        last = line.index(')')
    except ValueError:
        return None
    rest = line[last + 1:].split()
    if len(rest) >= 6:
        return {
            'glowworm': glowworm_id,
            'luciferin': parse_float(rest[2]),
            'scoring': parse_float(rest[5]),
        }
    if len(rest) >= 4:
        return {
            'glowworm': glowworm_id,
            'luciferin': parse_float(rest[0]),
            'scoring': parse_float(rest[3]),
        }
    return None


def read_gso_scores(gso_file):
    rows = []
    with open(gso_file, 'r', errors='ignore') as handle:
        glowworm_id = 0
        for line in handle:
            if line.startswith('#'):
                continue
            parsed = parse_gso_line(line.strip(), glowworm_id)
            if parsed is not None:
                rows.append(parsed)
                glowworm_id += 1
    return rows


def lightdock_scored_decoys(target_dir, score_field, allow_file_order_fallback=False):
    records = []
    for swarm_dir in sorted(target_dir.glob('swarm_*'), key=parse_swarm_id):
        swarm_id = parse_swarm_id(swarm_dir)
        gso_file = final_gso_file(swarm_dir)
        if gso_file is None:
            continue
        for row in read_gso_scores(gso_file):
            decoy = swarm_dir / f"lightdock_{row['glowworm']}.pdb"
            if not decoy.exists():
                continue
            score = row.get(score_field, math.nan)
            if not math.isfinite(score):
                continue
            records.append({
                'swarm': swarm_id,
                'glowworm': row['glowworm'],
                'scoring': row.get('scoring', math.nan),
                'luciferin': row.get('luciferin', math.nan),
                'score': score,
                'gso_file': gso_file,
                'pdb': decoy,
            })

    if records:
        return sorted(records, key=lambda r: (r['score'], -r['swarm'], -r['glowworm']), reverse=True)

    if not allow_file_order_fallback:
        return []

    fallback = []
    for decoy in lightdock_decoys_by_file_order(target_dir, None):
        swarm_id, glowworm_id, _ = numeric_key(decoy)
        fallback.append({
            'swarm': swarm_id,
            'glowworm': glowworm_id,
            'scoring': math.nan,
            'luciferin': math.nan,
            'score': math.nan,
            'gso_file': '',
            'pdb': decoy,
        })
    return fallback


def detect_chains(pdb_path):
    chains = []
    with open(pdb_path, 'r', errors='ignore') as handle:
        for line in handle:
            if line.startswith(('ATOM  ', 'HETATM')) and len(line) > 21:
                chain = line[21]
                if chain not in chains:
                    chains.append(chain)
    return chains


def split_decoy(decoy_pdb, native_rec_chains, native_lig_chains):
    detected = detect_chains(decoy_pdb)
    native_rec_set = set(native_rec_chains)
    native_lig_set = set(native_lig_chains)
    detected_set = set(detected)

    if native_rec_set.issubset(detected_set) and native_lig_set.issubset(detected_set):
        rec_set = native_rec_set
        lig_set = native_lig_set
    else:
        n_rec = len(native_rec_chains)
        n_lig = len(native_lig_chains)
        if len(detected) < n_rec + n_lig:
            raise RuntimeError(
                f'not enough chains in decoy: got {detected}, '
                f'expected {n_rec}+{n_lig}'
            )
        rec_set = set(detected[:n_rec])
        lig_set = set(detected[n_rec:n_rec + n_lig])

    rec_lines = []
    lig_lines = []
    with open(decoy_pdb, 'r', errors='ignore') as handle:
        for line in handle:
            if not line.startswith(('ATOM  ', 'HETATM')) or len(line) <= 21:
                continue
            chain = line[21]
            if chain in rec_set:
                rec_lines.append(line)
            elif chain in lig_set:
                lig_lines.append(line)

    if not rec_lines:
        raise RuntimeError('no receptor atoms after LightDock split')
    if not lig_lines:
        raise RuntimeError('no ligand atoms after LightDock split')
    return rec_lines, lig_lines


def write_pdb(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        for line in lines:
            handle.write(line if line.endswith('\n') else line + '\n')
        handle.write('END\n')


def rank_lig_path(out_root, prefix, rank, pdbid):
    pose_model = f'{prefix}_{rank}'
    return out_root / pose_model / pdbid / f'{pdbid}_{pose_model}_id.pdb'


def outputs_complete(out_root, prefix, pdbid, nmax):
    if nmax <= 0:
        return False
    return all(
        rank_lig_path(out_root, prefix, rank, pdbid).exists()
        for rank in range(1, nmax + 1)
    )


def lightdock_decoys_by_file_order(target_dir, nmax):
    paths = sorted(
        glob.glob(str(target_dir / 'swarm_*' / 'lightdock_*.pdb')),
        key=numeric_key,
    )
    paths = [Path(path) for path in paths]
    return paths if nmax <= 0 else paths[:nmax]


def process_target(args, target_dir):
    pdbid = target_dir.name
    out_root = Path(args.out_root).resolve()
    if outputs_complete(out_root, args.prefix, pdbid, args.nmax) and not args.overwrite:
        return {
            'dataset': args.dataset,
            'target': pdbid,
            'status': 'skipped_existing',
            'message': '',
            'n_requested': args.nmax,
            'n_exported': args.nmax,
            'ranking': args.score_field,
            'ld_target_dir': str(target_dir),
            'out_root': str(out_root),
        }, []

    rec_chains = read_chain_list(target_dir / 'rec_chains.txt')
    lig_chains = read_chain_list(target_dir / 'lig_chains.txt')
    records = lightdock_scored_decoys(
        target_dir, args.score_field,
        allow_file_order_fallback=args.allow_file_order_fallback,
    )
    export_n = len(records) if args.nmax <= 0 else args.nmax
    if len(records) < export_n:
        raise RuntimeError(
            f'only {len(records)} scored LightDock decoys, expected {export_n}; '
            'check swarm_*/gso_*.out or pass --allow_file_order_fallback'
        )

    n_exported = 0
    selected_rows = []
    for rank, record in enumerate(records[:export_n], 1):
        decoy = record['pdb']
        pose_model = f'{args.prefix}_{rank}'
        rank_dir = out_root / pose_model / pdbid
        rec_lines, lig_lines = split_decoy(decoy, rec_chains, lig_chains)

        lig_id = rank_dir / f'{pdbid}_{pose_model}_id.pdb'
        lig_plain = rank_dir / f'{pdbid}_{pose_model}.pdb'
        rec_id = rank_dir / f'{pdbid}_{pose_model}_rec_id.pdb'
        complex_path = rank_dir / f'{pdbid}_predicted.pdb'

        write_pdb(lig_id, lig_lines)
        write_pdb(lig_plain, lig_lines)
        write_pdb(rec_id, rec_lines)
        write_pdb(complex_path, rec_lines + lig_lines)
        n_exported += 1
        selected_rows.append({
            'dataset': args.dataset,
            'target': pdbid,
            'rank': rank,
            'pose_model': pose_model,
            'score_field': args.score_field,
            'score': record['score'],
            'scoring': record['scoring'],
            'luciferin': record['luciferin'],
            'swarm': record['swarm'],
            'glowworm': record['glowworm'],
            'gso_file': str(record['gso_file']),
            'source_pdb': str(decoy),
        })

    return {
        'dataset': args.dataset,
        'target': pdbid,
        'status': 'done',
        'message': '',
        'n_requested': args.nmax if args.nmax > 0 else export_n,
        'n_exported': n_exported,
        'ranking': args.score_field,
        'ld_target_dir': str(target_dir),
        'out_root': str(out_root),
    }, selected_rows


def selected_target_dirs(args):
    root = Path(args.ld_root).resolve()
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.targets:
        wanted = set(parse_csv(args.targets))
        dirs = [path for path in dirs if path.name in wanted]
    if args.limit:
        dirs = dirs[:args.limit]
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ld_root', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--dataset', default='DB5-u')
    parser.add_argument('--prefix', default='lightdock')
    parser.add_argument('--nmax', type=int, default=100,
                        help='Max decoys to export (0 = all scored decoys)')
    parser.add_argument('--score_field', default='scoring',
                        choices=['scoring', 'luciferin'],
                        help='LightDock field used for Original rank')
    parser.add_argument('--allow_file_order_fallback', action='store_true',
                        help='Use swarm/file order if no GSO scores are found')
    parser.add_argument('--targets', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fail_fast', action='store_true')
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest
        else out_root / f'{args.prefix}_{args.dataset}_manifest.csv'
    )
    selected_csv = out_root / f'{args.prefix}_{args.dataset}_selected.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)

    target_dirs = selected_target_dirs(args)
    print(f'ld_root={Path(args.ld_root).resolve()}')
    print(f'out_root={out_root}')
    print(f'targets={len(target_dirs)} nmax={args.nmax}')

    with open(manifest, 'w', newline='') as handle, open(selected_csv, 'w', newline='') as selected_handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        selected_writer = csv.DictWriter(selected_handle, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        selected_writer.writeheader()
        for index, target_dir in enumerate(target_dirs, 1):
            print(f'[{index}/{len(target_dirs)}] {target_dir.name}', flush=True)
            try:
                row, selected_rows = process_target(args, target_dir)
            except Exception as exc:
                row = {
                    'dataset': args.dataset,
                    'target': target_dir.name,
                    'status': 'error',
                    'message': str(exc),
                    'n_requested': args.nmax,
                    'n_exported': 0,
                    'ranking': args.score_field,
                    'ld_target_dir': str(target_dir),
                    'out_root': str(out_root),
                }
                selected_rows = []
                if args.fail_fast:
                    writer.writerow(row)
                    handle.flush()
                    raise
            writer.writerow(row)
            selected_writer.writerows(selected_rows)
            handle.flush()
            selected_handle.flush()
            print(f"  {row['status']} exported={row.get('n_exported', 0)}", flush=True)

    print(f'manifest -> {manifest}')
    print(f'selected -> {selected_csv}')


if __name__ == '__main__':
    main()
