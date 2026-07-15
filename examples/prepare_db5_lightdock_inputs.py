#!/usr/bin/env python3
"""
Prepare PPCBench DB5-u receptor/ligand files for examples/run_lightdock.py.

Output layout:
  <out_dir>/<PDB>_receptor.pdb
  <out_dir>/<PDB>_ligand.pdb
  <out_dir>/<PDB>.chains
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_csv(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def normalize_chains(value):
    if isinstance(value, list):
        return ''.join(str(x) for x in value)
    return str(value)


def read_targets(dataset_dir, dataset):
    targets = []
    with open(dataset_dir / f'{dataset}.json', 'r') as handle:
        for line in handle:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    return targets


def select_targets(targets, args):
    if args.targets:
        wanted = set(parse_csv(args.targets))
        targets = [t for t in targets if t['pdb'] in wanted]
    if args.limit:
        targets = targets[:args.limit]
    return targets


def input_paths(ppc_root, dataset, pdbid):
    struct_dir = ppc_root / 'dataset' / dataset / 'structures' / pdbid
    if dataset == 'DB5-u':
        return struct_dir / f'{pdbid}_r_u_f.pdb', struct_dir / f'{pdbid}_l_u_f.pdb'
    if dataset == 'DB5':
        return struct_dir / f'{pdbid}_r_b.pdb', struct_dir / f'{pdbid}_l_b.pdb'
    raise ValueError('only DB5 and DB5-u are supported')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5-u', choices=['DB5', 'DB5-u'])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--targets', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    ppc_root = Path(args.ppc_root).resolve()
    dataset_dir = ppc_root / 'dataset' / args.dataset
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in select_targets(read_targets(dataset_dir, args.dataset), args):
        pdbid = target['pdb']
        rec_src, lig_src = input_paths(ppc_root, args.dataset, pdbid)
        rec_dst = out_dir / f'{pdbid}_receptor.pdb'
        lig_dst = out_dir / f'{pdbid}_ligand.pdb'
        chains_dst = out_dir / f'{pdbid}.chains'
        status = 'written'
        if rec_dst.exists() and lig_dst.exists() and chains_dst.exists() and not args.overwrite:
            status = 'skipped_existing'
        else:
            shutil.copyfile(rec_src, rec_dst)
            shutil.copyfile(lig_src, lig_dst)
            with open(chains_dst, 'w') as handle:
                handle.write(
                    f"{normalize_chains(target['rchain'])} "
                    f"{normalize_chains(target['lchain'])}\n"
                )
        rows.append({
            'dataset': args.dataset,
            'target': pdbid,
            'receptor': str(rec_dst),
            'ligand': str(lig_dst),
            'chains': str(chains_dst),
            'status': status,
        })
        print(f'{pdbid}: {status}')

    manifest = out_dir / f'{args.dataset}_lightdock_inputs.csv'
    with open(manifest, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ['target'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'manifest -> {manifest}')


if __name__ == '__main__':
    main()
