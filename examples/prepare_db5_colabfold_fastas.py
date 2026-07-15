#!/usr/bin/env python3
"""
Prepare DB5/DB5-u FASTA files for ColabFold multimer sampling.

Each output file uses the ColabFold complex FASTA format:

  >1ABC
  RECEPTOR_SEQUENCE:LIGAND_SEQUENCE

For AlphaFold-vs-TraDock comparison, DB5 bound structures are the default
sequence source because the ground truth evaluation also uses DB5 bound
complexes. DB5-u is supported when apo sequence extraction is needed.
"""
import argparse
import csv
import json
from pathlib import Path


AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y',
    'VAL': 'V',
}


def parse_csv(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def normalize_chains(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    return list(str(value))


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
    if dataset == 'DB5':
        return struct_dir / f'{pdbid}_r_b.pdb', struct_dir / f'{pdbid}_l_b.pdb'
    if dataset == 'DB5-u':
        return struct_dir / f'{pdbid}_r_u_f.pdb', struct_dir / f'{pdbid}_l_u_f.pdb'
    raise ValueError('only DB5 and DB5-u are supported')


def sequence_for_chains(pdb_path, chain_ids):
    wanted = set(chain_ids)
    residues = []
    seen = set()
    with open(pdb_path, 'r', errors='ignore') as handle:
        for line in handle:
            if not line.startswith('ATOM  ') or len(line) < 27:
                continue
            chain = line[21]
            if chain not in wanted:
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            resname = line[17:20].strip()
            key = (chain, line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            residues.append((chain, AA3_TO_1.get(resname, 'X')))

    seq_by_chain = {chain: [] for chain in chain_ids}
    for chain, aa in residues:
        if chain in seq_by_chain:
            seq_by_chain[chain].append(aa)
    return ''.join(''.join(seq_by_chain[chain]) for chain in chain_ids)


def write_fasta(path, pdbid, rec_seq, lig_seq, overwrite):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(f'>{pdbid}\n')
        handle.write(f'{rec_seq}:{lig_seq}\n')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5', choices=['DB5', 'DB5-u'])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--targets', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    ppc_root = Path(args.ppc_root).resolve()
    dataset_dir = ppc_root / 'dataset' / args.dataset
    out_dir = Path(args.out_dir).resolve()
    targets = select_targets(read_targets(dataset_dir, args.dataset), args)

    rows = []
    for target in targets:
        pdbid = target['pdb']
        rec_chains = normalize_chains(target['rchain'])
        lig_chains = normalize_chains(target['lchain'])
        rec_pdb, lig_pdb = input_paths(ppc_root, args.dataset, pdbid)
        rec_seq = sequence_for_chains(rec_pdb, rec_chains)
        lig_seq = sequence_for_chains(lig_pdb, lig_chains)
        if not rec_seq or not lig_seq:
            status = 'empty_sequence'
        else:
            fasta = out_dir / f'{pdbid}.fasta'
            written = write_fasta(fasta, pdbid, rec_seq, lig_seq, args.overwrite)
            status = 'written' if written else 'skipped_existing'
        rows.append({
            'target': pdbid,
            'dataset': args.dataset,
            'receptor_chains': ''.join(rec_chains),
            'ligand_chains': ''.join(lig_chains),
            'receptor_length': len(rec_seq),
            'ligand_length': len(lig_seq),
            'status': status,
        })
        print(f"{pdbid}: {status} rec_len={len(rec_seq)} lig_len={len(lig_seq)}")

    manifest = out_dir / f'{args.dataset}_colabfold_fastas.csv'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ['target'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'manifest -> {manifest}')


if __name__ == '__main__':
    main()
