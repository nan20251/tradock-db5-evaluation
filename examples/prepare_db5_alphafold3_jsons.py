#!/usr/bin/env python3
"""
Prepare DB5/DB5-u query JSON files for OpenFold3 (default) or official AlphaFold 3.

OpenFold3 format (default):

  {
    "queries": {
      "1ABC": {
        "chains": [
          {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "..."},
          {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "..."}
        ]
      }
    }
  }

Official AF3 dialect (--format alphafold3) is also supported for DeepMind AF3.
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


def parse_seeds(value):
    seeds = [int(token) for token in parse_csv(value)]
    if not seeds:
        raise ValueError('modelSeeds must contain at least one integer')
    return seeds


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


def sequence_for_chain(pdb_path, chain_id):
    residues = []
    seen = set()
    with open(pdb_path, 'r', errors='ignore') as handle:
        for line in handle:
            if not line.startswith('ATOM  ') or len(line) < 27:
                continue
            if line[21] != chain_id:
                continue
            if line[12:16].strip() != 'CA':
                continue
            resname = line[17:20].strip()
            key = (line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            residues.append(AA3_TO_1.get(resname, 'X'))
    return ''.join(residues)


def collect_chain_entries(target, ppc_root, dataset):
    pdbid = target['pdb']
    rec_chains = normalize_chains(target['rchain'])
    lig_chains = normalize_chains(target['lchain'])
    rec_pdb, lig_pdb = input_paths(ppc_root, dataset, pdbid)
    entries = []
    empty = False
    for chain_id in rec_chains:
        seq = sequence_for_chain(rec_pdb, chain_id)
        if not seq:
            empty = True
        entries.append((chain_id, seq))
    for chain_id in lig_chains:
        seq = sequence_for_chain(lig_pdb, chain_id)
        if not seq:
            empty = True
        entries.append((chain_id, seq))
    return entries, empty


def build_openfold3_query(pdbid, chain_entries):
    # Group identical sequences into one chain entry (homomer-friendly).
    by_seq = {}
    for chain_id, seq in chain_entries:
        by_seq.setdefault(seq, []).append(chain_id)
    chains = []
    for seq, chain_ids in by_seq.items():
        chains.append({
            'molecule_type': 'protein',
            'chain_ids': chain_ids,
            'sequence': seq,
        })
    return {
        'queries': {
            pdbid: {'chains': chains},
        }
    }


def build_af3_json(pdbid, chain_entries, model_seeds, version):
    return {
        'name': pdbid,
        'modelSeeds': list(model_seeds),
        'sequences': [
            {'protein': {'id': chain_id, 'sequence': seq}}
            for chain_id, seq in chain_entries
        ],
        'dialect': 'alphafold3',
        'version': int(version),
    }


def write_json(path, payload, overwrite):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5', choices=['DB5', 'DB5-u'])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--targets', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument(
        '--format',
        default='openfold3',
        choices=['openfold3', 'alphafold3'],
        help='openfold3 (default) or official alphafold3 dialect.',
    )
    parser.add_argument(
        '--bundle',
        action='store_true',
        help='Write one combined OpenFold3 queries JSON instead of per-target files.',
    )
    parser.add_argument('--model_seeds', default='1,2,3',
                        help='Only used for --format alphafold3.')
    parser.add_argument('--json_version', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.bundle and args.format != 'openfold3':
        raise SystemExit('--bundle is only supported with --format openfold3')

    ppc_root = Path(args.ppc_root).resolve()
    dataset_dir = ppc_root / 'dataset' / args.dataset
    out_dir = Path(args.out_dir).resolve()
    model_seeds = parse_seeds(args.model_seeds)
    targets = select_targets(read_targets(dataset_dir, args.dataset), args)

    rows = []
    bundled = {'queries': {}}

    for target in targets:
        pdbid = target['pdb']
        chain_entries, empty = collect_chain_entries(target, ppc_root, args.dataset)
        ids = [cid for cid, _ in chain_entries]
        if len(ids) != len(set(ids)):
            status = 'duplicate_chain_ids'
        elif empty:
            status = 'empty_sequence'
        else:
            if args.format == 'openfold3':
                payload = build_openfold3_query(pdbid, chain_entries)
                if args.bundle:
                    bundled['queries'][pdbid] = payload['queries'][pdbid]
                    status = 'bundled'
                else:
                    out_path = out_dir / f'{pdbid}.json'
                    written = write_json(out_path, payload, args.overwrite)
                    status = 'written' if written else 'skipped_existing'
            else:
                payload = build_af3_json(
                    pdbid, chain_entries, model_seeds, args.json_version
                )
                out_path = out_dir / f'{pdbid}.json'
                written = write_json(out_path, payload, args.overwrite)
                status = 'written' if written else 'skipped_existing'

        rows.append({
            'target': pdbid,
            'dataset': args.dataset,
            'format': args.format,
            'receptor_chains': ''.join(normalize_chains(target['rchain'])),
            'ligand_chains': ''.join(normalize_chains(target['lchain'])),
            'n_chains': len(chain_entries),
            'total_residues': sum(len(seq) for _, seq in chain_entries),
            'status': status,
        })
        print(
            f"{pdbid}: {status} format={args.format} chains={''.join(ids)} "
            f"len={sum(len(seq) for _, seq in chain_entries)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        bundle_path = out_dir / f'{args.dataset}_openfold3_queries.json'
        write_json(bundle_path, bundled, overwrite=True)
        print(f'bundle -> {bundle_path} ({len(bundled["queries"])} queries)')

    manifest = out_dir / f'{args.dataset}_{args.format}_jsons.csv'
    with open(manifest, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()) if rows else ['target'],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'manifest -> {manifest}')


if __name__ == '__main__':
    main()
