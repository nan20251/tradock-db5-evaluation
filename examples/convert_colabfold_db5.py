#!/usr/bin/env python3
"""
Convert ColabFold multimer outputs into PPCBench/TraDock pose layout.

The conversion aligns the AlphaFold receptor part onto the DB5 reference
receptor by CA atoms, applies the same transform to the AlphaFold ligand, and
writes only the ligand pose:

  <results_root>/<dataset>/af2m_1/<PDB>/<PDB>_af2m_1_id.pdb
  <results_root>/<dataset>/af2m_2/<PDB>/<PDB>_af2m_2_id.pdb
  ...

If multiple ColabFold seed directories are present, all ranked PDB files are
pooled, ranked by AlphaFold confidence, and the final Top-N are written as
af2m_1, af2m_2, ...

This makes AlphaFold-generated poses compatible with
examples/eval_db5_paper_tradock.py.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path


AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def load_biopython():
    global PDBIO, PDBParser, Superimposer, Chain, Model, Residue, Structure
    from Bio.PDB import PDBIO, PDBParser, Superimposer
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure


def load_targets(path):
    targets = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                targets[row['pdb']] = row
    return targets


def normalize_chains(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    return list(str(value))


def first_model(structure):
    return next(structure.get_models())


def standard_residues(chain):
    out = []
    for residue in chain:
        if residue.id[0] == ' ' and residue.resname in AA3_TO_1:
            out.append(residue)
    return out


def ca_atoms_by_order(chain):
    atoms = []
    for residue in standard_residues(chain):
        if 'CA' in residue:
            atoms.append(residue['CA'])
    return atoms


def ca_atoms_from_residues(residues):
    atoms = []
    for residue in residues:
        if 'CA' in residue:
            atoms.append(residue['CA'])
    return atoms


def ordered_chains(model):
    return [chain for chain in model if len(standard_residues(chain)) > 0]


def parse_rank(path):
    name = path.name
    match = re.search(r'rank[_-](\d+)', name)
    if match:
        return int(match.group(1))
    match = re.search(r'model_(\d+)', name)
    if match:
        return int(match.group(1))
    return 999


def find_ranked_pdbs(output_dir):
    paths = []
    for path in Path(output_dir).rglob('*.pdb'):
        name = path.name.lower()
        if 'rank' in name and ('unrelaxed' in name or 'relaxed' in name):
            paths.append(path)
    return sorted(paths, key=lambda p: (parse_rank(p), p.name))


def filter_target_pdbs(paths, pdbid):
    key = pdbid.lower()
    out = []
    for path in paths:
        name = path.name.lower()
        if name.startswith(key + '_') or name.startswith(key + '-') or name.startswith(key):
            out.append(path)
    return out


def prefer_relaxed(paths):
    by_rank = {}
    for path in paths:
        rank = parse_rank(path)
        key = (path.parent, rank)
        old = by_rank.get(key)
        name = path.name.lower()
        is_relaxed = 'relaxed' in name and 'unrelaxed' not in name
        if old is None:
            by_rank[key] = path
            continue
        old_name = old.name.lower()
        old_relaxed = 'relaxed' in old_name and 'unrelaxed' not in old_name
        if is_relaxed and not old_relaxed:
            by_rank[key] = path
    return [by_rank[k] for k in sorted(by_rank, key=lambda x: (str(x[0]), x[1]))]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def confidence_score(scores):
    for key in ('ranking_confidence', 'iptm', 'mean_plddt', 'ptm'):
        value = scores.get(key)
        if finite(value):
            return value
    return math.nan


def sort_af_candidates(candidates):
    return sorted(
        candidates,
        key=lambda item: (
            finite(confidence_score(item['scores'])),
            confidence_score(item['scores']) if finite(confidence_score(item['scores'])) else -math.inf,
            -parse_rank(item['pdb']),
            str(item['pdb']),
        ),
        reverse=True,
    )


def score_json_candidates(pdb_path):
    stem = pdb_path.stem
    parent = pdb_path.parent
    candidates = []
    parts = stem.split('_')
    for i, part in enumerate(parts):
        if part == 'rank' and i + 1 < len(parts):
            rank_token = f'rank_{parts[i + 1]}'
            candidates.extend(parent.glob(f'*scores*{rank_token}*.json'))
            candidates.extend(parent.glob(f'*{rank_token}*scores*.json'))
    candidates.extend(parent.glob(stem.replace('unrelaxed', 'scores') + '*.json'))
    candidates.extend(parent.glob(stem.replace('relaxed', 'scores') + '*.json'))
    return sorted(set(candidates))


def safe_mean_plddt_from_pdb(pdb_path):
    vals = []
    with open(pdb_path) as handle:
        for line in handle:
            if line.startswith(('ATOM  ', 'HETATM')):
                try:
                    vals.append(float(line[60:66]))
                except ValueError:
                    pass
    return sum(vals) / len(vals) if vals else math.nan


def read_af_scores(pdb_path, ranking_debug):
    rank = parse_rank(pdb_path)
    out = {
        'af_rank': rank,
        'ranking_confidence': math.nan,
        'iptm': math.nan,
        'ptm': math.nan,
        'mean_plddt': safe_mean_plddt_from_pdb(pdb_path),
    }

    key = pdb_path.stem
    if ranking_debug:
        for score_key in ('iptm+ptm', 'ranking_confidence', 'iptm', 'ptm'):
            scores = ranking_debug.get(score_key)
            if isinstance(scores, dict):
                for model_key, value in scores.items():
                    if model_key in key:
                        if score_key == 'iptm+ptm':
                            out['ranking_confidence'] = value
                        else:
                            out[score_key] = value
        order = ranking_debug.get('order')
        if isinstance(order, list):
            for idx, model_key in enumerate(order, 1):
                if model_key in key:
                    out['af_rank'] = idx
                    break

    for json_path in score_json_candidates(pdb_path):
        try:
            with open(json_path) as handle:
                data = json.load(handle)
        except Exception:
            continue
        for source, dest in (
            ('ranking_confidence', 'ranking_confidence'),
            ('iptm', 'iptm'),
            ('ptm', 'ptm'),
            ('plddt', 'mean_plddt'),
        ):
            if source in data:
                value = data[source]
                if isinstance(value, list):
                    value = sum(value) / len(value) if value else math.nan
                out[dest] = value
        break
    return out


def copy_residue_to_chain(residue, chain, new_id):
    new_residue = Residue(new_id, residue.resname, residue.segid)
    for atom in residue:
        new_residue.add(atom.copy())
    chain.add(new_residue)


def make_transformed_ligand_structure(af_lig_segments, native_lig_chain_ids,
                                      rotation, translation):
    structure = Structure('af_ligand')
    model = Model(0)
    structure.add(model)

    for source_residues, native_chain_id in zip(af_lig_segments, native_lig_chain_ids):
        new_chain = Chain(native_chain_id)
        for idx, residue in enumerate(source_residues, 1):
            copy_residue_to_chain(residue, new_chain, (' ', idx, ' '))
        for atom in new_chain.get_atoms():
            atom.transform(rotation, translation)
        model.add(new_chain)
    return structure


def paired_ca_residues(native_chain, af_residues):
    native = ca_atoms_by_order(native_chain)
    pred = ca_atoms_from_residues(af_residues)
    n = min(len(native), len(pred))
    if n == 0:
        return [], []
    return native[:n], pred[:n]


def split_residues_by_native_chains(af_chain, native_model, native_chain_ids,
                                    label, pdbid):
    residues = standard_residues(af_chain)
    native_by_id = {chain.id: chain for chain in native_model}
    lengths = []
    for chain_id in native_chain_ids:
        native_chain = native_by_id.get(chain_id)
        if native_chain is None:
            raise RuntimeError(f'{pdbid}: missing native {label} chain {chain_id}')
        lengths.append(len(standard_residues(native_chain)))

    total = sum(lengths)
    if len(residues) != total:
        raise RuntimeError(
            f'{pdbid}: AF {label} chain length={len(residues)} does not match '
            f'native concatenated length={total} for chains {native_chain_ids}'
        )

    out = []
    start = 0
    for length in lengths:
        out.append(residues[start:start + length])
        start += length
    return out


def infer_af_segments(af_chains, native_rec_model, native_lig_model, target, pdbid):
    rec_chains = normalize_chains(target['rchain'])
    lig_chains = normalize_chains(target['lchain'])
    n_rec = len(rec_chains)
    n_lig = len(lig_chains)

    if len(af_chains) >= n_rec + n_lig and n_rec + n_lig > 2:
        rec_segments = [standard_residues(chain) for chain in af_chains[:n_rec]]
        lig_segments = [standard_residues(chain) for chain in af_chains[n_rec:n_rec + n_lig]]
        return rec_segments, lig_segments, 'chain_per_db5_chain'

    if len(af_chains) < 2:
        raise RuntimeError(f'{pdbid}: expected at least receptor and ligand AF chains')

    rec_segments = split_residues_by_native_chains(
        af_chains[0], native_rec_model, rec_chains, 'receptor', pdbid
    )
    lig_segments = split_residues_by_native_chains(
        af_chains[1], native_lig_model, lig_chains, 'ligand', pdbid
    )
    return rec_segments, lig_segments, 'merged_receptor_merged_ligand'


def reference_paths(ppc_root, dataset, pdbid):
    struct_dir = Path(ppc_root) / 'dataset' / dataset / 'structures' / pdbid
    if dataset == 'DB5':
        return struct_dir / f'{pdbid}_r_b.pdb', struct_dir / f'{pdbid}_l_b.pdb'
    if dataset == 'DB5-u':
        return struct_dir / f'{pdbid}_r_b_f.pdb', struct_dir / f'{pdbid}_l_b_f.pdb'
    raise ValueError('only DB5 and DB5-u are supported')


def convert_one(pdbid, target, af_pdb, ppc_root, dataset, results_root, pose_model):
    parser = PDBParser(QUIET=True)
    af_model = first_model(parser.get_structure(pdbid + '_af', str(af_pdb)))
    af_chains = ordered_chains(af_model)

    rec_ref, lig_ref = reference_paths(ppc_root, dataset, pdbid)
    native_rec_model = first_model(
        parser.get_structure(pdbid + '_native_rec', str(rec_ref))
    )
    native_lig_model = first_model(
        parser.get_structure(pdbid + '_native_lig', str(lig_ref))
    )
    af_rec_segments, af_lig_segments, mapping_mode = infer_af_segments(
        af_chains, native_rec_model, native_lig_model, target, pdbid
    )

    fixed_atoms = []
    moving_atoms = []
    native_by_id = {chain.id: chain for chain in native_rec_model}
    rec_chains = normalize_chains(target['rchain'])
    lig_chains = normalize_chains(target['lchain'])
    for native_chain_id, af_residues in zip(rec_chains, af_rec_segments):
        native_chain = native_by_id.get(native_chain_id)
        if native_chain is None:
            raise RuntimeError(f'{pdbid}: missing native receptor chain {native_chain_id}')
        fixed, moving = paired_ca_residues(native_chain, af_residues)
        fixed_atoms.extend(fixed)
        moving_atoms.extend(moving)

    if len(fixed_atoms) < 3:
        raise RuntimeError(f'{pdbid}: not enough paired CA atoms for superposition')

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    rotation, translation = sup.rotran

    out_dir = Path(results_root) / dataset / pose_model / pdbid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f'{pdbid}_{pose_model}_id.pdb'

    ligand_structure = make_transformed_ligand_structure(
        af_lig_segments, lig_chains, rotation, translation
    )
    io = PDBIO()
    io.set_structure(ligand_structure)
    io.save(str(out_pdb))

    return pose_model, out_pdb, sup.rms, mapping_mode


def load_ranking_debug(target_dirs, pdbid):
    for target_dir in target_dirs:
        candidates = [
            target_dir / 'ranking_debug.json',
            target_dir / f'{pdbid}_ranking_debug.json',
            target_dir / f'{pdbid.lower()}_ranking_debug.json',
        ]
        for path in candidates:
            if path.exists():
                with open(path) as handle:
                    return json.load(handle)
    return None


def load_ranking_debug_for_pdb(pdb_path, pdbid):
    pdb_path = Path(pdb_path)
    search_dirs = [pdb_path.parent, pdb_path.parent.parent]
    return load_ranking_debug(search_dirs, pdbid)


def target_dirs_for(colabfold_root, pdbid):
    root = Path(colabfold_root)
    if root.is_dir() and not list(root.glob('*.pdb')):
        dirs = [
            p for p in root.iterdir()
            if p.is_dir() and pdbid.lower() in p.name.lower()
        ]
        return dirs or [root]
    return [root]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5', choices=['DB5', 'DB5-u'])
    parser.add_argument('--dataset_json', required=True)
    parser.add_argument('--colabfold_root', required=True)
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--scores_csv', required=True)
    parser.add_argument('--targets', default='',
                        help='Comma-separated target IDs; default scans all.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max_models', type=int, default=5)
    args = parser.parse_args()

    load_biopython()

    targets = load_targets(args.dataset_json)
    selected = [x.strip().upper() for x in args.targets.split(',') if x.strip()]
    if not selected:
        selected = sorted(targets)
    if args.limit:
        selected = selected[:args.limit]

    fieldnames = [
        'target', 'pose_model', 'af_rank', 'source_af_rank',
        'ranking_confidence', 'iptm', 'ptm', 'mean_plddt',
        'receptor_align_rmsd', 'mapping_mode', 'source_pdb',
        'converted_ligand',
    ]
    rows = []
    for pdbid in selected:
        if pdbid not in targets:
            print(f'skip missing target metadata: {pdbid}')
            continue
        target_dirs = target_dirs_for(args.colabfold_root, pdbid)
        af_pdbs = []
        for target_dir in target_dirs:
            af_pdbs.extend(find_ranked_pdbs(target_dir))
        af_pdbs = prefer_relaxed(filter_target_pdbs(af_pdbs, pdbid))
        if not af_pdbs:
            print(f'skip no AF PDBs: {pdbid}')
            continue

        candidates = []
        for af_pdb in af_pdbs:
            ranking_debug = load_ranking_debug_for_pdb(af_pdb, pdbid)
            scores = read_af_scores(af_pdb, ranking_debug)
            candidates.append({'pdb': af_pdb, 'scores': scores})
        selected_candidates = sort_af_candidates(candidates)[:args.max_models]

        for selected_rank, candidate in enumerate(selected_candidates, 1):
            af_pdb = candidate['pdb']
            scores = candidate['scores']
            pose_model = f'af2m_{selected_rank}'
            try:
                pose_model, converted, align_rmsd, mapping_mode = convert_one(
                    pdbid, targets[pdbid], af_pdb, args.ppc_root,
                    args.dataset, args.results_root, pose_model
                )
                row = {
                    'target': pdbid,
                    'pose_model': pose_model,
                    'af_rank': selected_rank,
                    'source_af_rank': scores['af_rank'],
                    'ranking_confidence': scores['ranking_confidence'],
                    'iptm': scores['iptm'],
                    'ptm': scores['ptm'],
                    'mean_plddt': scores['mean_plddt'],
                    'receptor_align_rmsd': align_rmsd,
                    'mapping_mode': mapping_mode,
                    'source_pdb': str(af_pdb),
                    'converted_ligand': str(converted),
                }
                rows.append(row)
                print(f'{pdbid} {pose_model}: {converted} align_rmsd={align_rmsd:.3f}')
            except Exception as exc:
                print(f'ERROR {pdbid} {af_pdb}: {exc}')

    Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.scores_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r['target'], int(r['af_rank']))):
            writer.writerow(row)
    print(f'scores -> {args.scores_csv}')


if __name__ == '__main__':
    main()
