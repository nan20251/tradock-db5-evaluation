#!/usr/bin/env python3
"""
Convert OpenFold3 / AlphaFold 3 outputs into PPCBench/TraDock pose layout.

Supports both layouts under --af3_root:

OpenFold3 (default / recommended):
  <root>/<PDB>/seed_<seed>/<PDB>_seed_<seed>_sample_<k>_model.cif
  <root>/<PDB>/seed_<seed>/<PDB>_seed_<seed>_sample_<k>_confidences_aggregated.json

Official AlphaFold 3:
  <root>/<PDB>/seed-<seed>_sample-<k>/<job>_seed-<seed>_sample-<k>_model.cif
  ..._summary_confidences.json / ranking_scores.csv

Receptor CA atoms are aligned onto the DB5 reference receptor; the same
transform is applied to the predicted ligand, and only the ligand pose is
written:

  <results_root>/<dataset>/of3_1/<PDB>/<PDB>_of3_1_id.pdb
  ...

Samples are pooled across seeds, ranked by ranking_score / sample_ranking_score,
and Top-N kept.
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
    global PDBIO, PDBParser, MMCIFParser, Superimposer, Chain, Model, Residue, Structure
    from Bio.PDB import PDBIO, PDBParser, MMCIFParser, Superimposer
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


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def confidence_score(scores):
    for key in ('ranking_score', 'ranking_confidence', 'iptm', 'ptm'):
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
            str(item['cif']),
        ),
        reverse=True,
    )


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
    af_by_id = {chain.id: chain for chain in af_chains}

    if all(cid in af_by_id for cid in rec_chains + lig_chains):
        rec_segments = [standard_residues(af_by_id[cid]) for cid in rec_chains]
        lig_segments = [standard_residues(af_by_id[cid]) for cid in lig_chains]
        return rec_segments, lig_segments, 'chain_id_match'

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


def load_structure(path):
    path = Path(path)
    if path.suffix.lower() in ('.cif', '.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return first_model(parser.get_structure(path.stem, str(path)))


def convert_one(pdbid, target, af_cif, ppc_root, dataset, results_root, pose_model):
    af_model = load_structure(af_cif)
    af_chains = ordered_chains(af_model)

    rec_ref, lig_ref = reference_paths(ppc_root, dataset, pdbid)
    native_rec_model = load_structure(rec_ref)
    native_lig_model = load_structure(lig_ref)
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


def target_dirs_for(af3_root, pdbid):
    root = Path(af3_root)
    exact = root / pdbid
    if exact.is_dir():
        return [exact]
    sanitized = root / re.sub(r'[^A-Za-z0-9._-]+', '_', pdbid)
    if sanitized.is_dir():
        return [sanitized]
    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and pdbid.lower() in p.name.lower()
    ]
    return dirs or ([root] if root.is_dir() else [])


def find_sample_cifs(target_dir):
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
    # OpenFold3: ..._seed_42_sample_1_model.cif  or  seed_42/...
    m = re.search(r'seed[_-](\d+).*?sample[_-](\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'seed[_-](\d+)', text)
    if m:
        return int(m.group(1)), None
    return None, None


def _first_finite(data, keys, default=math.nan):
    for key in keys:
        if key in data and finite(data[key]):
            return data[key]
    return default


def read_summary_confidences(cif_path):
    cif_path = Path(cif_path)
    stem = cif_path.name
    for suffix in ('_model.cif', '_model.pdb', '.cif', '.pdb'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates = [
        cif_path.with_name(stem + '_confidences_aggregated.json'),  # OpenFold3
        cif_path.with_name(stem + '_summary_confidences.json'),     # AF3
        cif_path.parent / (stem + '_confidences_aggregated.json'),
        cif_path.parent / (stem + '_summary_confidences.json'),
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as handle:
            data = json.load(handle)
        ranking = _first_finite(
            data,
            ('sample_ranking_score', 'ranking_score'),
        )
        return {
            'ranking_score': ranking,
            'iptm': _first_finite(data, ('iptm',)),
            'ptm': _first_finite(data, ('ptm',)),
            'fraction_disordered': _first_finite(
                data, ('fraction_disordered', 'disorder')
            ),
            'has_clash': _first_finite(data, ('has_clash',)),
        }
    return {
        'ranking_score': math.nan,
        'iptm': math.nan,
        'ptm': math.nan,
        'fraction_disordered': math.nan,
        'has_clash': math.nan,
    }


def load_ranking_scores_csv(target_dir):
    target_dir = Path(target_dir)
    candidates = list(target_dir.glob('*ranking_scores.csv')) + list(target_dir.glob('ranking_scores.csv'))
    table = {}
    for path in candidates:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    seed = int(row.get('seed') or row.get('model_seed') or row.get('rng_seed'))
                    sample = int(row.get('sample') or row.get('diffusion_sample') or 0)
                    score = float(row.get('ranking_score') or row.get('score'))
                except (TypeError, ValueError):
                    continue
                table[(seed, sample)] = score
        if table:
            break
    return table


def read_af3_scores(cif_path, ranking_table):
    seed, sample = parse_seed_sample(cif_path)
    scores = read_summary_confidences(cif_path)
    if seed is not None and sample is not None and (seed, sample) in ranking_table:
        scores['ranking_score'] = ranking_table[(seed, sample)]
    # Alias for compare_af_tradock.py which expects ranking_confidence.
    scores['ranking_confidence'] = scores.get('ranking_score', math.nan)
    scores['af_seed'] = seed if seed is not None else ''
    scores['af_sample'] = sample if sample is not None else ''
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc_root', required=True)
    parser.add_argument('--dataset', default='DB5', choices=['DB5', 'DB5-u'])
    parser.add_argument('--dataset_json', required=True)
    parser.add_argument('--af3_root', required=True)
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--scores_csv', required=True)
    parser.add_argument('--targets', default='',
                        help='Comma-separated target IDs; default scans all.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max_models', type=int, default=5)
    parser.add_argument(
        '--pose_prefix',
        default='of3',
        help='Pose model prefix written for TraDock (default: of3). Use af3 for DeepMind AF3.',
    )
    args = parser.parse_args()

    load_biopython()

    targets = load_targets(args.dataset_json)
    selected = [x.strip().upper() for x in args.targets.split(',') if x.strip()]
    if not selected:
        selected = sorted(targets)
    if args.limit:
        selected = selected[:args.limit]

    fieldnames = [
        'target', 'pose_model', 'af_rank', 'af_seed', 'af_sample',
        'ranking_score', 'ranking_confidence', 'iptm', 'ptm',
        'fraction_disordered', 'has_clash',
        'receptor_align_rmsd', 'mapping_mode', 'source_cif',
        'converted_ligand',
    ]
    rows = []
    for pdbid in selected:
        if pdbid not in targets:
            print(f'skip missing target metadata: {pdbid}')
            continue
        target_dirs = target_dirs_for(args.af3_root, pdbid)
        if not target_dirs:
            print(f'skip no OF3/AF3 dir: {pdbid}')
            continue

        candidates = []
        for target_dir in target_dirs:
            ranking_table = load_ranking_scores_csv(target_dir)
            for cif_path in find_sample_cifs(target_dir):
                scores = read_af3_scores(cif_path, ranking_table)
                candidates.append({'cif': cif_path, 'scores': scores})

        if not candidates:
            print(f'skip no OF3/AF3 models: {pdbid}')
            continue

        selected_candidates = sort_af_candidates(candidates)[:args.max_models]
        for selected_rank, candidate in enumerate(selected_candidates, 1):
            cif_path = candidate['cif']
            scores = candidate['scores']
            pose_model = f'{args.pose_prefix}_{selected_rank}'
            try:
                pose_model, converted, align_rmsd, mapping_mode = convert_one(
                    pdbid, targets[pdbid], cif_path, args.ppc_root,
                    args.dataset, args.results_root, pose_model
                )
                row = {
                    'target': pdbid,
                    'pose_model': pose_model,
                    'af_rank': selected_rank,
                    'af_seed': scores['af_seed'],
                    'af_sample': scores['af_sample'],
                    'ranking_score': scores['ranking_score'],
                    'ranking_confidence': scores['ranking_confidence'],
                    'iptm': scores['iptm'],
                    'ptm': scores['ptm'],
                    'fraction_disordered': scores['fraction_disordered'],
                    'has_clash': scores['has_clash'],
                    'receptor_align_rmsd': align_rmsd,
                    'mapping_mode': mapping_mode,
                    'source_cif': str(cif_path),
                    'converted_ligand': str(converted),
                }
                rows.append(row)
                print(f'{pdbid} {pose_model}: {converted} align_rmsd={align_rmsd:.3f}')
            except Exception as exc:
                print(f'ERROR {pdbid} {cif_path}: {exc}')

    Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.scores_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r['target'], int(r['af_rank']))):
            writer.writerow(row)
    print(f'scores -> {args.scores_csv}')


if __name__ == '__main__':
    main()
