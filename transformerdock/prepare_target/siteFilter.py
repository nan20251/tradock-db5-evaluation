"""Oracle / predicted binding-site helpers for decoy filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ResidueKey = Tuple[str, int, str]  # chain, resseq, icode


@dataclass
class AtomRecord:
    chain: str
    resseq: int
    icode: str
    resname: str
    coord: np.ndarray  # (3,)


def _parse_resseq(line: str) -> Tuple[int, str]:
    raw = line[22:26]
    icode = line[26] if len(line) > 26 else ' '
    try:
        resseq = int(raw)
    except ValueError:
        resseq = int(raw.strip() or 0)
    return resseq, icode


def read_pdb_atoms(
    pdb_file: str,
    chains: Optional[Iterable[str]] = None,
    heavy_only: bool = True,
) -> List[AtomRecord]:
    """Read ATOM/HETATM records as AtomRecord list."""
    chain_set = None if chains is None else set(chains)
    atoms: List[AtomRecord] = []
    with open(pdb_file, 'r', errors='ignore') as handle:
        for line in handle:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if len(line) < 54:
                continue
            chain = line[21] if len(line) > 21 else ' '
            if chain_set is not None and chain not in chain_set:
                continue
            name = line[12:16]
            if heavy_only and name.strip().startswith('H'):
                continue
            element = line[76:78].strip() if len(line) >= 78 else ''
            if heavy_only and element == 'H':
                continue
            resseq, icode = _parse_resseq(line)
            atoms.append(
                AtomRecord(
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    resname=line[17:20].strip(),
                    coord=np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=np.float64,
                    ),
                )
            )
    if not atoms:
        raise ValueError(f'No atoms found in {pdb_file}')
    return atoms


def residue_keys(atoms: Sequence[AtomRecord]) -> Set[ResidueKey]:
    return {(a.chain, a.resseq, a.icode) for a in atoms}


def _coords_and_res_index(atoms: Sequence[AtomRecord]):
    coords = np.stack([a.coord for a in atoms], axis=0)
    keys = [(a.chain, a.resseq, a.icode) for a in atoms]
    return coords, keys


def interface_residue_keys(
    self_atoms: Sequence[AtomRecord],
    partner_atoms: Sequence[AtomRecord],
    cutoff: float = 5.0,
) -> Set[ResidueKey]:
    """Residues on self with any heavy atom within cutoff of partner."""
    if not self_atoms or not partner_atoms:
        return set()
    self_xyz, self_keys = _coords_and_res_index(self_atoms)
    partner_xyz, _ = _coords_and_res_index(partner_atoms)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(partner_xyz)
        dists, _ = tree.query(self_xyz, k=1)
        near = np.asarray(dists) < cutoff
    except Exception:
        diff = self_xyz[:, None, :] - partner_xyz[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)
        near = dists < cutoff
    return {self_keys[i] for i, flag in enumerate(near) if flag}


def oracle_receptor_site(
    native_rec_pdb: str,
    native_lig_pdb: str,
    cutoff: float = 5.0,
) -> Set[ResidueKey]:
    """Native receptor residues within cutoff of native ligand."""
    rec = read_pdb_atoms(native_rec_pdb)
    lig = read_pdb_atoms(native_lig_pdb)
    return interface_residue_keys(rec, lig, cutoff=cutoff)


def decoy_site_overlap(
    decoy_rec_pdb: str,
    decoy_lig_pdb: str,
    site: Set[ResidueKey],
    contact_cutoff: float = 8.0,
    match_by_resseq: bool = False,
) -> dict:
    """
    Measure how much a decoy ligand contacts the oracle receptor site.

    site_frac = |contacting_site_residues| / |site|
    where contacting_site_residues are site residues with any atom within
    contact_cutoff of the decoy ligand.
    """
    if not site:
        return {
            'n_site': 0,
            'n_hit': 0,
            'site_frac': 0.0,
            'n_lig_near_site': 0,
            'lig_near_frac': 0.0,
        }

    rec_atoms = read_pdb_atoms(decoy_rec_pdb)
    lig_atoms = read_pdb_atoms(decoy_lig_pdb)
    lig_xyz = np.stack([a.coord for a in lig_atoms], axis=0)

    if match_by_resseq:
        site_resseq = {(r[1], r[2]) for r in site}
        site_atoms = [
            a for a in rec_atoms if (a.resseq, a.icode) in site_resseq
        ]
        site_keys_present = {(a.chain, a.resseq, a.icode) for a in site_atoms}
        n_site = len(site_resseq)
    else:
        site_atoms = [
            a for a in rec_atoms if (a.chain, a.resseq, a.icode) in site
        ]
        site_keys_present = {(a.chain, a.resseq, a.icode) for a in site_atoms}
        n_site = len(site)

    if not site_atoms:
        return {
            'n_site': n_site,
            'n_hit': 0,
            'site_frac': 0.0,
            'n_lig_near_site': 0,
            'lig_near_frac': 0.0,
            'n_site_on_decoy_rec': 0,
        }

    site_xyz = np.stack([a.coord for a in site_atoms], axis=0)
    site_atom_keys = [(a.chain, a.resseq, a.icode) for a in site_atoms]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(lig_xyz)
        dists, _ = tree.query(site_xyz, k=1)
        near_site = np.asarray(dists) < contact_cutoff
        tree_site = cKDTree(site_xyz)
        lig_dists, _ = tree_site.query(lig_xyz, k=1)
        near_lig = np.asarray(lig_dists) < contact_cutoff
    except Exception:
        diff = site_xyz[:, None, :] - lig_xyz[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)
        near_site = dists < contact_cutoff
        diff2 = lig_xyz[:, None, :] - site_xyz[None, :, :]
        lig_dists = np.sqrt((diff2 ** 2).sum(axis=-1)).min(axis=1)
        near_lig = lig_dists < contact_cutoff

    hit_keys = {site_atom_keys[i] for i, flag in enumerate(near_site) if flag}
    if match_by_resseq:
        hit_resseq = {(k[1], k[2]) for k in hit_keys}
        n_hit = len(hit_resseq)
        denom = max(1, len({(r[1], r[2]) for r in site}))
    else:
        n_hit = len(hit_keys)
        denom = max(1, n_site)

    n_lig_near = int(near_lig.sum())
    return {
        'n_site': n_site,
        'n_hit': n_hit,
        'site_frac': float(n_hit) / float(denom),
        'n_lig_near_site': n_lig_near,
        'lig_near_frac': float(n_lig_near) / float(max(1, len(lig_atoms))),
        'n_site_on_decoy_rec': len(site_keys_present),
    }


def keep_by_site_frac(site_frac: float, threshold: float = 0.3) -> bool:
    return float(site_frac) >= float(threshold)
