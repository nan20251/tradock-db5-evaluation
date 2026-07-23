"""Crop protein surface PLY files to interface nodes near a partner structure."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Set, Union

import numpy as np
from plyfile import PlyData, PlyElement

ChainSpec = Union[str, Iterable[str], None]


def _normalize_chains(chains: ChainSpec) -> Optional[Set[str]]:
    if chains is None:
        return None
    if isinstance(chains, str):
        return set(chains.replace(',', '').replace(' ', ''))
    return {str(c) for c in chains}


def read_pdb_coords(pdb_file: str, chains: ChainSpec = None) -> np.ndarray:
    """Read heavy-atom coordinates from a PDB file, optionally filtered by chain."""
    chain_set = _normalize_chains(chains)
    coords = []
    with open(pdb_file, 'r', errors='ignore') as handle:
        for line in handle:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if chain_set is not None and len(line) > 21 and line[21] not in chain_set:
                continue
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
    if not coords:
        raise ValueError(f'No coordinates found in {pdb_file}')
    return np.asarray(coords, dtype=np.float64)


def binding_site_mask(
    surface_coords: np.ndarray,
    partner_coords: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """True for surface nodes within threshold Angstrom of any partner atom."""
    if len(surface_coords) == 0:
        return np.zeros((0,), dtype=bool)
    if len(partner_coords) == 0:
        return np.zeros((len(surface_coords),), dtype=bool)
    try:
        from scipy.spatial import cKDTree
        dists, _ = cKDTree(partner_coords).query(surface_coords, k=1)
        return np.asarray(dists) < threshold
    except Exception:
        # Fallback for tiny inputs / missing scipy
        diff = surface_coords[:, None, :] - partner_coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        return dist.min(axis=1) < threshold


def _remap_faces(face_data, mask: np.ndarray):
    new_idx = np.full(len(mask), -1, dtype=np.int64)
    new_idx[mask] = np.arange(mask.sum())

    kept_faces = []
    for face in face_data:
        indices = np.asarray(face, dtype=np.int64).reshape(-1)
        if indices.size != 3:
            continue
        if np.all(mask[indices]):
            kept_faces.append(new_idx[indices])
    if not kept_faces:
        return None
    return np.asarray(kept_faces, dtype=np.int32)


def crop_ply_file(
    ply_in: str,
    ply_out: str,
    partner_pdb: str,
    threshold: float = 10.0,
    partner_chains: ChainSpec = None,
) -> dict:
    """
    Keep only surface vertices within threshold Angstrom of partner atoms.

    Returns a stats dict with original/kept vertex and face counts.
    """
    ply = PlyData.read(ply_in)
    verts = ply['vertex'].data
    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
    partner_coords = read_pdb_coords(partner_pdb, partner_chains)
    mask = binding_site_mask(xyz, partner_coords, threshold)

    kept_verts = verts[mask]
    n_orig = len(verts)
    n_kept = len(kept_verts)
    if n_kept == 0:
        raise ValueError(
            f'No interface nodes kept for {ply_in} '
            f'(partner={partner_pdb}, threshold={threshold})'
        )

    elements = [PlyElement.describe(kept_verts, 'vertex')]
    n_faces_orig = 0
    n_faces_kept = 0
    if 'face' in ply and len(ply['face']) > 0:
        n_faces_orig = len(ply['face'])
        remapped = _remap_faces(ply['face']['vertex_indices'], mask)
        if remapped is not None:
            n_faces_kept = len(remapped)
            face_array = np.empty(
                n_faces_kept,
                dtype=[('vertex_indices', 'i4', (3,))],
            )
            face_array['vertex_indices'] = remapped
            elements.append(PlyElement.describe(face_array, 'face'))

    os.makedirs(os.path.dirname(os.path.abspath(ply_out)) or '.', exist_ok=True)
    PlyData(elements, text=True).write(ply_out)

    return {
        'input': ply_in,
        'output': ply_out,
        'partner_pdb': partner_pdb,
        'threshold': threshold,
        'vertices_before': n_orig,
        'vertices_after': n_kept,
        'vertices_kept_pct': 100.0 * n_kept / n_orig,
        'faces_before': n_faces_orig,
        'faces_after': n_faces_kept,
    }
