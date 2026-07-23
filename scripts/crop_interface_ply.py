#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crop receptor/ligand surface PLY files to interface nodes.

Interface nodes = surface vertices within --threshold Angstrom of the partner chain.

Examples
--------
# Existing full surfaces + partner PDBs
python scripts/crop_interface_ply.py \\
  --rec-ply rec_full.ply --lig-ply lig_full.ply \\
  --rec-partner-pdb lig.pdb --lig-partner-pdb rec.pdb \\
  --out-dir cropped/

# One decoy complex PDB (chains A/B)
python scripts/crop_interface_ply.py \\
  --decoy-pdb decoy.pdb --rec-chain A --lig-chain B \\
  --out-dir cropped/

# Generate full surfaces from monomer PDBs, then crop
python scripts/crop_interface_ply.py \\
  --rec-pdb rec.pdb --lig-pdb lig.pdb \\
  --out-dir cropped/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.prep_dockground_decoys import split_chains
from examples.surface_gen import pdb_to_surface_ply
from transformerdock.prepare_target.cropInterface import crop_ply_file


def _ensure_surface(pdb_path: str, ply_path: str, voxel_size: float, verbose: bool) -> None:
    if os.path.exists(ply_path):
        return
    ok = pdb_to_surface_ply(
        pdb_path,
        ply_path,
        voxel_size=voxel_size,
        verbose=verbose,
    )
    if not ok:
        raise RuntimeError(f'Failed to generate surface: {pdb_path} -> {ply_path}')


def _print_stats(label: str, stats: dict) -> None:
    print(
        f'{label}: '
        f'{stats["vertices_before"]} -> {stats["vertices_after"]} nodes '
        f'({stats["vertices_kept_pct"]:.1f}%), '
        f'{stats["faces_before"]} -> {stats["faces_after"]} faces'
    )
    print(f'  wrote {stats["output"]}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop surface PLY files to interface nodes near the partner chain.',
    )
    parser.add_argument('--rec-ply', help='Full receptor .ply (skip generation)')
    parser.add_argument('--lig-ply', help='Full ligand .ply (skip generation)')
    parser.add_argument('--rec-pdb', help='Receptor PDB (generate full surface if needed)')
    parser.add_argument('--lig-pdb', help='Ligand PDB (generate full surface if needed)')
    parser.add_argument(
        '--decoy-pdb',
        help='Complex decoy PDB; split into rec/lig by --rec-chain / --lig-chain',
    )
    parser.add_argument('--rec-chain', default='A', help='Receptor chain id(s), e.g. A or AB')
    parser.add_argument('--lig-chain', default='B', help='Ligand chain id(s), e.g. B')
    parser.add_argument(
        '--rec-partner-pdb',
        help='Partner PDB used to crop receptor (default: ligand PDB)',
    )
    parser.add_argument(
        '--lig-partner-pdb',
        help='Partner PDB used to crop ligand (default: receptor PDB)',
    )
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=10.0, help='Distance cutoff in Angstrom')
    parser.add_argument('--voxel-size', type=float, default=2.0, help='Surface generation voxel size')
    parser.add_argument('--prefix', default='iface', help='Output filename prefix')
    parser.add_argument('--keep-full', action='store_true', help='Also write uncropped full surfaces')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tmpdir = None
    rec_pdb = args.rec_pdb
    lig_pdb = args.lig_pdb

    try:
        if args.decoy_pdb:
            tmpdir = tempfile.mkdtemp(prefix='crop_iface_')
            rec_pdb = os.path.join(tmpdir, 'rec.pdb')
            lig_pdb = os.path.join(tmpdir, 'lig.pdb')
            split_chains(args.decoy_pdb, rec_pdb, lig_pdb, args.rec_chain, args.lig_chain)
            if args.verbose:
                print(f'Split decoy: {args.decoy_pdb} -> {rec_pdb}, {lig_pdb}')

        if not ((args.rec_ply and args.lig_ply) or (rec_pdb and lig_pdb)):
            raise SystemExit(
                'Provide either (--rec-ply + --lig-ply) or (--rec-pdb + --lig-pdb) '
                'or --decoy-pdb.'
            )

        rec_partner = args.rec_partner_pdb or lig_pdb or args.lig_pdb
        lig_partner = args.lig_partner_pdb or rec_pdb or args.rec_pdb
        if rec_partner is None or lig_partner is None:
            raise SystemExit('Could not infer partner PDB paths for cropping.')

        rec_full_ply = args.rec_ply or os.path.join(args.out_dir, f'{args.prefix}_rec_full.ply')
        lig_full_ply = args.lig_ply or os.path.join(args.out_dir, f'{args.prefix}_lig_full.ply')

        if rec_pdb:
            _ensure_surface(rec_pdb, rec_full_ply, args.voxel_size, args.verbose)
        if lig_pdb:
            _ensure_surface(lig_pdb, lig_full_ply, args.voxel_size, args.verbose)

        rec_out = os.path.join(args.out_dir, f'{args.prefix}_receptor_iface.ply')
        lig_out = os.path.join(args.out_dir, f'{args.prefix}_ligand_iface.ply')

        print(f'Partner for receptor crop: {rec_partner}')
        rec_stats = crop_ply_file(
            rec_full_ply,
            rec_out,
            rec_partner,
            threshold=args.threshold,
            partner_chains=args.lig_chain if args.decoy_pdb else None,
        )
        _print_stats('receptor', rec_stats)

        print(f'Partner for ligand crop: {lig_partner}')
        lig_stats = crop_ply_file(
            lig_full_ply,
            lig_out,
            lig_partner,
            threshold=args.threshold,
            partner_chains=args.rec_chain if args.decoy_pdb else None,
        )
        _print_stats('ligand', lig_stats)

        if args.keep_full and args.rec_ply is None and rec_pdb:
            print(f'  full receptor surface: {rec_full_ply}')
        if args.keep_full and args.lig_ply is None and lig_pdb:
            print(f'  full ligand surface: {lig_full_ply}')

        print('Done. Use the *_iface.ply files with prepare_complex() / TraDock scoring.')
        return 0
    finally:
        if tmpdir and not args.verbose:
            for name in ('rec.pdb', 'lig.pdb'):
                path = os.path.join(tmpdir, name)
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)


if __name__ == '__main__':
    raise SystemExit(main())
