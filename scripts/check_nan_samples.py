#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect selected receptor/ligand PLY pairs for NaN/Inf and rSASA outliers."""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformerdock.utils.data import FEATURE_NAMES, prepare_complex


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')
DEFAULT_SAMPLES = ['1yk0_A_B', '1u0c_A_B']


def parse_args():
    parser = argparse.ArgumentParser(description='检查指定或抽样 PLY 对是否包含 NaN/Inf')
    parser.add_argument(
        'samples',
        nargs='*',
        help='样本名前缀，例如 1yk0_A_B；未指定时先查旧问题样本，找不到则自动抽样',
    )
    parser.add_argument(
        '--data_dir',
        default=DEFAULT_DATA_DIR,
        help='表面 PLY 目录，默认读取 DIPS_SURFACES 或 /root/autodl-tmp/dips_with_sasa_full',
    )
    parser.add_argument('--limit', type=int, default=5, help='自动抽样数量')
    return parser.parse_args()


def collect_sample_names(data_dir, requested, limit):
    names = requested or DEFAULT_SAMPLES
    existing = [
        name for name in names
        if os.path.exists(os.path.join(data_dir, f'{name}_receptor.ply'))
        and os.path.exists(os.path.join(data_dir, f'{name}_ligand.ply'))
    ]
    if existing:
        return existing

    rec_files = sorted(glob.glob(os.path.join(data_dir, '*_receptor.ply')))[:limit]
    return [
        os.path.basename(path).replace('_receptor.ply', '')
        for path in rec_files
        if os.path.exists(path.replace('_receptor.ply', '_ligand.ply'))
    ]


def describe_graph(label, data):
    print(f"\n{label}:")
    print(f"  节点数: {data.x.shape[0]}")
    print(f"  特征维度: {data.x.shape[1]}")
    print(f"  特征范围: [{data.x.min():.4f}, {data.x.max():.4f}]")
    has_nan = bool(torch.isnan(data.x).any().item())
    has_inf = bool(torch.isinf(data.x).any().item())
    print(f"  有 NaN: {has_nan}")
    print(f"  有 Inf: {has_inf}")

    ok = not has_nan and not has_inf
    print("\n  各维度统计:")
    for dim in range(data.x.shape[1]):
        values = data.x[:, dim]
        name = FEATURE_NAMES[dim] if dim < len(FEATURE_NAMES) else f'feat_{dim}'
        dim_nan = bool(torch.isnan(values).any().item())
        dim_inf = bool(torch.isinf(values).any().item())
        print(
            f"    {dim:2d}. {name:20s} "
            f"[{values.min():.4f}, {values.max():.4f}], "
            f"mean={values.mean():.4f}, std={values.std(unbiased=False):.4f}"
        )
        if dim_nan or dim_inf:
            ok = False
            print(f"        [错误] NaN={dim_nan}, Inf={dim_inf}")
        if name == 'rSASA':
            print(
                f"        rSASA > 2: {int((values > 2).sum().item())}, "
                f"> 5: {int((values > 5).sum().item())}, "
                f"> 10: {int((values > 10).sum().item())}"
            )
    return ok


def main():
    args = parse_args()

    print("=" * 60)
    print("检查 NaN/Inf 问题样本")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")

    sample_names = collect_sample_names(args.data_dir, args.samples, args.limit)
    if not sample_names:
        print("[跳过] 未找到 receptor/ligand PLY 对。AutoDL 上请设置 DIPS_SURFACES。")
        return 0

    ok = True
    for sample in sample_names:
        rec_file = os.path.join(args.data_dir, f'{sample}_receptor.ply')
        lig_file = os.path.join(args.data_dir, f'{sample}_ligand.ply')
        print(f"\n{'=' * 60}")
        print(sample)
        print(f"{'=' * 60}")
        try:
            rec, lig = prepare_complex(rec_file, lig_file, in_channels=11)
            ok = describe_graph('Receptor', rec) and ok
            ok = describe_graph('Ligand', lig) and ok
        except Exception as exc:
            ok = False
            print(f"[错误] {sample}: {exc}")

    print("\n" + "=" * 60)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
