#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check feature ranges and non-finite values in surface PLY files."""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformerdock.utils.data import FEATURE_NAMES, read_ply


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')


def parse_args():
    parser = argparse.ArgumentParser(description='检查 PLY 模型特征范围和 NaN/Inf')
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=DEFAULT_DATA_DIR,
        help='表面 PLY 目录，默认读取 DIPS_SURFACES 或 /root/autodl-tmp/dips_with_sasa_full',
    )
    parser.add_argument('--limit', type=int, default=10, help='最多抽样多少个 receptor 文件')
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.data_dir, '*_receptor.ply')
    files = sorted(glob.glob(pattern))[:args.limit]

    print("=" * 60)
    print("检查数据特征范围")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")

    if not files:
        print(f"[跳过] 未找到 PLY 文件: {pattern}")
        print("AutoDL 上请确认 DIPS_SURFACES 指向完整 surfaces 目录。")
        return 0

    all_features = []
    failed = []
    for ply_file in files:
        try:
            all_features.append(read_ply(ply_file).x)
        except Exception as exc:
            failed.append((ply_file, str(exc)))

    if not all_features:
        print("[错误] 抽样文件全部读取失败。")
        for path, err in failed:
            print(f"  {path}: {err}")
        return 1

    features = torch.cat(all_features, dim=0)
    print(f"\n抽样文件数: {len(all_features)}")
    print(f"总顶点数: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    print()

    ok = True
    print("各维度统计:")
    print("-" * 60)
    for i in range(features.shape[1]):
        col = features[:, i]
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'feat_{i}'
        has_nan = bool(torch.isnan(col).any().item())
        has_inf = bool(torch.isinf(col).any().item())
        if has_nan or has_inf:
            ok = False
            print(f"{i:2d}. {name:20s} [错误] NaN={has_nan}, Inf={has_inf}")
        else:
            print(
                f"{i:2d}. {name:20s} "
                f"min={col.min():.4f}, max={col.max():.4f}, "
                f"mean={col.mean():.4f}, std={col.std(unbiased=False):.4f}"
            )

    if failed:
        ok = False
        print("\n读取失败文件:")
        for path, err in failed:
            print(f"  {path}: {err}")

    print()
    if ok:
        print("[OK] 抽样数据无 NaN/Inf")
    else:
        print("[错误] 抽样数据存在问题")

    print("=" * 60)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
