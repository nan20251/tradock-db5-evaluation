#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check model feature dimensions read from PLY surface files."""

import argparse
import glob
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformerdock.utils.data import FEATURE_NAMES, read_ply


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')


def parse_args():
    parser = argparse.ArgumentParser(description='检查 PLY 读入后的模型特征维度')
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=DEFAULT_DATA_DIR,
        help='表面 PLY 目录，默认读取 DIPS_SURFACES 或 /root/autodl-tmp/dips_with_sasa_full',
    )
    parser.add_argument('--limit', type=int, default=5, help='最多检查多少个 receptor 文件')
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.data_dir, '*_receptor.ply')
    files = sorted(glob.glob(pattern))[:args.limit]

    print("=" * 60)
    print("检查 PLY 模型特征维度")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")

    if not files:
        print(f"[跳过] 未找到 PLY 文件: {pattern}")
        print("AutoDL 上请确认 DIPS_SURFACES 指向完整 surfaces 目录。")
        return 0

    print(f"\n找到 {len(files)} 个测试文件\n")
    ok = True
    for ply_file in files:
        print(f"文件: {ply_file}")
        try:
            data = read_ply(ply_file)
            feat_dim = int(data.x.shape[1])
            n_edges = int(data.edge_index.shape[1]) if data.edge_index is not None else 0
            print(f"  读取维度: {feat_dim}")
            print(f"  节点数: {data.x.shape[0]}")
            print(f"  边数: {n_edges}")
            print(f"  特征值范围: [{data.x.min():.4f}, {data.x.max():.4f}]")

            if feat_dim != len(FEATURE_NAMES):
                ok = False
                print(f"  [错误] 期望 {len(FEATURE_NAMES)} 维，实际 {feat_dim} 维")
            else:
                rsasa_col = data.x[:, FEATURE_NAMES.index('rSASA')]
                nonzero = int((rsasa_col > 0).sum().item())
                ratio = nonzero * 100.0 / max(1, len(rsasa_col))
                print(f"  rSASA 非零数: {nonzero}/{len(rsasa_col)} ({ratio:.1f}%)")
                print(f"  rSASA 范围: [{rsasa_col.min():.4f}, {rsasa_col.max():.4f}]")
            print()
        except Exception as exc:
            ok = False
            print(f"  [错误] 读取失败: {exc}\n")

    print("=" * 60)
    print(f"期望模型输入维度: {len(FEATURE_NAMES)}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {i:2d}. {name}")
    print("说明: 原始 PLY 通常有 14 个 vertex 字段，其中 x/y/z 是坐标，模型特征为其余 11 维。")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
