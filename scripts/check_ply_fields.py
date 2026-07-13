#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect raw vertex fields in PLY surface files."""

import argparse
import glob
import os
from pathlib import Path

from plyfile import PlyData


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')
EXPECTED_VERTEX_FIELDS = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'charge', 'hydrophobicity',
    'hbond_donor', 'hbond_acceptor',
    'curvature', 'shape_index',
    'aa_polar', 'rSASA',
]


def parse_args():
    parser = argparse.ArgumentParser(description='检查 PLY 原始 vertex 字段')
    parser.add_argument(
        'paths',
        nargs='*',
        help='PLY 文件或目录。未指定时读取 DIPS_SURFACES 或 /root/autodl-tmp/dips_with_sasa_full',
    )
    parser.add_argument('--limit', type=int, default=3, help='目录模式下最多检查多少个 receptor 文件')
    return parser.parse_args()


def collect_files(paths, limit):
    if not paths:
        paths = [DEFAULT_DATA_DIR]
    files = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files.extend(sorted(glob.glob(str(p / '*_receptor.ply')))[:limit])
        elif p.is_file():
            files.append(str(p))
    return files


def main():
    args = parse_args()
    files = collect_files(args.paths, args.limit)

    print("=" * 60)
    print("检查 PLY 原始 vertex 字段")
    print("=" * 60)

    if not files:
        print("[跳过] 未找到 PLY 文件。AutoDL 上请设置 DIPS_SURFACES=/path/to/surfaces。")
        return 0

    ok = True
    for ply_file in files:
        print(f"\n文件: {ply_file}")
        print("-" * 60)
        try:
            ply = PlyData.read(ply_file)
            verts = ply['vertex']
            names = list(verts.data.dtype.names or [])

            print("PLY 文件中的 vertex 字段:")
            for i, name in enumerate(names, 1):
                print(f"  {i:2d}. {name}")
            print(f"\n  总 vertex 字段数: {len(names)}")

            missing = [field for field in EXPECTED_VERTEX_FIELDS if field not in names]
            if missing:
                ok = False
                print(f"  [错误] 缺失字段: {', '.join(missing)}")
            else:
                print("  [OK] 14 个期望 vertex 字段都存在")

            if len(verts.data) > 0:
                print("\n  第一个顶点:")
                first_vertex = verts.data[0]
                for name in names:
                    print(f"    {name:20s} = {first_vertex[name]}")
        except Exception as exc:
            ok = False
            print(f"  [错误] 读取失败: {exc}")

    print("\n" + "=" * 60)
    print("说明: x/y/z 是坐标字段；模型输入特征为其余 11 维。")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
