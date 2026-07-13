#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a filtered DIPS pairs.csv by excluding bad samples and PDB prefixes."""

import argparse
import csv
from pathlib import Path


DEFAULT_EXCLUDES = {'1u0c_A_B', '1yk0_A_B'}


def parse_args():
    parser = argparse.ArgumentParser(description='过滤 DIPS pairs.csv 中的坏样本')
    parser.add_argument('--input', required=True, help='输入 pairs.csv')
    parser.add_argument('--output', required=True, help='输出过滤后的 pairs.csv')
    parser.add_argument('--exclude', nargs='*', default=sorted(DEFAULT_EXCLUDES),
                        help='要排除的样本名，默认排除已知 rSASA NaN 样本')
    parser.add_argument('--exclude_file', default=None,
                        help='按 PDB ID 前缀排除的文本文件，如 data/dips/exclude_capri.txt')
    return parser.parse_args()


def load_exclude_prefixes(path):
    if not path:
        return set()
    prefixes = set()
    with Path(path).open() as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            for token in line.replace(',', ' ').split():
                token = token.strip().lower()
                if token:
                    prefixes.add(token[:4])
    return prefixes


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    excludes = set(args.exclude)
    exclude_prefixes = load_exclude_prefixes(args.exclude_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed_exact = 0
    removed_prefix = 0
    with in_path.open(newline='') as src, out_path.open('w', newline='') as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise SystemExit(f'[错误] 空 CSV: {in_path}')
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            name = (row.get('name') or row.get('pdb_id') or '').strip()
            if name in excludes:
                removed_exact += 1
                continue
            if name[:4].lower() in exclude_prefixes:
                removed_prefix += 1
                continue
            writer.writerow(row)
            kept += 1

    print(f'[OK] 写入 {out_path}')
    print(f'  保留: {kept}')
    print(f'  删除坏样本: {removed_exact} ({", ".join(sorted(excludes))})')
    if args.exclude_file:
        print(f'  删除排除前缀: {removed_prefix} ({len(exclude_prefixes)} prefixes from {args.exclude_file})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
