"""
从 final_raw_dips.tar.gz 流式提取 metadata.csv（不需要全部解压）。
直接从 tar 中逐个读取 .dill 文件，提取 pdb_id / rec_chains / lig_chains。
"""
import os
import sys
import csv
import tarfile
import io
from collections import OrderedDict

import dill
import pandas as pd


def extract_metadata_from_tar(tar_path, output_csv, limit=None):
    seen = OrderedDict()
    n_ok = 0
    n_fail = 0

    with tarfile.open(tar_path, 'r:gz') as tar:
        for i, member in enumerate(tar):
            if not member.name.endswith('.dill'):
                continue

            if limit and n_ok >= limit:
                break

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                obj = dill.load(f)

                # obj[0] = "5cxc.pdb1", obj[1] = rec DataFrame, obj[2] = lig DataFrame
                pdb_name = obj[0]  # e.g. "5cxc.pdb1"
                pdb_id = pdb_name.split('.')[0].lower()  # "5cxc"

                rec_df = obj[1]
                lig_df = obj[2]

                rec_chains = ''.join(sorted(rec_df['chain'].unique().tolist()))
                lig_chains = ''.join(sorted(lig_df['chain'].unique().tolist()))

                key = (pdb_id, rec_chains, lig_chains)
                if key not in seen:
                    seen[key] = True

                n_ok += 1

            except Exception as e:
                n_fail += 1
                if n_fail <= 10:
                    print(f'  Error: {member.name}: {e}')

            total = n_ok + n_fail
            if total % 500 == 0:
                print(f'  [{total}] ok={n_ok} fail={n_fail} unique_pairs={len(seen)}')

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    results = sorted(seen.keys())
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pdb_id', 'rec_chains', 'lig_chains'])
        for pid, rec, lig in results:
            w.writerow([pid, rec, lig])

    print(f'\nDone: {output_csv}')
    print(f'  Total .dill processed: {n_ok}')
    print(f'  Failed: {n_fail}')
    print(f'  Unique complexes: {len(results)}')


if __name__ == '__main__':
    tar_path = r'C:\Users\57102\Desktop\final_raw_dips.tar.gz'
    output_csv = r'C:\Users\57102\Desktop\TraDock\data\dips\metadata.csv'

    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        print(f'Limit: {limit}')

    print(f'Reading: {tar_path}')
    print(f'Output: {output_csv}')
    extract_metadata_from_tar(tar_path, output_csv, limit=limit)
