"""
Dockground Decoy Set 1.0 → 批量生成诱饵的真实分子表面 .ply + DockQ 标签

输入：data/dockground/decoy/decoy/*.tgz（61 个复合物）
输出：
    <out_dir>/{stem}_d{decoy_id}_receptor.ply
    <out_dir>/{stem}_d{decoy_id}_ligand.ply
    <out_dir>/decoys.csv      # 全表：stem,decoy_id,name,dockq,fnat,irms,lrms,classification

注：
    - rmsd.list 列约定: id, col2, lrms, irms, fnat, fnonnat
    - DockQ = (fnat + 1/(1+(iRMS/1.5)^2) + 1/(1+(lRMS/8.5)^2)) / 3
    - decoy 内部链命名是 A=受体,B=配体（已确认）
"""

import os
import sys
import csv
import glob
import shutil
import tarfile
import tempfile
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.surface_gen import pdb_to_surface_ply


def parse_stem_chains(stem):
    """
    从 stem 解析受体/配体链集合。
    命名规则: {pdb4}_{rec_chains}_{lig_chains}
    例: 1avw_A_B → rec={'A'}, lig={'B'}
        1bth_LH_P → rec={'L','H'}, lig={'P'}
        1akj_AB_DE → rec={'A','B'}, lig={'D','E'}
    """
    parts = stem.split('_')
    if len(parts) < 3:
        return set('A'), set('B')
    rec_str = parts[1]
    lig_str = parts[2]
    return set(rec_str), set(lig_str)


def split_chains(pdb_path, out_rec_pdb, out_lig_pdb, rec_chains, lig_chains):
    """按链集合拆分 PDB。rec_chains/lig_chains 是 set 或 str。"""
    if isinstance(rec_chains, str):
        rec_chains = set(rec_chains)
    if isinstance(lig_chains, str):
        lig_chains = set(lig_chains)
    rec_lines, lig_lines = [], []
    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM', 'TER', 'ANISOU')) and len(line) > 21:
                c = line[21]
                if c in rec_chains:
                    rec_lines.append(line)
                elif c in lig_chains:
                    lig_lines.append(line)
    with open(out_rec_pdb, 'w') as f:
        f.writelines(rec_lines)
        f.write('END\n')
    with open(out_lig_pdb, 'w') as f:
        f.writelines(lig_lines)
        f.write('END\n')


def calc_dockq(fnat, irms, lrms):
    """DockQ 公式 (Basu & Wallner 2016)."""
    s_fnat = fnat
    s_irms = 1.0 / (1.0 + (irms / 1.5) ** 2)
    s_lrms = 1.0 / (1.0 + (lrms / 8.5) ** 2)
    return (s_fnat + s_irms + s_lrms) / 3.0


def classify(dockq):
    if dockq >= 0.80: return 'high'
    if dockq >= 0.49: return 'medium'
    if dockq >= 0.23: return 'acceptable'
    return 'incorrect'


def parse_rmsd_list(path):
    """
    返回 dict: decoy_id -> (lrms, irms, fnat).
    rmsd.list 每行: id col2 lrms irms fnat fnonnat
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                did = int(parts[0])
                lrms = float(parts[2])
                irms = float(parts[3])
                fnat = float(parts[4])
            except ValueError:
                continue
            out[did] = (lrms, irms, fnat)
    return out


def process_decoy(args_tuple):
    """
    单个 decoy 处理（在子进程里跑）。
    输入：(stem, tmp_target_dir, did, decoy_pdb, lrms, irms, fnat, out_dir, voxel_size)
    返回：dict 行,失败时 ok=False。
    """
    (stem, did, decoy_pdb, lrms, irms, fnat, out_dir, voxel_size) = args_tuple

    rec_ply = os.path.join(out_dir, f'{stem}_d{did}_receptor.ply')
    lig_ply = os.path.join(out_dir, f'{stem}_d{did}_ligand.ply')
    name = f'{stem}_d{did}'

    dockq = calc_dockq(fnat, irms, lrms)
    cls = classify(dockq)

    if os.path.exists(rec_ply) and os.path.exists(lig_ply):
        return {
            'ok': True, 'cached': True, 'name': name, 'stem': stem,
            'decoy_id': did, 'dockq': dockq, 'fnat': fnat,
            'irms': irms, 'lrms': lrms, 'classification': cls,
        }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_pdb = os.path.join(tmp, 'rec.pdb')
            lig_pdb = os.path.join(tmp, 'lig.pdb')
            rec_chains, lig_chains = parse_stem_chains(stem)
            split_chains(decoy_pdb, rec_pdb, lig_pdb, rec_chains, lig_chains)

            ok_r = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
            ok_l = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)

            if not ok_r or not ok_l:
                for f in (rec_ply, lig_ply):
                    if os.path.exists(f):
                        os.remove(f)
                return {'ok': False, 'name': name, 'msg': f'ply rec={ok_r} lig={ok_l}'}

        return {
            'ok': True, 'cached': False, 'name': name, 'stem': stem,
            'decoy_id': did, 'dockq': dockq, 'fnat': fnat,
            'irms': irms, 'lrms': lrms, 'classification': cls,
        }
    except Exception as e:
        for f in (rec_ply, lig_ply):
            if os.path.exists(f):
                os.remove(f)
        return {'ok': False, 'name': name, 'msg': f'异常: {e}'}


def collect_decoy_jobs(decoy_dir, extract_root, out_dir, voxel_size,
                       max_decoys_per_target=None):
    """
    解压所有 tgz 到 extract_root（保留缓存),收集所有 decoy 任务。
    返回 (jobs, target_skip_count)。
    """
    jobs = []
    for tgz in sorted(glob.glob(os.path.join(decoy_dir, '*.tgz'))):
        stem = os.path.basename(tgz).replace('.tgz', '')
        target_dir = os.path.join(extract_root, stem)
        # 解压（已存在则跳过）
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            with tarfile.open(tgz, 'r:gz') as tar:
                tar.extractall(target_dir)
        # 找 rmsd.list 与 decoys 目录
        rmsd_paths = glob.glob(os.path.join(target_dir, '**', 'rmsd.list'), recursive=True)
        if not rmsd_paths:
            print(f'  [跳过] {stem}: 找不到 rmsd.list')
            continue
        labels = parse_rmsd_list(rmsd_paths[0])
        decoys_glob = sorted(glob.glob(os.path.join(target_dir, '**', 'decoys', 'r-l_*.pdb'),
                                       recursive=True))
        if not decoys_glob:
            print(f'  [跳过] {stem}: 找不到 r-l_*.pdb')
            continue
        if max_decoys_per_target:
            decoys_glob = decoys_glob[:max_decoys_per_target]
        for decoy_pdb in decoys_glob:
            base = os.path.basename(decoy_pdb)  # r-l_<id>.pdb
            try:
                did = int(base[len('r-l_'):-len('.pdb')])
            except ValueError:
                continue
            if did not in labels:
                continue
            lrms, irms, fnat = labels[did]
            jobs.append((stem, did, decoy_pdb, lrms, irms, fnat, out_dir, voxel_size))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decoy_dir',
                   default='/root/TransformerDock/data/dockground/decoy/decoy')
    p.add_argument('--extract_root',
                   default='/root/autodl-tmp/data/dockground_extracted',
                   help='tgz 解压缓存目录')
    p.add_argument('--out_dir', required=True,
                   help='.ply 与 decoys.csv 输出目录')
    p.add_argument('--voxel_size', type=float, default=3.5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max_per_target', type=int, default=None,
                   help='每个复合物最多处理 N 个 decoy（调试用）')
    p.add_argument('--report_every', type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.extract_root, exist_ok=True)

    print(f'解压目录:    {args.extract_root}')
    print(f'输出目录:    {args.out_dir}')
    print(f'voxel_size:  {args.voxel_size}')
    print(f'workers:     {args.workers}')
    print('---')

    print('扫描并解压 tgz ...')
    jobs = collect_decoy_jobs(args.decoy_dir, args.extract_root,
                              args.out_dir, args.voxel_size,
                              args.max_per_target)
    print(f'共收集 {len(jobs)} 个 decoy 任务')
    if not jobs:
        print('[错误] 没有任务')
        sys.exit(1)

    t_start = time.time()
    rows = []
    n_ok = 0
    n_fail = 0
    fail_log = []

    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            r = process_decoy(job)
            if r['ok']:
                n_ok += 1
                rows.append(r)
            else:
                n_fail += 1
                fail_log.append((r.get('name', '?'), r.get('msg', '')))
            if i % args.report_every == 0 or i == len(jobs):
                rate = i / max(time.time() - t_start, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6) / 60
                print(f'[{i}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                      f'rate={rate:.1f}/s eta={eta:.1f}min')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futs = {exe.submit(process_decoy, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if r['ok']:
                    n_ok += 1
                    rows.append(r)
                else:
                    n_fail += 1
                    fail_log.append((r.get('name', '?'), r.get('msg', '')))
                if done % args.report_every == 0 or done == len(jobs):
                    rate = done / max(time.time() - t_start, 1e-6)
                    eta = (len(jobs) - done) / max(rate, 1e-6) / 60
                    print(f'[{done}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                          f'rate={rate:.1f}/s eta={eta:.1f}min')

    # 写 decoys.csv
    rows.sort(key=lambda x: (x['stem'], x['decoy_id']))
    csv_path = os.path.join(args.out_dir, 'decoys.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'stem', 'decoy_id', 'dockq', 'fnat', 'irms', 'lrms', 'classification'])
        for r in rows:
            w.writerow([r['name'], r['stem'], r['decoy_id'],
                        f"{r['dockq']:.4f}", f"{r['fnat']:.4f}",
                        f"{r['irms']:.3f}", f"{r['lrms']:.3f}",
                        r['classification']])

    total = (time.time() - t_start) / 60
    print('---')
    print(f'成功: {n_ok}   失败: {n_fail}   总耗时: {total:.1f}min')
    print(f'csv: {csv_path}')

    if fail_log:
        log_path = os.path.join(args.out_dir, 'failures.log')
        with open(log_path, 'w') as f:
            for n, m in fail_log:
                f.write(f'{n}\t{m}\n')
        print(f'失败日志: {log_path}')


if __name__ == '__main__':
    main()
