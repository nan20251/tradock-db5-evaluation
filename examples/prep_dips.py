"""
DIPS 数据集准备：从 metadata CSV 下载 PDB → 按链拆分 → 真实表面 .ply

DIPS (Database of Interacting Protein Structures, Townshend et al. 2019) 是
PPI 对接的大规模训练集（~42K 复合物），从 PDB 自动筛选得到。本脚本不内置
完整列表，而是接受用户提供的 metadata CSV，每行一个复合物。

metadata.csv 格式（必需列：pdb_id, rec_chains, lig_chains）:
    pdb_id,rec_chains,lig_chains
    1abc,A,B
    2xyz,AB,C
    3def,H,L

输出（与 prep_dockground_native.py / prep_db55_surface.py 一致）：
    <out_dir>/
        <pdbid>_<rec>_<lig>_receptor.ply
        <pdbid>_<rec>_<lig>_ligand.ply
        ...
        pairs.csv     (name,label)

去冗余：
    --exclude_file 接收一行一个 PDB ID 的文本，去重时按 4 字符 PDB ID 前缀匹配，
    用于排除 CAPRI 同源数据，避免评估泄漏。

用法:
    # 下载 + 拆链 + 表面（典型用法）
    python examples/prep_dips.py \\
        --metadata data/dips/metadata.csv \\
        --pdb_dir    /root/autodl-tmp/dips/pdbs \\
        --split_dir  /root/autodl-tmp/dips/split_pdbs \\
        --out_dir    /root/autodl-tmp/dips_with_sasa_full \\
        --voxel_size 3.5 \\
        --workers 8 \\
        --exclude_file data/dips/exclude_homologs.txt

    # 调试：只跑前 50 个
    python examples/prep_dips.py ... --limit 50
"""

import os
import sys
import csv
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.surface_gen import pdb_to_surface_ply
from examples.prep_dockground_decoys import split_chains

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. 下载
# ─────────────────────────────────────────────────────────────

def download_pdb(pdb_id, pdb_dir, retries=3, timeout=30):
    """从 RCSB 下载 PDB。已存在且大小正常则跳过。返回 (pdb_id, ok, msg)。"""
    out = os.path.join(pdb_dir, f'{pdb_id.lower()}.pdb')
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return pdb_id, True, 'cached'

    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                with open(out, 'w', encoding='utf-8') as f:
                    f.write(r.text)
                return pdb_id, True, 'ok'
            return pdb_id, False, f'HTTP {r.status_code}'
        except requests.exceptions.Timeout:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                return pdb_id, False, 'timeout'
        except Exception as e:
            return pdb_id, False, str(e)
    return pdb_id, False, 'max retries'


# ─────────────────────────────────────────────────────────────
# 2. metadata 读取 + 去冗余
# ─────────────────────────────────────────────────────────────

def read_metadata(csv_path):
    """读 metadata CSV，返回 [(pdb_id, rec_chains_str, lig_chains_str)]。"""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get('pdb_id') or '').strip().lower()
            rec = (row.get('rec_chains') or '').strip()
            lig = (row.get('lig_chains') or '').strip()
            if not pid or not rec or not lig:
                continue
            rows.append((pid, rec, lig))
    return rows


def read_exclude(path):
    """读 PDB ID 排除列表，返回 lower-case set（按 4 字符前缀）。
    支持：每行一个 ID；# 之后为注释；空行忽略；行内多个 ID 用逗号或空白分隔。"""
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            # 行内允许逗号或空白分隔多个 token
            for tok in line.replace(',', ' ').split():
                if tok:
                    out.add(tok[:4].lower())
    return out


# ─────────────────────────────────────────────────────────────
# 3. 单复合物处理（在子进程跑）
# ─────────────────────────────────────────────────────────────

def process_one(args_tuple):
    """
    输入：(pdb_id, rec_chains, lig_chains, pdb_dir, split_dir, out_dir, voxel_size[, split_only])
    流程：拆链 → 保留拆出的 receptor/ligand PDB（供 LightDock / 去同源用）
          → split_only=True 时到此为止；否则两次 pdb_to_surface_ply → 返回结果。
    """
    # 兼容旧的 7 元组调用
    if len(args_tuple) == 8:
        pdb_id, rec_chains, lig_chains, pdb_dir, split_dir, out_dir, voxel_size, split_only = args_tuple
    else:
        pdb_id, rec_chains, lig_chains, pdb_dir, split_dir, out_dir, voxel_size = args_tuple
        split_only = False

    stem = f'{pdb_id}_{rec_chains}_{lig_chains}'
    rec_ply = os.path.join(out_dir, f'{stem}_receptor.ply')
    lig_ply = os.path.join(out_dir, f'{stem}_ligand.ply')
    rec_pdb = os.path.join(split_dir, f'{stem}_receptor.pdb')
    lig_pdb = os.path.join(split_dir, f'{stem}_ligand.pdb')

    if split_only:
        # 只需拆链 PDB 就位即视为完成（去同源阶段只读这两个文件）
        if os.path.exists(rec_pdb) and os.path.exists(lig_pdb):
            return stem, True, 'cached', 0.0
    else:
        # 4 个文件都在则跳过
        if (os.path.exists(rec_ply) and os.path.exists(lig_ply) and
                os.path.exists(rec_pdb) and os.path.exists(lig_pdb)):
            return stem, True, 'cached', 0.0

    raw_pdb = os.path.join(pdb_dir, f'{pdb_id}.pdb')
    if not os.path.exists(raw_pdb):
        return stem, False, 'PDB 未下载', 0.0

    t0 = time.time()
    try:
        split_chains(raw_pdb, rec_pdb, lig_pdb,
                     set(rec_chains), set(lig_chains))
        if os.path.getsize(rec_pdb) < 100 or os.path.getsize(lig_pdb) < 100:
            for f in (rec_pdb, lig_pdb):
                if os.path.exists(f):
                    os.remove(f)
            return stem, False, '拆链后内容为空（链 ID 错误？）', time.time() - t0

        if split_only:
            return stem, True, 'split', time.time() - t0

        ok_r = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
        ok_l = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)
        if not (ok_r and ok_l):
            for f in (rec_ply, lig_ply):
                if os.path.exists(f):
                    os.remove(f)
            return stem, False, f'ply 失败 rec={ok_r} lig={ok_l}', time.time() - t0
        return stem, True, 'ok', time.time() - t0
    except Exception as e:
        for f in (rec_ply, lig_ply, rec_pdb, lig_pdb):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        return stem, False, f'异常: {e}', time.time() - t0


# ─────────────────────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True,
                   help='CSV: 必需列 pdb_id, rec_chains, lig_chains')
    p.add_argument('--pdb_dir', required=True,
                   help='PDB 下载缓存目录')
    p.add_argument('--split_dir', required=True,
                   help='拆链后的 receptor/ligand PDB 输出目录（供 LightDock 用）')
    p.add_argument('--out_dir', required=True,
                   help='.ply 与 pairs.csv 输出目录')
    p.add_argument('--voxel_size', type=float, default=3.5)
    p.add_argument('--dl_workers', type=int, default=8,
                   help='下载并行线程数')
    p.add_argument('--workers', type=int, default=8,
                   help='表面生成并行进程数')
    p.add_argument('--exclude_file', default=None,
                   help='排除的 PDB ID 列表（一行一个，4 字符匹配）')
    p.add_argument('--limit', type=int, default=None,
                   help='只处理前 N 行（调试用）')
    p.add_argument('--report_every', type=int, default=50)
    p.add_argument('--skip_download', action='store_true',
                   help='跳过下载（PDB 已就位）')
    p.add_argument('--min_free_gb', type=float, default=10.0,
                   help='磁盘剩余空间低于此值(GB)时停止派发新任务，保住已完成成果')
    p.add_argument('--split_only', action='store_true',
                   help='只下载+拆链，不生成表面（用于去同源阶段先拿到序列）')
    args = p.parse_args()

    os.makedirs(args.pdb_dir, exist_ok=True)
    os.makedirs(args.split_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_metadata(args.metadata)
    log.info(f'metadata 共 {len(rows)} 条')

    exclude = read_exclude(args.exclude_file)
    if exclude:
        before = len(rows)
        rows = [r for r in rows if r[0][:4] not in exclude]
        log.info(f'去冗余：排除 {before - len(rows)} 条（exclude 列表 {len(exclude)} 个 PDB）')

    if args.limit:
        rows = rows[:args.limit]
    log.info(f'实际处理 {len(rows)} 条')

    # ── 4.1 下载 ──
    if not args.skip_download:
        log.info('── 下载 PDB ──')
        unique_ids = sorted({r[0] for r in rows})
        n_ok = n_fail = 0
        t_dl = time.time()
        with ThreadPoolExecutor(max_workers=args.dl_workers) as exe:
            futs = {exe.submit(download_pdb, pid, args.pdb_dir): pid
                    for pid in unique_ids}
            for i, fut in enumerate(as_completed(futs), 1):
                pid, ok, msg = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    if n_fail <= 20:
                        log.warning(f'  {pid}: {msg}')
                if i % args.report_every == 0 or i == len(unique_ids):
                    log.info(f'  [{i}/{len(unique_ids)}] ok={n_ok} fail={n_fail} '
                             f'elapsed={(time.time()-t_dl)/60:.1f}min')
        log.info(f'下载完成: {n_ok}/{len(unique_ids)}')

    # ── 4.2 拆链 + 表面生成 ──
    log.info('── 拆链 + 生成表面 ──' if not args.split_only else '── 仅拆链（split_only）──')
    jobs = [(pid, rec, lig, args.pdb_dir, args.split_dir, args.out_dir,
             args.voxel_size, args.split_only)
            for (pid, rec, lig) in rows]

    # 增量写 pairs.csv：每成功一个就 append + flush，prep 中途崩溃也不丢已完成的成果。
    # split_only 模式不写 pairs.csv（它是表面产物清单，拆链阶段还没有表面）。
    pairs_csv = os.path.join(args.out_dir, 'pairs.csv')
    already = set()
    pairs_fh = None
    pairs_w = None
    if not args.split_only:
        if os.path.exists(pairs_csv):
            with open(pairs_csv, newline='') as f:
                for r in csv.reader(f):
                    if r and r[0] != 'name':
                        already.add(r[0])
            log.info(f'续跑：pairs.csv 已有 {len(already)} 条，将跳过重复写入')
        pairs_fh = open(pairs_csv, 'a', newline='')
        pairs_w = csv.writer(pairs_fh)
        if not already:
            pairs_w.writerow(['name', 'label'])
            pairs_fh.flush()

    # 磁盘空间预警：低于阈值就停止派发新任务，保住已完成的表面 + pairs.csv
    import shutil as _shutil
    def _free_gb(path):
        try:
            return _shutil.disk_usage(path).free / (1024 ** 3)
        except OSError:
            return float('inf')
    disk_stop = False

    n_ok = n_fail = 0
    ok_stems = []
    fails = []
    t0 = time.time()

    def _record(stem, ok, msg):
        nonlocal n_ok, n_fail
        if ok:
            n_ok += 1
            ok_stems.append(stem)
            if pairs_w is not None and stem not in already:
                pairs_w.writerow([stem, 1])
                pairs_fh.flush()
                already.add(stem)
        else:
            n_fail += 1
            fails.append((stem, msg))

    if args.workers == 1:
        for i, j in enumerate(jobs, 1):
            if _free_gb(args.out_dir) < args.min_free_gb:
                log.error(f'磁盘剩余 < {args.min_free_gb}GB，停止派发（已完成 {n_ok} 个已写入 pairs.csv）')
                disk_stop = True
                break
            stem, ok, msg, _ = process_one(j)
            _record(stem, ok, msg)
            if i % args.report_every == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-6)
                log.info(f'  [{i}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                         f'rate={rate:.1f}/s free={_free_gb(args.out_dir):.1f}GB')
    else:
        from concurrent.futures import ProcessPoolExecutor as _PPE
        with _PPE(max_workers=args.workers) as exe:
            futs = {}
            job_iter = iter(jobs)
            # 先填满 worker 队列
            for _ in range(min(args.workers * 4, len(jobs))):
                try:
                    j = next(job_iter)
                    futs[exe.submit(process_one, j)] = j
                except StopIteration:
                    break
            done = 0
            while futs:
                for fut in as_completed(list(futs.keys())):
                    j = futs.pop(fut)
                    stem, ok, msg, _ = fut.result()
                    _record(stem, ok, msg)
                    done += 1
                    if done % args.report_every == 0 or done == len(jobs):
                        rate = done / max(time.time() - t0, 1e-6)
                        log.info(f'  [{done}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                                 f'rate={rate:.1f}/s free={_free_gb(args.out_dir):.1f}GB')
                    # 派发新任务（磁盘够才派发）
                    if not disk_stop and _free_gb(args.out_dir) < args.min_free_gb:
                        log.error(f'磁盘剩余 < {args.min_free_gb}GB，停止派发新任务'
                                  f'（已完成 {n_ok} 个均已写入 pairs.csv）')
                        disk_stop = True
                    if not disk_stop:
                        try:
                            nj = next(job_iter)
                            futs[exe.submit(process_one, nj)] = nj
                        except StopIteration:
                            pass
                    break  # 重新 as_completed 剩余 futures

    if pairs_fh is not None:
        pairs_fh.close()

    if fails:
        flog = os.path.join(args.out_dir, 'failures.log')
        with open(flog, 'w') as f:
            for s, m in fails:
                f.write(f'{s}\t{m}\n')
        log.info(f'失败日志: {flog}')

    total = (time.time() - t0) / 60
    log.info('---')
    log.info(f'成功: {n_ok}   失败: {n_fail}   表面生成耗时: {total:.1f}min')
    log.info(f'pairs.csv: {pairs_csv}')
    if disk_stop:
        log.warning('注意：因磁盘空间不足提前停止，未处理完全部复合物。'
                    '扩容后直接重跑本脚本即可续跑（已完成的会跳过）。')


if __name__ == '__main__':
    main()
