"""
LightDock decoys → TransformerDock 训练数据

输入目录结构（LightDock 标准输出）：
  <ld_root>/<stem>/
    native.pdb                    # 你提供的 native 复合物（用于算 DockQ）
    rec_chains.txt                # 单行：受体链 ID，例 'A' 或 'AB'
    lig_chains.txt                # 单行：配体链 ID
    swarm_0/
      lightdock_0.pdb
      lightdock_1.pdb
      ...
    swarm_1/
      ...

输出：
  <out_dir>/
    {stem}_d{global_id}_receptor.ply
    {stem}_d{global_id}_ligand.ply
    decoys.csv    # name,stem,decoy_id,dockq,fnat,irms,lrms,classification

依赖：
  - DockQ 官方脚本可调用：环境变量 DOCKQ_BIN 或默认 'DockQ.py'
  - examples/surface_gen.py 在同目录
"""
import os
import sys
import csv
import glob
import argparse
import subprocess
import tempfile
import time
import re
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.surface_gen import pdb_to_surface_ply
from examples.prep_dockground_decoys import split_chains, calc_dockq, classify


DOCKQ_BIN = os.environ.get('DOCKQ_BIN', 'DockQ')


def run_dockq(decoy_pdb, native_pdb,
              native_chains=None, model_chains=None):
    """调 DockQ 官方脚本，解析出 (fnat, irms, lrms, dockq)。失败返回 None。
    native_chains / model_chains: 可选 list，如 ['A','B']，传给 DockQ 做链映射。
    兼容 DockQ v1（旧 CLI 'DockQ.py'）和 v2（新 CLI 'DockQ'，输出带冒号）。"""
    cmd = [DOCKQ_BIN, decoy_pdb, native_pdb]
    if native_chains and model_chains:
        # DockQ v2 用 --mapping model_chain:native_chain
        # DockQ v1 用 -native_chain1 / -model_chain1
        # 优先尝试 v2 语法（更明确）
        mapping = ''.join(model_chains) + ':' + ''.join(native_chains)
        cmd += ['--mapping', mapping]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    def grab(*pats):
        for pat in pats:
            m = re.search(pat, out, re.MULTILINE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    # v2 格式: "DockQ: 0.014" / "iRMSD: 22.561" / "LRMSD: 43.302" / "fnat: 0.000"
    # v1 格式: "DockQ 0.014" / "iRMS 22.561" / "Lrms 43.302" / "Fnat 0.000"
    # 用 [:\s]+ 同时匹配冒号和空格
    fnat  = grab(r'(?<!non)[Ff]nat[:\s]+([\d.]+)')
    irms  = grab(r'\bi[Rr][Mm][Ss][Dd]?[:\s]+([\d.]+)')
    lrms  = grab(r'\b[Ll][Rr][Mm][Ss][Dd]?[:\s]+([\d.]+)')
    dockq = grab(r'DockQ(?:\s+Score)?[:\s]+([\d.]+)')
    if None in (fnat, irms, lrms):
        return None
    if dockq is None:
        dockq = calc_dockq(fnat, irms, lrms)
    return fnat, irms, lrms, dockq


def read_chain_list_file(path):
    """读链映射文件。接受 'A B' / 'A\\nB' / 'AB'（单 token 多字符自动拆）。返回 list 保留顺序。"""
    with open(path) as f:
        content = f.read().strip()
    parts = content.split()
    if len(parts) == 1 and len(parts[0]) > 1:
        return list(parts[0])   # 'AB' → ['A','B']
    return parts


def detect_chains_in_pdb(pdb_path):
    """扫描 PDB 文件，返回按出现顺序去重的链 ID 列表（只看 ATOM）。"""
    seen = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and len(line) > 21:
                c = line[21]
                if c not in seen:
                    seen.append(c)
    return seen


def auto_split_decoy_chains(decoy_pdb, n_rec_chains, n_lig_chains):
    """
    LightDock 输出的 decoy PDB 链 ID 与 native 不同（通常 receptor 链按
    原顺序、ligand 链紧随其后，但 ID 可能被 LightDock 重命名）。
    按"前 n_rec_chains 个出现的链当 receptor，后 n_lig_chains 个当 ligand"
    的策略自动拆。
    返回 (rec_chains_list, lig_chains_list) 或 None（链数不够）。
    返回 list 而非 set，因为后续需要顺序对应 native_chains 做 DockQ 链映射。
    """
    chains = detect_chains_in_pdb(decoy_pdb)
    if len(chains) < n_rec_chains + n_lig_chains:
        return None
    rec = chains[:n_rec_chains]
    lig = chains[n_rec_chains:n_rec_chains + n_lig_chains]
    return rec, lig


def process_decoy(job):
    (stem, gid, decoy_pdb, native_pdb,
     native_rec_chains, native_lig_chains,
     native_chains_map, model_chains_map,
     out_dir, voxel_size) = job

    name = f'{stem}_d{gid}'
    rec_ply = os.path.join(out_dir, f'{name}_receptor.ply')
    lig_ply = os.path.join(out_dir, f'{name}_ligand.ply')

    # 自动探测 decoy 实际链；按"前 N rec 链 + 后 M lig 链"对应回 native
    decoy_split = auto_split_decoy_chains(
        decoy_pdb, len(native_rec_chains), len(native_lig_chains))
    if decoy_split is None:
        return {'ok': False, 'name': name,
                'msg': f'decoy 链数不足，期望 {len(native_rec_chains)}+'
                       f'{len(native_lig_chains)}'}
    decoy_rec_chains, decoy_lig_chains = decoy_split  # list 保序

    # 如未显式提供链映射，按 native ↔ decoy 的链顺序对应构造
    if native_chains_map is None or model_chains_map is None:
        native_chains_map = list(native_rec_chains) + list(native_lig_chains)
        model_chains_map  = list(decoy_rec_chains)  + list(decoy_lig_chains)

    if os.path.exists(rec_ply) and os.path.exists(lig_ply):
        labels = run_dockq(decoy_pdb, native_pdb,
                           native_chains_map, model_chains_map)
        if labels is None:
            return {'ok': False, 'name': name, 'msg': 'DockQ 失败(cached ply)'}
        fnat, irms, lrms, dockq = labels
        return {'ok': True, 'name': name, 'stem': stem, 'decoy_id': gid,
                'dockq': dockq, 'fnat': fnat, 'irms': irms, 'lrms': lrms,
                'classification': classify(dockq)}

    labels = run_dockq(decoy_pdb, native_pdb,
                       native_chains_map, model_chains_map)
    if labels is None:
        return {'ok': False, 'name': name, 'msg': 'DockQ 失败'}
    fnat, irms, lrms, dockq = labels

    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec_pdb = os.path.join(tmp, 'rec.pdb')
            lig_pdb = os.path.join(tmp, 'lig.pdb')
            # split_chains 接受 list/str/set，这里传 list 即可
            split_chains(decoy_pdb, rec_pdb, lig_pdb,
                         set(decoy_rec_chains), set(decoy_lig_chains))

            ok_r = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
            ok_l = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)
            if not (ok_r and ok_l):
                for f in (rec_ply, lig_ply):
                    if os.path.exists(f):
                        os.remove(f)
                return {'ok': False, 'name': name,
                        'msg': f'ply 失败 rec={ok_r} lig={ok_l}'}
    except Exception as e:
        for f in (rec_ply, lig_ply):
            if os.path.exists(f):
                os.remove(f)
        return {'ok': False, 'name': name, 'msg': f'异常: {e}'}

    return {'ok': True, 'name': name, 'stem': stem, 'decoy_id': gid,
            'dockq': dockq, 'fnat': fnat, 'irms': irms, 'lrms': lrms,
            'classification': classify(dockq)}


def _iter_stem_dirs(ld_root, limit=None):
    stem_dirs = sorted(d for d in glob.glob(os.path.join(ld_root, '*'))
                       if os.path.isdir(d))
    if limit:
        stem_dirs = stem_dirs[:limit]
    return stem_dirs


def iter_jobs(ld_root, out_dir, voxel_size, max_per_target=None, limit=None):
    """生成器版 collect_jobs：streaming yield job，避免 50 万 tuple 同时驻留内存。
    顺序：按 stem_dir 排序，再按 decoy 文件排序。"""
    for stem_dir in _iter_stem_dirs(ld_root, limit):
        stem = os.path.basename(stem_dir)
        native = os.path.join(stem_dir, 'native.pdb')
        rec_f = os.path.join(stem_dir, 'rec_chains.txt')
        lig_f = os.path.join(stem_dir, 'lig_chains.txt')
        if not all(os.path.exists(p) for p in (native, rec_f, lig_f)):
            print(f'  [跳过] {stem}: 缺 native.pdb / rec_chains.txt / lig_chains.txt')
            continue
        rec_chains = read_chain_list_file(rec_f)
        lig_chains = read_chain_list_file(lig_f)

        nc_f = os.path.join(stem_dir, 'native_chains.txt')
        mc_f = os.path.join(stem_dir, 'model_chains.txt')
        native_map = read_chain_list_file(nc_f) if os.path.exists(nc_f) else None
        model_map  = read_chain_list_file(mc_f) if os.path.exists(mc_f) else None

        decoys = sorted(glob.glob(os.path.join(stem_dir, 'swarm_*', 'lightdock_*.pdb')))
        if max_per_target:
            decoys = decoys[:max_per_target]

        for gid, pdb in enumerate(decoys):
            yield (stem, gid, pdb, native,
                   rec_chains, lig_chains,
                   native_map, model_map,
                   out_dir, voxel_size)


def count_jobs(ld_root, max_per_target=None, limit=None):
    """轻量计数（不构造完整 tuple），用于进度报告。"""
    total = 0
    for stem_dir in _iter_stem_dirs(ld_root, limit):
        rec_f = os.path.join(stem_dir, 'rec_chains.txt')
        lig_f = os.path.join(stem_dir, 'lig_chains.txt')
        native = os.path.join(stem_dir, 'native.pdb')
        if not all(os.path.exists(p) for p in (native, rec_f, lig_f)):
            continue
        n = len(glob.glob(os.path.join(stem_dir, 'swarm_*', 'lightdock_*.pdb')))
        if max_per_target:
            n = min(n, max_per_target)
        total += n
    return total


def _wait_any(futures):
    """等任意 future 完成，返回 (completed_set, pending_set)。"""
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return done, not_done


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ld_root', required=True,
                   help='LightDock 输出根目录（含 <stem>/swarm_*/lightdock_*.pdb）')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--voxel_size', type=float, default=3.5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max_per_target', type=int, default=None)
    p.add_argument('--limit', type=int, default=None,
                   help='只处理前 N 个 target（按目录名排序，调试用）')
    p.add_argument('--report_every', type=int, default=200)
    p.add_argument('--max_in_flight', type=int, default=None,
                   help='in-flight job 上限，默认 workers*8。N>1000 时调小可省内存')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.workers < 1:
        sys.exit('--workers 必须 ≥ 1')
    if args.max_in_flight is not None and args.max_in_flight < 1:
        sys.exit('--max_in_flight 必须 ≥ 1')
    max_in_flight = args.max_in_flight or max(args.workers * 8, 32)
    print(f'LightDock 根目录: {args.ld_root}')
    print(f'输出目录:         {args.out_dir}')
    print(f'DockQ 脚本:       {DOCKQ_BIN}')
    print(f'workers:          {args.workers}  (in_flight 上限 {max_in_flight})')
    print('---')

    total_est = count_jobs(args.ld_root, args.max_per_target, args.limit)
    print(f'预估 decoy 任务数: {total_est}')
    if total_est == 0:
        sys.exit(1)

    job_iter = iter_jobs(args.ld_root, args.out_dir, args.voxel_size,
                         args.max_per_target, args.limit)

    rows, fails = [], []
    n_ok = n_fail = 0
    t0 = time.time()
    done = 0

    def handle_result(r):
        nonlocal n_ok, n_fail, done
        (rows if r['ok'] else fails).append(r)
        if r['ok']: n_ok += 1
        else: n_fail += 1
        done += 1
        if done % args.report_every == 0 or done == total_est:
            rate = done / max(time.time()-t0, 1e-6)
            print(f'[{done}/{total_est}] ok={n_ok} fail={n_fail} {rate:.1f}/s')

    if args.workers == 1:
        for j in job_iter:
            handle_result(process_decoy(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            in_flight = set()
            pool_broken = False
            # 先填满 in-flight
            for _ in range(max_in_flight):
                try:
                    j = next(job_iter)
                except StopIteration:
                    break
                in_flight.add(exe.submit(process_decoy, j))
            # 滑动窗口：每完成一个就 submit 下一个
            while in_flight and not pool_broken:
                try:
                    completed, in_flight = _wait_any(in_flight)
                except BrokenProcessPool as e:
                    print(f'[严重] worker pool 崩溃: {e}; 已完成 {done}/{total_est}')
                    pool_broken = True
                    break
                for fut in completed:
                    try:
                        r = fut.result()
                    except BrokenProcessPool as e:
                        print(f'[严重] worker pool 崩溃: {e}')
                        pool_broken = True
                        break
                    except Exception as e:
                        r = {'ok': False, 'name': '?', 'msg': f'worker 异常: {e}'}
                    handle_result(r)
                    if pool_broken:
                        break
                    try:
                        j = next(job_iter)
                        in_flight.add(exe.submit(process_decoy, j))
                    except StopIteration:
                        pass
                    except BrokenProcessPool as e:
                        print(f'[严重] worker pool 崩溃（submit 时）: {e}')
                        pool_broken = True
                        break

    rows.sort(key=lambda x: (x['stem'], x['decoy_id']))
    csv_path = os.path.join(args.out_dir, 'decoys.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name','stem','decoy_id','dockq','fnat','irms','lrms','classification'])
        for r in rows:
            w.writerow([r['name'], r['stem'], r['decoy_id'],
                        f"{r['dockq']:.4f}", f"{r['fnat']:.4f}",
                        f"{r['irms']:.3f}", f"{r['lrms']:.3f}",
                        r['classification']])
    print(f'---\nok={n_ok} fail={n_fail}  耗时={(time.time()-t0)/60:.1f}min')
    print(f'csv: {csv_path}')

    if fails:
        log = os.path.join(args.out_dir, 'failures.log')
        with open(log, 'w') as f:
            for r in fails:
                f.write(f"{r.get('name','?')}\t{r.get('msg','')}\n")
        print(f'失败日志: {log}')


if __name__ == '__main__':
    main()
