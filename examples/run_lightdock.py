"""
用 LightDock 批量为 PDB 复合物生成 decoy。

输入目录（任选其一）：
  方式 A：每个 target 一个 native 复合物 PDB
    <pdb_dir>/<stem>.pdb
    + <pdb_dir>/<stem>.chains    # 单行：'A B' (受体链 配体链)

  方式 B：受体/配体已分开
    <pdb_dir>/<stem>_receptor.pdb
    <pdb_dir>/<stem>_ligand.pdb

输出（适配 prep_lightdock_decoys.py）：
  <out_root>/<stem>/
    native.pdb
    rec_chains.txt
    lig_chains.txt
    swarm_*/lightdock_*.pdb

依赖：
  pip install lightdock
  确保 PATH 里有：
    lightdock3_setup.py
    lightdock3.py
    lgd_generate_conformations.py
    lgd_cluster_bsas.py

"""
import os
import re
import sys
import glob
import shutil
import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run(cmd, cwd=None, timeout=3600):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f'cmd failed: {" ".join(cmd)}\nstderr:\n{r.stderr[-1000:]}')
    return r.stdout


def _sim_timeout(explicit=None):
    """LightDock simulation timeout in seconds (default 12h)."""
    if explicit is not None:
        return int(explicit)
    return int(os.environ.get('LIGHTDOCK_TIMEOUT', '43200'))


def split_complex(complex_pdb, rec_chains, lig_chains, rec_out, lig_out):
    """按链拆 native 复合物为 receptor.pdb / ligand.pdb。"""
    rec_set = set(rec_chains)
    lig_set = set(lig_chains)
    rec_lines, lig_lines = [], []
    with open(complex_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM', 'TER')) and len(line) > 21:
                c = line[21]
                if c in rec_set:
                    rec_lines.append(line)
                elif c in lig_set:
                    lig_lines.append(line)
    with open(rec_out, 'w') as f:
        f.writelines(rec_lines)
        f.write('END\n')
    with open(lig_out, 'w') as f:
        f.writelines(lig_lines)
        f.write('END\n')


# LightDock 支持的元素（标准蛋白质 + 常见金属）
_LIGHTDOCK_ELEMENTS = {'C', 'N', 'O', 'S', 'P', 'H', 'F', 'CL', 'BR',
                       'FE', 'ZN', 'MG', 'CA', 'NA', 'K', 'MN', 'CU', 'NI', 'CO'}

def _clean_pdb_for_lightdock(in_pdb, out_pdb):
    """为 LightDock 清理 PDB：
    - 只保留 ATOM 行（去掉 HETATM 配体/离子，避免奇怪元素如 I/SE 等）
    - 只保留标准氨基酸残基
    - 去掉氢原子
    """
    standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                   'THR', 'TRP', 'TYR', 'VAL', 'MSE'}  # MSE 是硒代蛋氨酸，会被识别为 SE
    out_lines = []
    with open(in_pdb, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('ATOM') and len(line) >= 78:
                resname = line[17:20].strip()
                element = line[76:78].strip().upper()
                # 跳过 H 原子和 MSE（含 SE）
                if element == 'H' or resname == 'MSE':
                    continue
                # 跳过非标准元素
                if element and element not in _LIGHTDOCK_ELEMENTS:
                    continue
                # 只保留标准氨基酸
                if resname not in standard_aa:
                    continue
                out_lines.append(line)
            elif line.startswith(('TER', 'END')):
                out_lines.append(line)
    with open(out_pdb, 'w') as f:
        f.writelines(out_lines)
        if not out_lines or not out_lines[-1].startswith('END'):
            f.write('END\n')


def process_one(args_tuple):
    (stem, rec_pdb, lig_pdb,
     rec_chains, lig_chains, out_root,
     swarms, glowworms, steps, sim_timeout) = args_tuple

    out_dir = os.path.join(out_root, stem)
    t0 = time.time()

    # Resume：已有足够 decoy 则跳过（允许先前 40×200=8000 的结果在改成 20×50 后仍复用）
    if os.path.isdir(out_dir):
        has_native = os.path.exists(os.path.join(out_dir, 'native.pdb'))
        existing = glob.glob(os.path.join(out_dir, 'swarm_*', 'lightdock_*.pdb'))
        expected_min = int(swarms * glowworms * 0.9)
        if has_native and len(existing) >= expected_min:
            return stem, True, f'cached {len(existing)} decoys', 0.0
        # 半完成或残留：lightdock3_setup.py 不允许 swarm_X 已存在，必须清掉
        shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)

    # 1. 在 out_dir 合并 native 复合物（供 Step 6 DockQ 使用）+ 写链文件
    native_path = os.path.join(out_dir, 'native.pdb')
    with open(native_path, 'w') as out:
        for p in (rec_pdb, lig_pdb):
            with open(p) as f:
                for line in f:
                    if not line.startswith('END'):
                        out.write(line)
        out.write('END\n')
    with open(os.path.join(out_dir, 'rec_chains.txt'), 'w') as f:
        f.write(rec_chains)
    with open(os.path.join(out_dir, 'lig_chains.txt'), 'w') as f:
        f.write(lig_chains)

    # 2. 复制受体/配体（LightDock 就地写中间文件，拷贝到 out_dir）
    # 同时过滤掉 LightDock 不支持的元素（HETATM、非标准元素）
    rec_local = os.path.join(out_dir, 'receptor.pdb')
    lig_local = os.path.join(out_dir, 'ligand.pdb')
    _clean_pdb_for_lightdock(rec_pdb, rec_local)
    _clean_pdb_for_lightdock(lig_pdb, lig_local)

    try:
        # 3. setup
        run(['lightdock3_setup.py', 'receptor.pdb', 'ligand.pdb',
             '-s', str(swarms), '-g', str(glowworms),
             '--noxt', '--noh', '--now'],
            cwd=out_dir)

        # 4. simulation
        run(['lightdock3.py', 'setup.json', str(steps), '-c', '1'],
            cwd=out_dir, timeout=sim_timeout)

        # 5. generate conformations（每个 swarm 各自生成 PDB）
        # LightDock 0.9.4: lgd_generate_conformations.py 输入是原始 PDB 名
        # （内部会自动找对应的 lightdock_X.pdb）
        # 旧版需要直接传 receptor_lightdock.pdb
        new_format = os.path.exists(os.path.join(out_dir, 'lightdock_receptor.pdb'))

        for swarm_dir in sorted(glob.glob(os.path.join(out_dir, 'swarm_*'))):
            gso_files = glob.glob(os.path.join(swarm_dir, 'gso_*.out'))
            if not gso_files:
                continue
            # 按数值排序，避免 'gso_100' < 'gso_80' 这种字符串排序坑
            gso_files.sort(key=lambda p: int(re.search(r'gso_(\d+)', p).group(1)))
            last = gso_files[-1]
            if new_format:
                # 0.9.4：传原始 PDB 名
                rec_arg = '../receptor.pdb'
                lig_arg = '../ligand.pdb'
            else:
                rec_arg = '../receptor_lightdock.pdb'
                lig_arg = '../ligand_lightdock.pdb'
            run(['lgd_generate_conformations.py',
                 rec_arg, lig_arg,
                 os.path.basename(last), str(glowworms)],
                cwd=swarm_dir)

    except Exception as e:
        return stem, False, f'lightdock 失败: {e}', time.time() - t0

    # 6. 清理大文件（保留 native + 链文件 + lightdock_*.pdb）
    keep = {'native.pdb', 'rec_chains.txt', 'lig_chains.txt',
            'lightdock_receptor.pdb', 'lightdock_ligand.pdb',
            'receptor_lightdock.pdb', 'ligand_lightdock.pdb'}
    for fname in os.listdir(out_dir):
        full = os.path.join(out_dir, fname)
        if os.path.isfile(full) and fname not in keep:
            os.remove(full)

    n_pdb = len(glob.glob(os.path.join(out_dir, 'swarm_*', 'lightdock_*.pdb')))
    return stem, True, f'生成 {n_pdb} 个 decoy', time.time() - t0


def read_chains_file(path):
    """读 .chains 文件。期望两个 token (rec_chains lig_chains)，忽略 # 注释和空行。
    返回 (rec_str, lig_str)。格式错误抛 ValueError。"""
    tokens = []
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            tokens.extend(line.split())
            if len(tokens) >= 2:
                break
    if len(tokens) < 2:
        raise ValueError(f'{path}: 期望两个 token (rec_chains lig_chains)，'
                         f'实际只有 {len(tokens)} 个')
    return tokens[0], tokens[1]


def collect_targets(pdb_dir):
    """扫描 pdb_dir，返回 [(stem, rec_pdb, lig_pdb, rec_chains, lig_chains)]。
    支持两种命名（方式 A 和 B）。
    native 复合物（供 DockQ 使用）在 process_one 里动态生成，不污染输入目录。"""
    targets = []

    # 方式 B：已经分开
    for rec in sorted(glob.glob(os.path.join(pdb_dir, '*_receptor.pdb'))):
        stem = os.path.basename(rec)[:-len('_receptor.pdb')]
        lig = os.path.join(pdb_dir, f'{stem}_ligand.pdb')
        if not os.path.exists(lig):
            continue
        # 链信息：stem 名编码（如 1avw_A_B）或 .chains 文件
        chains_f = os.path.join(pdb_dir, f'{stem}.chains')
        if os.path.exists(chains_f):
            try:
                rc, lc = read_chains_file(chains_f)
            except ValueError as e:
                print(f'  [跳过] {stem}: {e}')
                continue
        else:
            parts = stem.split('_')
            if len(parts) >= 3:
                rc, lc = parts[1], parts[2]
            else:
                print(f'  [跳过] {stem}: 无法推断链')
                continue
        targets.append((stem, rec, lig, rc, lc))

    # 方式 A：单文件 + .chains（需要现场拆）
    for cpx in sorted(glob.glob(os.path.join(pdb_dir, '*.pdb'))):
        stem = os.path.basename(cpx)[:-4]
        if (stem.endswith('_receptor') or stem.endswith('_ligand')
                or stem.endswith('_complex')):
            continue
        chains_f = os.path.join(pdb_dir, f'{stem}.chains')
        if not os.path.exists(chains_f):
            continue
        if any(t[0] == stem for t in targets):
            continue
        try:
            rc, lc = read_chains_file(chains_f)
        except ValueError as e:
            print(f'  [跳过] {stem}: {e}')
            continue
        rec = os.path.join(pdb_dir, f'{stem}_receptor.pdb')
        lig = os.path.join(pdb_dir, f'{stem}_ligand.pdb')
        if not (os.path.exists(rec) and os.path.exists(lig)):
            split_complex(cpx, rc, lc, rec, lig)
        targets.append((stem, rec, lig, rc, lc))

    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb_dir', required=True,
                   help='输入目录（含 *_receptor.pdb / *_ligand.pdb 或 *.pdb + *.chains）')
    p.add_argument('--out_root', required=True,
                   help='LightDock 输出根目录')
    p.add_argument('--swarms', type=int, default=20,
                   help='swarm 数（覆盖整个表面，默认 20；与 glowworms 乘积≈1000）')
    p.add_argument('--glowworms', type=int, default=50,
                   help='每个 swarm 的 glowworm 数（构象数，默认 50）')
    p.add_argument('--steps', type=int, default=100,
                   help='GSO 优化步数')
    p.add_argument('--workers', type=int, default=4,
                   help='并行 target 数（每个 LightDock 自身单线程）')
    p.add_argument('--limit', type=int, default=None,
                   help='只处理前 N 个 target（调试用）')
    p.add_argument('--targets', type=str, default=None,
                   help='只跑这些 target，逗号分隔，如 1AKJ,1ATN')
    p.add_argument('--timeout', type=int, default=None,
                   help='单靶 lightdock3 超时秒数；默认读 LIGHTDOCK_TIMEOUT 或 43200')
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    targets = collect_targets(args.pdb_dir)
    if args.targets:
        wanted = {t.strip() for t in args.targets.split(',') if t.strip()}
        targets = [t for t in targets if t[0] in wanted]
        missing = sorted(wanted - {t[0] for t in targets})
        if missing:
            print(f'[warn] 输入目录找不到: {", ".join(missing)}')
    if args.limit:
        targets = targets[:args.limit]
    sim_timeout = _sim_timeout(args.timeout)
    print(f'目标数: {len(targets)}')
    print(f'每 target 预期 decoy 数 ≈ {args.swarms * args.glowworms}')
    print(f'lightdock3 timeout: {sim_timeout}s ({sim_timeout/3600:.1f}h)')
    print('---')

    jobs = [(stem, rec, lig, rc, lc, args.out_root,
             args.swarms, args.glowworms, args.steps, sim_timeout)
            for (stem, rec, lig, rc, lc) in targets]

    n_ok = n_fail = 0
    if args.workers == 1:
        for j in jobs:
            stem, ok, msg, dt = process_one(j)
            print(f'[{stem}] {"OK" if ok else "FAIL"}  {msg}  ({dt/60:.1f}min)')
            if ok: n_ok += 1
            else: n_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futs = {exe.submit(process_one, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                stem, ok, msg, dt = fut.result()
                print(f'[{stem}] {"OK" if ok else "FAIL"}  {msg}  ({dt/60:.1f}min)')
                if ok: n_ok += 1
                else: n_fail += 1

    print('---')
    print(f'成功: {n_ok}   失败: {n_fail}')
    print(f'下一步: python examples/prep_lightdock_decoys.py '
          f'--ld_root {args.out_root} --out_dir <surfaces_out>')


if __name__ == '__main__':
    main()
