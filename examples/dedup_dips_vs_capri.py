"""DIPS 去 CAPRI 同源：用 MMseqs2 把 CAPRI 参考序列比对到 DIPS 复合物序列，
identity ≥ 阈值的 DIPS 复合物写入 exclude 列表，供 prep_dips.py --exclude_file 使用。

前置：
  1. 已跑过 prep_dips.py 的下载+拆链阶段（split_pdbs/<stem>_receptor.pdb / _ligand.pdb 就位）
     —— 即使表面没生成也行，本脚本只读拆链 PDB 提序列。
  2. 服务器已装 mmseqs2：  conda install -c bioconda mmseqs2   或   apt 自带二进制
  3. capri_ref.fasta 已由 extract_capri_sequences.py 生成。

流程：
  A. 遍历 split_dir 下所有 <stem>_receptor.pdb / _ligand.pdb，提序列 -> dips_seqs.fasta
     （header = <stem>__R / <stem>__L，便于回溯到复合物 stem）
  B. mmseqs easy-search  capri_ref.fasta  dips_seqs.fasta  ->  命中表
  C. 命中(identity≥id_thresh 且 coverage≥cov_thresh)的 stem 收集成 set
  D. 把这些 stem 对应的 PDB 4字符前缀 + stem 本身都写进 exclude
     —— prep_dips.py 的 read_exclude 按 4 字符前缀匹配，这里额外输出 stem 级精确列表。

用法:
  python examples/dedup_dips_vs_capri.py \\
      --capri_fasta data/capri_exclude/capri_ref.fasta \\
      --split_dir   /root/autodl-tmp/dips/split_pdbs \\
      --work_dir    /root/autodl-tmp/dips/dedup \\
      --out_exclude data/dips/exclude_capri.txt \\
      --id_thresh 0.30 --cov_thresh 0.50 --threads 20
"""
import os, sys, csv, argparse, subprocess, shutil

THREE2ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'U','PYL':'O','HSD':'H','HSE':'H','HSP':'H',
}

def pdb_to_seq(path):
    """单链/多链 PDB -> 拼接序列（按出现顺序，残基号去重）。"""
    seen, seq = set(), []
    with open(path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                one = THREE2ONE.get(line[17:20].strip())
                if one is None:
                    continue
                key = (line[21], line[22:26].strip(), line[26].strip())
                if key in seen:
                    continue
                seen.add(key)
                seq.append(one)
    return ''.join(seq)


def build_dips_fasta(split_dir, out_fasta, min_len=20):
    """遍历 split_dir 的 *_receptor.pdb/_ligand.pdb，写 DIPS 序列 FASTA。
    header: <stem>__R / <stem>__L，stem = 去掉 _receptor/_ligand 后缀。
    返回写出的序列条数。"""
    n = 0
    with open(out_fasta, 'w') as out:
        for fn in sorted(os.listdir(split_dir)):
            if fn.endswith('_receptor.pdb'):
                stem, tag = fn[:-len('_receptor.pdb')], 'R'
            elif fn.endswith('_ligand.pdb'):
                stem, tag = fn[:-len('_ligand.pdb')], 'L'
            else:
                continue
            seq = pdb_to_seq(os.path.join(split_dir, fn))
            if len(seq) < min_len:
                continue
            out.write(f'>{stem}__{tag}\n')
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + '\n')
            n += 1
    return n


def run_mmseqs(capri_fasta, dips_fasta, work_dir, id_thresh, cov_thresh, threads):
    """mmseqs easy-search: query=CAPRI, target=DIPS。返回命中 tsv 路径。"""
    os.makedirs(work_dir, exist_ok=True)
    res_tsv = os.path.join(work_dir, 'capri_vs_dips.tsv')
    tmp = os.path.join(work_dir, 'mmseqs_tmp')
    mmseqs = shutil.which('mmseqs')
    if not mmseqs:
        sys.exit('✗ 未找到 mmseqs，请先 conda install -c bioconda mmseqs2')
    # --min-seq-id 在 search 阶段预过滤；-c/--cov-mode 控制覆盖度
    cmd = [
        mmseqs, 'easy-search', capri_fasta, dips_fasta, res_tsv, tmp,
        '--min-seq-id', str(id_thresh),
        '-c', str(cov_thresh), '--cov-mode', '0',
        '-s', '7.5',                      # 高灵敏度，宁可多召回
        '--threads', str(threads),
        '--format-output', 'query,target,fident,alnlen,qcov,tcov,evalue',
    ]
    print('运行:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return res_tsv


def collect_hits(res_tsv, id_thresh, cov_thresh):
    """从 mmseqs 结果收集命中的 DIPS stem。target header = <stem>__R/L。"""
    stems = set()
    with open(res_tsv) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 7:
                continue
            target, fident, qcov, tcov = p[1], float(p[2]), float(p[4]), float(p[5])
            if fident >= id_thresh and max(qcov, tcov) >= cov_thresh:
                stem = target.rsplit('__', 1)[0]
                stems.add(stem)
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capri_fasta', required=True)
    ap.add_argument('--split_dir', required=True, help='DIPS 拆链 PDB 目录')
    ap.add_argument('--work_dir', required=True, help='中间文件目录')
    ap.add_argument('--out_exclude', required=True, help='输出 exclude 列表(给 prep_dips.py)')
    ap.add_argument('--out_exclude_stems', default=None,
                    help='额外输出 stem 级精确命中列表(可选)')
    ap.add_argument('--id_thresh', type=float, default=0.30,
                    help='序列 identity 阈值，≥ 则判同源排除（发表常用 0.30）')
    ap.add_argument('--cov_thresh', type=float, default=0.50)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--reuse_fasta', action='store_true',
                    help='复用已存在的 dips_seqs.fasta，跳过重新提序列')
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    dips_fasta = os.path.join(args.work_dir, 'dips_seqs.fasta')

    if args.reuse_fasta and os.path.exists(dips_fasta):
        print(f'复用已有 {dips_fasta}')
    else:
        print('── A. 从 DIPS 拆链 PDB 提序列 ──')
        n = build_dips_fasta(args.split_dir, dips_fasta)
        print(f'  写出 {n} 条 DIPS 链序列 -> {dips_fasta}')

    print('── B. MMseqs2 比对 CAPRI vs DIPS ──')
    res_tsv = run_mmseqs(args.capri_fasta, dips_fasta, args.work_dir,
                         args.id_thresh, args.cov_thresh, args.threads)

    print('── C. 收集命中 ──')
    stems = collect_hits(res_tsv, args.id_thresh, args.cov_thresh)
    print(f'  命中（同源需排除）的 DIPS 复合物: {len(stems)} 个')

    # exclude 列表：prep_dips.py read_exclude 按 4 字符前缀匹配。
    # stem 形如 <pdbid>_<rec>_<lig>，取前 4 字符即 PDB ID 前缀。
    prefixes = sorted({s[:4] for s in stems})
    with open(args.out_exclude, 'w') as f:
        f.write('# 由 dedup_dips_vs_capri.py 生成：与 CAPRI Score_set 序列同源的 DIPS PDB 前缀\n')
        f.write(f'# id_thresh={args.id_thresh} cov_thresh={args.cov_thresh} '
                f'命中复合物={len(stems)} 唯一前缀={len(prefixes)}\n')
        for p in prefixes:
            f.write(p + '\n')
    print(f'── D. exclude 列表 -> {args.out_exclude}（{len(prefixes)} 个 PDB 前缀）')

    if args.out_exclude_stems:
        with open(args.out_exclude_stems, 'w') as f:
            for s in sorted(stems):
                f.write(s + '\n')
        print(f'  stem 级精确列表 -> {args.out_exclude_stems}')


if __name__ == '__main__':
    main()
