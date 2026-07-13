"""从 CAPRI Score_set tar(.bz2) 提取每个 target 的参考序列。
同一 target 的所有 decoy 是同两条蛋白的不同对接位姿，序列相同，取 MODEL 1 即可。
输出 FASTA：每条链一个 entry >T<num>_<group>_<chain>，用作 DIPS 去同源的 query。
本地无 biopython，用内置最小 PDB 解析。
"""
import sys, os, bz2, tarfile, io

THREE2ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','SEC':'U','PYL':'O','HSD':'H','HSE':'H','HSP':'H',
}

def model1_sequences(pdb_text):
    """解析 PDB 文本第一个 MODEL，返回 {chain: seq}。按残基序号去重。"""
    seqs = {}        # chain -> list of (resseq, icode, oneletter)
    seen = {}        # chain -> set of (resseq,icode)
    in_model = False
    started = False
    for line in pdb_text.splitlines():
        if line.startswith('MODEL'):
            if started:      # 已处理完第一个 model
                break
            in_model = True
            started = True
            continue
        if line.startswith('ENDMDL'):
            if in_model:
                break
        # 没有 MODEL 关键字的单模型 PDB 也支持
        if line.startswith(('ATOM', 'HETATM')):
            resn = line[17:20].strip()
            one = THREE2ONE.get(resn)
            if one is None:
                continue
            chain = line[21].strip() or '_'
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (resseq, icode)
            s = seen.setdefault(chain, set())
            if key in s:
                continue
            s.add(key)
            seqs.setdefault(chain, []).append(one)
    return {c: ''.join(v) for c, v in seqs.items()}

def main():
    tar_path = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\57102\Desktop\Scoreset_v2022_Scorers.tar.bz2'
    out_fasta = sys.argv[2] if len(sys.argv) > 2 else \
        r'C:\Users\57102\Desktop\TraDock_extracted\TraDock\data\capri_exclude\capri_ref.fasta'
    os.makedirs(os.path.dirname(out_fasta), exist_ok=True)

    entries = []
    n_files = 0
    with tarfile.open(tar_path, 'r:bz2') as tf:
        members = [m for m in tf.getmembers() if m.name.endswith('.pdb')]
        members.sort(key=lambda m: m.name)
        for m in members:
            n_files += 1
            base = os.path.basename(m.name)[:-4]   # S-T029.1
            tag = base.replace('S-', '').replace('.', '_')  # T029_1
            f = tf.extractfile(m)
            text = f.read().decode('utf-8', errors='replace')
            seqs = model1_sequences(text)
            for chain, seq in seqs.items():
                if len(seq) >= 20:   # 过滤太短的（多肽/离子）
                    entries.append((f'{tag}_{chain}', seq))

    with open(out_fasta, 'w') as out:
        for name, seq in entries:
            out.write(f'>{name}\n')
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + '\n')

    print(f'解析 {n_files} 个 decoy PDB')
    print(f'提取 {len(entries)} 条参考链序列 -> {out_fasta}')
    lens = [len(s) for _, s in entries]
    if lens:
        print(f'链长度: min={min(lens)} max={max(lens)} 平均={sum(lens)//len(lens)}')

if __name__ == '__main__':
    main()
