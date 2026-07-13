"""
Generate HDOCKlite candidate poses for PPCBench DB5 / DB5-u.

The output layout matches PPCBench result directories consumed by
examples/eval_db5_paper_tradock.py:

  <paper_root>/results/DB5/hdock_1/<PDB>/<PDB>_hdock_1_id.pdb
  <paper_root>/results/DB5/hdock_2/<PDB>/<PDB>_hdock_2_id.pdb
  ...

Default docking command:

  hdock receptor.pdb ligand.pdb -out hdock.out
  createpl_linux hdock.out topN_complex.pdb -nmax N -complex -models

If a local HDOCKlite package uses a different CLI, pass --hdock_template
and/or --createpl_template.
"""
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


MANIFEST_FIELDS = [
    'dataset', 'target_index', 'target_total', 'target', 'status', 'message',
    'elapsed_sec', 'n_requested', 'n_generated', 'receptor', 'ligand',
    'hdock_out', 'models_pdb', 'out_root',
]


def parse_csv(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]


def normalize_chains(value):
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)}


def read_targets(dataset_dir, dataset):
    json_path = dataset_dir / f'{dataset}.json'
    if not json_path.exists():
        raise FileNotFoundError(f'missing dataset json: {json_path}')
    targets = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    return targets


def select_targets(targets, args):
    if args.targets:
        wanted = set(parse_csv(args.targets))
        targets = [t for t in targets if t['pdb'] in wanted]
    if args.start_index > 1:
        targets = targets[args.start_index - 1:]
    if args.limit:
        targets = targets[:args.limit]
    return targets


def input_paths(paper_root, dataset, pdbid):
    struct_dir = paper_root / 'dataset' / dataset / 'structures' / pdbid
    if dataset == 'DB5':
        rec = struct_dir / f'{pdbid}_r_b.pdb'
        lig = struct_dir / f'{pdbid}_l_b.pdb'
    elif dataset == 'DB5-u':
        rec = struct_dir / f'{pdbid}_r_u_f.pdb'
        lig = struct_dir / f'{pdbid}_l_u_f.pdb'
    else:
        raise ValueError('only DB5 and DB5-u are supported for HDOCKlite generation')

    for path in (rec, lig):
        if not path.exists():
            raise FileNotFoundError(f'missing input structure: {path}')
    return rec, lig


def resolve_executable(value):
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    found = shutil.which(value)
    if found:
        return found
    return value


def format_template(template, values):
    quoted = {}
    for key, value in values.items():
        if value is None:
            quoted[key] = ''
        elif key.endswith('_args'):
            quoted[key] = str(value)
        else:
            quoted[key] = shlex.quote(str(value))
    return template.format(**quoted)


def run_command(cmd, cwd, log_path, use_shell=False):
    t0 = time.time()
    with open(log_path, 'w') as log:
        log.write(f'cwd: {cwd}\n')
        if use_shell:
            log.write(f'cmd: {cmd}\n\n')
            proc = subprocess.run(
                cmd, cwd=str(cwd), shell=True, stdout=log,
                stderr=subprocess.STDOUT, text=True
            )
        else:
            log.write('cmd: ' + ' '.join(shlex.quote(str(x)) for x in cmd) + '\n\n')
            proc = subprocess.run(
                [str(x) for x in cmd], cwd=str(cwd), stdout=log,
                stderr=subprocess.STDOUT, text=True
            )
    if proc.returncode != 0:
        tail = ''
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            tail = ''.join(lines[-20:])
        except OSError:
            pass
        raise RuntimeError(
            f'command failed with exit code {proc.returncode}; log={log_path}\n{tail}'
        )
    return time.time() - t0


def run_hdock(args, pdbid, receptor, ligand, hdock_out, work_dir):
    values = {
        'hdock': args.hdock_bin,
        'receptor': receptor,
        'ligand': ligand,
        'out': hdock_out,
        'work_dir': work_dir,
        'pdb': pdbid,
        'hdock_extra_args': args.hdock_extra_args,
    }
    log_path = work_dir / 'hdock.log'
    if args.hdock_template:
        cmd = format_template(args.hdock_template, values)
        return run_command(cmd, work_dir, log_path, use_shell=True)

    cmd = [
        args.hdock_bin,
        receptor,
        ligand,
        '-out',
        hdock_out,
    ] + shlex.split(args.hdock_extra_args or '')
    return run_command(cmd, work_dir, log_path)


def run_createpl(args, pdbid, hdock_out, models_pdb, work_dir):
    values = {
        'createpl': args.createpl_bin,
        'hdock_out': hdock_out,
        'models_pdb': models_pdb,
        'nmax': args.nmax,
        'work_dir': work_dir,
        'pdb': pdbid,
        'createpl_extra_args': args.createpl_extra_args,
    }
    log_path = work_dir / 'createpl.log'
    if args.createpl_template:
        cmd = format_template(args.createpl_template, values)
        return run_command(cmd, work_dir, log_path, use_shell=True)

    cmd = [
        args.createpl_bin,
        hdock_out,
        models_pdb,
        '-nmax',
        str(args.nmax),
        '-complex',
        '-models',
    ] + shlex.split(args.createpl_extra_args or '')
    return run_command(cmd, work_dir, log_path)


def parse_models(pdb_path):
    models = []
    current = []
    saw_model = False
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                saw_model = True
                if current:
                    models.append(current)
                    current = []
                continue
            if line.startswith('ENDMDL'):
                models.append(current)
                current = []
                continue
            if line.startswith('END') and not saw_model:
                if current:
                    models.append(current)
                    current = []
                continue
            if line.startswith(('ATOM  ', 'HETATM', 'TER')):
                current.append(line)
    if current:
        models.append(current)
    return models


def chain_id(line):
    if len(line) > 21:
        return line[21]
    return ''


def split_model_lines(model_lines, rchains, lchains, receptor_input):
    rec_lines = []
    lig_lines = []
    for line in model_lines:
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        chain = chain_id(line)
        if chain in lchains:
            lig_lines.append(line)
        elif chain in rchains:
            rec_lines.append(line)

    if not lig_lines:
        raise RuntimeError(
            'no ligand atoms found after splitting complex by ligand chains '
            f'{sorted(lchains)}'
        )
    if not rec_lines:
        with open(receptor_input, 'r') as f:
            rec_lines = [
                line for line in f
                if line.startswith(('ATOM  ', 'HETATM'))
            ]
    return rec_lines, lig_lines


def write_pdb(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line if line.endswith('\n') else line + '\n')
        f.write('END\n')


def rank_lig_path(out_root, rank, pdbid):
    model = f'hdock_{rank}'
    return out_root / model / pdbid / f'{pdbid}_{model}_id.pdb'


def outputs_complete(out_root, pdbid, nmax):
    return all(rank_lig_path(out_root, rank, pdbid).exists() for rank in range(1, nmax + 1))


def write_rank_outputs(out_root, dataset, pdbid, models, rchains, lchains, receptor, nmax):
    generated = 0
    if len(models) < nmax:
        raise RuntimeError(f'createpl produced {len(models)} models, expected {nmax}')

    for rank, model_lines in enumerate(models[:nmax], 1):
        pose_model = f'hdock_{rank}'
        rank_dir = out_root / pose_model / pdbid
        rec_lines, lig_lines = split_model_lines(model_lines, rchains, lchains, receptor)

        lig_id = rank_dir / f'{pdbid}_{pose_model}_id.pdb'
        lig_plain = rank_dir / f'{pdbid}_{pose_model}.pdb'
        rec_id = rank_dir / f'{pdbid}_{pose_model}_rec_id.pdb'
        complex_path = rank_dir / f'{pdbid}_predicted.pdb'

        write_pdb(lig_id, lig_lines)
        shutil.copyfile(lig_id, lig_plain)
        write_pdb(rec_id, rec_lines)
        write_pdb(complex_path, rec_lines + lig_lines)
        generated += 1
    return generated


def process_target(args, target, index, total):
    t0 = time.time()
    paper_root = Path(args.paper_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else paper_root / 'results' / args.dataset
    work_root = Path(args.work_root).resolve()

    pdbid = target['pdb']
    receptor, ligand = input_paths(paper_root, args.dataset, pdbid)
    work_dir = work_root / args.dataset / pdbid
    hdock_out = work_dir / f'{pdbid}_hdock.out'
    models_pdb = work_dir / f'{pdbid}_top{args.nmax}_complex.pdb'

    if outputs_complete(out_root, pdbid, args.nmax) and not args.overwrite:
        return {
            'dataset': args.dataset,
            'target_index': index,
            'target_total': total,
            'target': pdbid,
            'status': 'skipped_existing',
            'message': '',
            'elapsed_sec': f'{time.time() - t0:.3f}',
            'n_requested': args.nmax,
            'n_generated': args.nmax,
            'receptor': str(receptor),
            'ligand': str(ligand),
            'hdock_out': str(hdock_out),
            'models_pdb': str(models_pdb),
            'out_root': str(out_root),
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return {
            'dataset': args.dataset,
            'target_index': index,
            'target_total': total,
            'target': pdbid,
            'status': 'dry_run',
            'message': '',
            'elapsed_sec': f'{time.time() - t0:.3f}',
            'n_requested': args.nmax,
            'n_generated': 0,
            'receptor': str(receptor),
            'ligand': str(ligand),
            'hdock_out': str(hdock_out),
            'models_pdb': str(models_pdb),
            'out_root': str(out_root),
        }

    if args.overwrite or not hdock_out.exists():
        run_hdock(args, pdbid, receptor, ligand, hdock_out, work_dir)
    if not hdock_out.exists():
        raise RuntimeError(f'HDOCK did not create expected output: {hdock_out}')

    if args.overwrite or not models_pdb.exists():
        run_createpl(args, pdbid, hdock_out, models_pdb, work_dir)
    if not models_pdb.exists():
        raise RuntimeError(f'createpl did not create expected model file: {models_pdb}')

    models = parse_models(models_pdb)
    generated = write_rank_outputs(
        out_root, args.dataset, pdbid, models,
        normalize_chains(target['rchain']),
        normalize_chains(target['lchain']),
        receptor,
        args.nmax,
    )
    return {
        'dataset': args.dataset,
        'target_index': index,
        'target_total': total,
        'target': pdbid,
        'status': 'done',
        'message': '',
        'elapsed_sec': f'{time.time() - t0:.3f}',
        'n_requested': args.nmax,
        'n_generated': generated,
        'receptor': str(receptor),
        'ligand': str(ligand),
        'hdock_out': str(hdock_out),
        'models_pdb': str(models_pdb),
        'out_root': str(out_root),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-style HDOCKlite candidates for DB5 / DB5-u'
    )
    parser.add_argument('--paper_root', required=True,
                        help='PPCBench root containing dataset/ and results/')
    parser.add_argument('--dataset', required=True, choices=['DB5', 'DB5-u'])
    parser.add_argument('--out_root', default=None,
                        help='Default: <paper_root>/results/<dataset>')
    parser.add_argument('--work_root', default='/root/autodl-tmp/hdocklite_work',
                        help='Stores HDOCK .out files and createpl multi-model PDBs')
    parser.add_argument('--hdock_bin', default=os.environ.get('HDOCK_BIN', 'hdock'))
    parser.add_argument('--createpl_bin', default=os.environ.get('CREATEPL_BIN', 'createpl_linux'))
    parser.add_argument('--hdock_extra_args', default=os.environ.get('HDOCK_EXTRA_ARGS', ''))
    parser.add_argument('--createpl_extra_args', default=os.environ.get('CREATEPL_EXTRA_ARGS', ''))
    parser.add_argument('--hdock_template', default=os.environ.get('HDOCK_TEMPLATE', ''),
                        help='Optional shell template overriding the HDOCK command')
    parser.add_argument('--createpl_template', default=os.environ.get('CREATEPL_TEMPLATE', ''),
                        help='Optional shell template overriding the createpl command')
    parser.add_argument('--nmax', type=int, default=5,
                        help='Number of ranked HDOCK poses to export')
    parser.add_argument('--targets', default='',
                        help='Comma-separated PDB IDs for a subset run')
    parser.add_argument('--start_index', type=int, default=1,
                        help='1-based start index after optional --targets filtering')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--manifest', default=None,
                        help='Default: <out_root>/hdocklite_<dataset>_manifest.csv')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--fail_fast', action='store_true')
    args = parser.parse_args()

    if args.nmax < 1:
        raise ValueError('--nmax must be >= 1')

    args.hdock_bin = resolve_executable(args.hdock_bin)
    args.createpl_bin = resolve_executable(args.createpl_bin)

    paper_root = Path(args.paper_root).resolve()
    dataset_dir = paper_root / 'dataset' / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f'missing dataset directory: {dataset_dir}')

    out_root = Path(args.out_root).resolve() if args.out_root else paper_root / 'results' / args.dataset
    manifest = Path(args.manifest).resolve() if args.manifest else out_root / f'hdocklite_{args.dataset}_manifest.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)

    targets = select_targets(read_targets(dataset_dir, args.dataset), args)
    print(f'dataset={args.dataset} targets={len(targets)} nmax={args.nmax}')
    print(f'hdock_bin={args.hdock_bin}')
    print(f'createpl_bin={args.createpl_bin}')
    print(f'out_root={out_root}')
    print(f'work_root={Path(args.work_root).resolve()}')
    print(f'manifest={manifest}')

    with open(manifest, 'w', newline='') as manifest_f:
        writer = csv.DictWriter(manifest_f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index, target in enumerate(targets, 1):
            pdbid = target['pdb']
            print(f'[{index}/{len(targets)}] {pdbid}', flush=True)
            try:
                row = process_target(args, target, index, len(targets))
            except Exception as exc:
                row = {
                    'dataset': args.dataset,
                    'target_index': index,
                    'target_total': len(targets),
                    'target': pdbid,
                    'status': 'error',
                    'message': str(exc),
                    'elapsed_sec': '0.000',
                    'n_requested': args.nmax,
                    'n_generated': 0,
                    'receptor': '',
                    'ligand': '',
                    'hdock_out': '',
                    'models_pdb': '',
                    'out_root': str(out_root),
                }
                if args.fail_fast:
                    writer.writerow(row)
                    manifest_f.flush()
                    raise
            writer.writerow(row)
            manifest_f.flush()
            print(f"  {row['status']} generated={row.get('n_generated', 0)}", flush=True)

    print(f'manifest -> {manifest}')


if __name__ == '__main__':
    main()
