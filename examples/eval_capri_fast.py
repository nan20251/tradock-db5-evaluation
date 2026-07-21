"""
TransformerDock — CAPRI Scoreset v2022 评估脚本（加速版）

加速点：
  1. surface 生成（pdb -> .ply）多进程并行，吃满 CPU 核数
  2. 临时文件用 /dev/shm（ramdisk）减少磁盘 IO
  3. 模型推理保持串行（避免改动 prepare_complex/forward），但用 GPU
  4. 模型一次性加载，所有 target 共用

预计加速：CPU 16 核 -> surface 阶段 16 倍提速；GPU vs CPU 推理 10-20 倍提速。

用法（默认评估全部 S-T*.pdb，CAPRI v2022 通常为 113 个 target）：
  python examples/eval_capri_fast.py \
      --data_dir data/database \
      --checkpoint Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
      --out results/capri_eval_113_fast.csv \
      --pos_metric classification --pos_threshold 0.3 \
      --score_type mdn \
      --n_workers 16
"""
import sys
import os
import csv
import argparse
import tempfile
import glob
import time
import shutil
import numpy as np
from multiprocessing import get_context

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformerdock.models import DeepDock_PPI, NO_INTERFACE_SCORE, ppi_score
from transformerdock.utils.data import prepare_complex
from examples.surface_gen import pdb_to_surface_ply


def split_models(pdb_path):
    """解析多模型 PDB，返回 {model_id: {chain_id: [lines]}}。"""
    models = {}
    current_model = None
    current_chains = {}

    with open(pdb_path, 'r', errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()
            if rec == 'MODEL':
                current_model = int(line.split()[1])
                current_chains = {}
            elif rec == 'ENDMDL':
                if current_model is not None:
                    models[current_model] = current_chains
                current_model = None
            elif rec in ('ATOM', 'HETATM', 'TER', 'ANISOU') and current_model is not None:
                if len(line) > 21:
                    chain = line[21]
                    if chain not in current_chains:
                        current_chains[chain] = []
                    current_chains[chain].append(line)

    if not models:
        chains = {}
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ('ATOM', 'HETATM') and len(line) > 21:
                    chain = line[21]
                    if chain not in chains:
                        chains[chain] = []
                    chains[chain].append(line)
        if chains:
            models[0] = chains

    return models


def write_chain_pdb(lines, out_path):
    with open(out_path, 'w') as f:
        f.writelines(lines)
        f.write('END\n')


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _pick(row, *keys, default=''):
    lower_map = {k.lower(): k for k in row.keys()}
    for k in keys:
        real = lower_map.get(k.lower())
        if real is not None and row[real] not in (None, ''):
            return row[real]
    return default


def _classify(dockq):
    if dockq >= 0.80:
        return 'high'
    if dockq >= 0.49:
        return 'medium'
    if dockq >= 0.23:
        return 'acceptable'
    return 'incorrect'


def load_annotations(csv_path):
    ann_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dockq = _to_float(_pick(row, 'dockq', 'DockQ', 'dockq_score'))
            fnat = _to_float(_pick(row, 'fnat', 'Fnat', 'f_nat'))
            lrms = _to_float(_pick(row, 'lrms', 'L_rms', 'lrmsd', 'L-RMSD'))
            irms = _to_float(_pick(row, 'irms', 'i_rms', 'irmsd', 'I-RMSD'))
            cls = str(_pick(row, 'classification', 'class', 'capri_class', default='')).strip()
            if not cls:
                cls = _classify(dockq)
            ident = str(_pick(row, 'identification', 'id', 'model_id', 'name', default='')).strip()
            ann_list.append({
                'dockq': dockq,
                'fnat': fnat,
                'lrms': lrms,
                'irms': irms,
                'classification': cls,
                'identification': ident,
            })
    return ann_list


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    in_channels = int(args.get('in_channels', 11))
    model = DeepDock_PPI(
        in_channels=in_channels,
        hidden_dim=args.get('hidden_dim', 128),
        n_gaussians=args.get('n_gaussians', 10),
        n_transformer_blocks=args.get('n_tf_blocks', 6),
        transformer_heads=args.get('tf_heads', 4),
        use_global_attn=True,
        global_attn_layers=2,
        cross_attn_heads=args.get('cross_heads', 8),
        n_cross_attn_layers=args.get('n_cross_layers', 2),
        dist_threshold=args.get('dist_threshold', 10.0),
        dropout_rate=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dist_threshold = args.get('dist_threshold', 10.0)
    print(f"模型加载: {checkpoint_path}  epoch={ckpt.get('epoch', '?')}  in_channels={in_channels}")
    return model, dist_threshold, in_channels


def _extract_model_num(s):
    import re as _re
    if s is None:
        return None
    m = _re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None


def _build_ann_index(ann_list, model_ids):
    ann_by_num = {}
    for a in ann_list:
        num = _extract_model_num(a.get('identification', ''))
        if num is not None and num not in ann_by_num:
            ann_by_num[num] = a

    matched = [ann_by_num.get(mid) for mid in model_ids]
    n_hit = sum(1 for a in matched if a is not None)
    if n_hit >= 0.9 * len(model_ids) and n_hit > 0:
        return matched, 'by_identification'

    n = min(len(ann_list), len(model_ids))
    return ann_list[:n] + [None] * (len(model_ids) - n), 'by_position'


GOOD_LEVELS = {'acceptable', 'medium', 'high'}
MID_LEVELS = {'medium', 'high'}
HIGH_LEVELS = {'high'}


def _positive_mask(rows, pos_metric, pos_threshold, dockq_th=0.23):
    if pos_metric == 'classification':
        return np.array([r.get('classification') in GOOD_LEVELS for r in rows], dtype=bool)
    if pos_metric == 'fnat':
        return np.array([r.get('fnat', 0.0) > pos_threshold for r in rows], dtype=bool)
    if pos_metric == 'dockq':
        return np.array([r.get('dockq', 0.0) >= pos_threshold for r in rows], dtype=bool)
    raise ValueError(f'unknown pos_metric: {pos_metric}')


def _safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float('nan')


def _float_or_zero(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(v):
        return 0.0
    return v


def compute_metrics(results, pos_metric='classification', pos_threshold=0.3, dockq_th=0.23, topks=None):
    if topks is None:
        topks = [1, 2, 5, 10, 20, 100]
    valid = [
        r for r in results
        if np.isfinite(r.get('score', np.nan))
        and np.isfinite(r.get('dockq', np.nan))
        and np.isfinite(r.get('fnat', np.nan))
    ]
    if len(valid) < 2:
        return {}

    scores = np.array([r['score'] for r in valid])
    dockqs = np.array([r['dockq'] for r in valid])
    fnats = np.array([r['fnat'] for r in valid])

    from scipy.stats import pearsonr, spearmanr
    if len(np.unique(scores)) > 1 and len(np.unique(dockqs)) > 1:
        corr, pval = spearmanr(scores, dockqs)
        pearson, _ = pearsonr(scores, dockqs)
    else:
        corr, pval, pearson = float('nan'), float('nan'), float('nan')

    head = sorted(valid, key=lambda x: x['dockq'], reverse=True)[:20]
    if len(head) >= 3:
        head_scores = np.array([r['score'] for r in head])
        head_dockqs = np.array([r['dockq'] for r in head])
        if len(np.unique(head_scores)) > 1 and len(np.unique(head_dockqs)) > 1:
            top20_spearman, _ = spearmanr(head_scores, head_dockqs)
        else:
            top20_spearman = float('nan')
    else:
        top20_spearman = float('nan')

    ranked = sorted(valid, key=lambda x: x['score'], reverse=True)
    top1_dockq = ranked[0]['dockq']
    top1_class = ranked[0]['classification']

    pos = _positive_mask(valid, pos_metric, pos_threshold, dockq_th)
    any_pos = np.array([r.get('classification') in GOOD_LEVELS for r in valid], dtype=bool)
    med_pos = np.array([r.get('classification') in MID_LEVELS for r in valid], dtype=bool)
    high_pos = np.array([r.get('classification') in HIGH_LEVELS for r in valid], dtype=bool)
    dockq_pos = dockqs >= dockq_th

    ranked_pos = _positive_mask(ranked, pos_metric, pos_threshold, dockq_th)
    ranked_any = np.array([r.get('classification') in GOOD_LEVELS for r in ranked], dtype=bool)
    ranked_med = np.array([r.get('classification') in MID_LEVELS for r in ranked], dtype=bool)
    ranked_high = np.array([r.get('classification') in HIGH_LEVELS for r in ranked], dtype=bool)
    ranked_dockq = np.array([r.get('dockq', 0.0) >= dockq_th for r in ranked], dtype=bool)

    def success_in_topn(labels, n):
        top = labels[:min(n, len(labels))]
        return int(bool(top.any()))

    def hitrate_in_topn(labels, n):
        top = ranked[:min(n, len(ranked))]
        if not top:
            return 0.0
        return float(labels[:len(top)].mean())

    n = len(valid)
    n_top1pct = max(1, int(n * 0.01))
    n_acceptable_total = int(pos.sum())
    random_rate = n_acceptable_total / n if n > 0 else 0
    top1pct_rate = ranked_pos[:n_top1pct].mean()
    ef1 = top1pct_rate / random_rate if random_rate > 0 else 0.0

    out = {
        'n_models': n,
        'n_decoys': n,
        'n_acceptable': n_acceptable_total,
        'n_acceptable_plus': int(any_pos.sum()),
        'n_medium_plus': int(med_pos.sum()),
        'n_high': int(high_pos.sum()),
        'n_positive': n_acceptable_total,
        'auc': _float_or_zero(_safe_auc(pos, scores)),
        'spearman_r': _float_or_zero(corr),
        'spearman_p': _float_or_zero(pval) if not np.isnan(pval) else 1.0,
        'spearman': _float_or_zero(corr),
        'pearson': _float_or_zero(pearson),
        'top20_spearman': _float_or_zero(top20_spearman),
        'auc_pos': _float_or_zero(_safe_auc(pos, scores)),
        'auc_any': _float_or_zero(_safe_auc(any_pos, scores)),
        'auc_med': _float_or_zero(_safe_auc(med_pos, scores)),
        'auc_high': _float_or_zero(_safe_auc(high_pos, scores)),
        'auc_dockq': _float_or_zero(_safe_auc(dockq_pos, scores)),
        'top1_dockq': top1_dockq,
        'top1_class': top1_class,
        'ef1pct': ef1,
    }
    for k in topks:
        out[f'success_top{k}'] = success_in_topn(ranked_pos, k)
        out[f'success@{k}'] = success_in_topn(ranked_pos, k)
        out[f'success_any@{k}'] = success_in_topn(ranked_any, k)
        out[f'success_dockq@{k}'] = success_in_topn(ranked_dockq, k)
        out[f'success_med@{k}'] = success_in_topn(ranked_med, k)
        out[f'success_high@{k}'] = success_in_topn(ranked_high, k)
        out[f'hitrate@{k}'] = hitrate_in_topn(ranked_dockq, k)
    return out

DETAIL_FIELDS = [
    'target_index', 'target_total', 'target',
    'model_id', 'score', 'mdn_score', 'energy_score', 'enhanced_score',
    'n_interface_pairs', 'n_close_pairs', 'n_clash_pairs',
    'contact_bonus', 'close_contact_bonus', 'sparse_penalty', 'clash_penalty',
    'score_reason',
    'dockq', 'fnat', 'lrms', 'irms',
    'classification', 'identification',
]

SUMMARY_FIELDS = [
    'target_index', 'target_total', 'target', 'status', 'message',
    'elapsed_sec',
    'n_models', 'n_acceptable', 'auc', 'spearman_r', 'spearman_p',
    'top1_dockq', 'top1_class',
    'success_top1', 'success_top2', 'success_top5',
    'success_top10', 'success_top20', 'success_top100', 'ef1pct',
    'n_decoys', 'n_acceptable_plus', 'n_medium_plus', 'n_high',
    'n_positive', 'spearman', 'pearson', 'top20_spearman',
    'auc_pos', 'auc_any', 'auc_med', 'auc_high', 'auc_dockq',
    'success@1', 'success_any@1', 'success_dockq@1', 'success_med@1', 'success_high@1', 'hitrate@1',
    'success@2', 'success_any@2', 'success_dockq@2', 'success_med@2', 'success_high@2', 'hitrate@2',
    'success@5', 'success_any@5', 'success_dockq@5', 'success_med@5', 'success_high@5', 'hitrate@5',
    'success@10', 'success_any@10', 'success_dockq@10', 'success_med@10', 'success_high@10', 'hitrate@10',
    'success@20', 'success_any@20', 'success_dockq@20', 'success_med@20', 'success_high@20', 'hitrate@20',
    'success@100', 'success_any@100', 'success_dockq@100', 'success_med@100', 'success_high@100', 'hitrate@100',
]


def _summary_path(out_path):
    if out_path.endswith('.csv'):
        return out_path[:-4] + '.summary.csv'
    return out_path + '.summary.csv'


def _open_incremental_csv(path, fieldnames, append=False):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    mode = 'a' if append else 'w'
    exists_with_data = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, mode, newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    if mode == 'w' or not exists_with_data:
        writer.writeheader()
        _flush_csv(f)
    return f, writer


def _flush_csv(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def _summary_row(index, total, target, status, message='', elapsed_sec=0.0, metrics=None):
    metrics = metrics or {}
    row = {
        'target_index': index,
        'target_total': total,
        'target': target,
        'status': status,
        'message': message,
        'elapsed_sec': f'{elapsed_sec:.3f}',
    }
    for key in SUMMARY_FIELDS:
        if key not in row:
            row[key] = metrics.get(key, 0)
    return row


def _load_existing_targets(summary_path):
    if not os.path.exists(summary_path):
        return set()
    done_statuses = {'done', 'missing_csv', 'no_valid_models', 'no_metrics', 'error'}
    targets = set()
    with open(summary_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') in done_statuses and row.get('target'):
                targets.add(row['target'])
    return targets


def _write_aggregate_summary(summary_path, pos_metric, pos_threshold,
                             success_denominator='with_positives'):
    if not os.path.exists(summary_path):
        return None
    rows = []
    with open(summary_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'done':
                rows.append(row)
    if not rows:
        return None

    def mean_field(name, nonzero=False):
        vals = []
        for r in rows:
            raw = r.get(name, '')
            if raw in ('', None):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if nonzero and val == 0.0:
                continue
            vals.append(val)
        return float(np.mean(vals)) if vals else 0.0

    n_targets = len(rows)
    with_positive = [r for r in rows if int(float(r.get('n_positive', 0) or 0)) > 0]
    denom_rows = with_positive if success_denominator == 'with_positives' else rows
    denom = len(denom_rows) or 1

    out = {
        'pos_metric': pos_metric,
        'pos_threshold': pos_threshold,
        'success_denominator': success_denominator,
        'n_targets': n_targets,
        'n_targets_with_positive': len(with_positive),
        'n_targets_with_acceptable+': sum(int(float(r.get('n_acceptable_plus', 0) or 0)) > 0 for r in rows),
        'n_targets_with_medium+': sum(int(float(r.get('n_medium_plus', 0) or 0)) > 0 for r in rows),
        'n_targets_with_high': sum(int(float(r.get('n_high', 0) or 0)) > 0 for r in rows),
        'mean_spearman': mean_field('spearman'),
        'median_spearman': float(np.median([float(r.get('spearman', 0) or 0) for r in rows])),
        'mean_pearson': mean_field('pearson'),
        'mean_top20_spearman': mean_field('top20_spearman'),
        'mean_auc_pos': mean_field('auc_pos', nonzero=True),
        'std_auc_pos': float(np.std([
            float(r.get('auc_pos', 0) or 0) for r in rows
            if float(r.get('auc_pos', 0) or 0) != 0
        ])) if any(float(r.get('auc_pos', 0) or 0) != 0 for r in rows) else 0.0,
        'mean_auc_any': mean_field('auc_any', nonzero=True),
        'mean_auc_med': mean_field('auc_med', nonzero=True),
        'mean_auc_high': mean_field('auc_high', nonzero=True),
        'mean_auc_dockq': mean_field('auc_dockq', nonzero=True),
    }
    for k in (1, 2, 5, 10, 20, 100):
        out[f'success@{k}'] = sum(int(float(r.get(f'success@{k}', 0) or 0)) for r in denom_rows) / denom
        out[f'success_any@{k}'] = sum(int(float(r.get(f'success_any@{k}', 0) or 0)) for r in denom_rows) / denom
        out[f'success_dockq@{k}'] = sum(int(float(r.get(f'success_dockq@{k}', 0) or 0)) for r in denom_rows) / denom
        med_rows = [r for r in rows if int(float(r.get('n_medium_plus', 0) or 0)) > 0]
        high_rows = [r for r in rows if int(float(r.get('n_high', 0) or 0)) > 0]
        out[f'success_med@{k}'] = (
            sum(int(float(r.get(f'success_med@{k}', 0) or 0)) for r in med_rows) / len(med_rows)
            if med_rows else 0.0
        )
        out[f'success_high@{k}'] = (
            sum(int(float(r.get(f'success_high@{k}', 0) or 0)) for r in high_rows) / len(high_rows)
            if high_rows else 0.0
        )
        out[f'mean_hitrate@{k}'] = mean_field(f'hitrate@{k}')

    agg_path = summary_path[:-4] + '.aggregate.csv' if summary_path.endswith('.csv') else summary_path + '.aggregate.csv'
    with open(agg_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        writer.writeheader()
        writer.writerow(out)
    return agg_path


def gen_surface_one(args):
    """单个 decoy 的 surface 生成（pure CPU，可并行）。返回 (mid, rec_ply, lig_ply) 或 None。"""
    mid, chains_rec, chains_lig, workdir, voxel_size = args
    rec_pdb = os.path.join(workdir, f'rec_{mid}.pdb')
    lig_pdb = os.path.join(workdir, f'lig_{mid}.pdb')
    rec_ply = os.path.join(workdir, f'rec_{mid}.ply')
    lig_ply = os.path.join(workdir, f'lig_{mid}.ply')

    try:
        write_chain_pdb(chains_rec, rec_pdb)
        write_chain_pdb(chains_lig, lig_pdb)
        ok_r = pdb_to_surface_ply(rec_pdb, rec_ply, voxel_size=voxel_size)
        ok_l = pdb_to_surface_ply(lig_pdb, lig_ply, voxel_size=voxel_size)
        # PDB 文件不再需要，立刻删
        for f in (rec_pdb, lig_pdb):
            try: os.remove(f)
            except: pass
        if not ok_r or not ok_l:
            return None
        return (mid, rec_ply, lig_ply)
    except Exception:
        return None


@torch.no_grad()
def score_one(model, dist_threshold, device, rec_ply, lig_ply,
              score_type='mdn', fusion_alpha=0.5,
              contact_weight=0.003, close_contact_weight=0.002,
              sparse_weight=0.02, clash_weight=0.02,
              min_interface_pairs=25, close_threshold=5.0,
              in_channels=11,
              clash_threshold=1.5):
    rec_data, lig_data = prepare_complex(rec_ply, lig_ply, in_channels=in_channels)
    rec_data = rec_data.to(device)
    lig_data = lig_data.to(device)
    amp_on = str(device).startswith('cuda')
    with torch.cuda.amp.autocast(enabled=amp_on):
        pi, sigma, mu, dist, _, pred_energy = model(rec_data, lig_data)
    pi, sigma, mu, dist = pi.float(), sigma.float(), mu.float(), dist.float()
    d = dist.squeeze(1)
    n_interface_pairs = int((d < dist_threshold).sum().item())
    n_close_pairs = int((d < close_threshold).sum().item())
    n_clash_pairs = int((d < clash_threshold).sum().item())
    mdn_score = ppi_score(pi, sigma, mu, dist, dist_threshold)
    energy_score = float(pred_energy.float().item())
    score_reason = 'ok' if n_interface_pairs > 0 else 'no_interface_pairs'
    contact_bonus = float(np.log1p(n_interface_pairs))
    close_contact_bonus = float(np.log1p(n_close_pairs))
    sparse_penalty = float(max(0, min_interface_pairs - n_interface_pairs) / max(1, min_interface_pairs))
    clash_penalty = float(np.log1p(n_clash_pairs))
    if mdn_score <= NO_INTERFACE_SCORE:
        enhanced_score = mdn_score
    else:
        enhanced_score = (
            mdn_score
            + fusion_alpha * energy_score
            + contact_weight * contact_bonus
            + close_contact_weight * close_contact_bonus
            - sparse_weight * sparse_penalty
            - clash_weight * clash_penalty
        )
    info = {
        'mdn_score': mdn_score,
        'energy_score': energy_score,
        'enhanced_score': enhanced_score,
        'n_interface_pairs': n_interface_pairs,
        'n_close_pairs': n_close_pairs,
        'n_clash_pairs': n_clash_pairs,
        'contact_bonus': contact_bonus,
        'close_contact_bonus': close_contact_bonus,
        'sparse_penalty': sparse_penalty,
        'clash_penalty': clash_penalty,
        'score_reason': score_reason,
    }
    if score_type == 'energy':
        info['score'] = energy_score
        return info
    if score_type == 'fusion':
        if mdn_score <= NO_INTERFACE_SCORE:
            info['score'] = mdn_score
        else:
            info['score'] = fusion_alpha * mdn_score + (1 - fusion_alpha) * energy_score
        return info
    if score_type == 'enhanced':
        info['score'] = enhanced_score
        return info
    info['score'] = mdn_score
    return info


def eval_target_fast(pdb_path, csv_path, model, dist_threshold, device,
                     workdir, pool, max_models=None, voxel_size=3.5,
                     score_type='mdn', fusion_alpha=0.5,
                     contact_weight=0.03, close_contact_weight=0.02,
                     sparse_weight=0.20, clash_weight=0.20,
                     min_interface_pairs=25, close_threshold=5.0,
                     in_channels=11,
                     clash_threshold=1.5):
    """
    并行 surface 生成 + 串行 GPU 推理。
    """
    ann_list = load_annotations(csv_path)
    models = split_models(pdb_path)
    model_ids = sorted(models.keys())
    if max_models:
        model_ids = model_ids[:max_models]

    matched_anns, match_mode = _build_ann_index(ann_list, model_ids)
    if len(ann_list) != len(model_ids):
        print(f"  [警告] PDB MODEL 数 {len(model_ids)} ≠ CSV 行数 {len(ann_list)}, 模式={match_mode}")

    # 1. 并行 surface 生成
    tasks = []
    for mid in model_ids:
        chains = models[mid]
        ch_ids = list(chains.keys())
        if len(ch_ids) < 2:
            continue
        rec_ch = 'A' if 'A' in chains else ch_ids[0]
        lig_ch = 'B' if 'B' in chains else ch_ids[1]
        tasks.append((mid, chains[rec_ch], chains[lig_ch], workdir, voxel_size))

    t0 = time.time()
    if pool is not None:
        surfs = pool.map(gen_surface_one, tasks)
    else:
        surfs = [gen_surface_one(t) for t in tasks]
    surfs = [s for s in surfs if s is not None]
    t_surf = time.time() - t0

    # 2. 串行 GPU 推理
    t0 = time.time()
    results = []
    id2idx = {mid: i for i, mid in enumerate(model_ids)}
    for mid, rec_ply, lig_ply in surfs:
        try:
            score_info = score_one(
                model, dist_threshold, device, rec_ply, lig_ply,
                score_type=score_type, fusion_alpha=fusion_alpha,
                contact_weight=contact_weight,
                close_contact_weight=close_contact_weight,
                sparse_weight=sparse_weight,
                clash_weight=clash_weight,
                min_interface_pairs=min_interface_pairs,
                close_threshold=close_threshold,
                in_channels=in_channels,
                clash_threshold=clash_threshold,
            )
        except Exception:
            score_info = {
                'score': float('-inf'),
                'mdn_score': float('-inf'),
                'energy_score': 0.0,
                'enhanced_score': float('-inf'),
                'n_interface_pairs': 0,
                'n_close_pairs': 0,
                'n_clash_pairs': 0,
                'contact_bonus': 0.0,
                'close_contact_bonus': 0.0,
                'sparse_penalty': 1.0,
                'clash_penalty': 0.0,
                'score_reason': 'error',
            }
        idx = id2idx[mid]
        a = matched_anns[idx] if idx < len(matched_anns) and matched_anns[idx] else {}
        results.append({
            'model_id':       mid,
            'score':          score_info['score'],
            'mdn_score':      score_info['mdn_score'],
            'energy_score':   score_info['energy_score'],
            'enhanced_score': score_info['enhanced_score'],
            'n_interface_pairs': score_info['n_interface_pairs'],
            'n_close_pairs':  score_info['n_close_pairs'],
            'n_clash_pairs':  score_info['n_clash_pairs'],
            'contact_bonus':  score_info['contact_bonus'],
            'close_contact_bonus': score_info['close_contact_bonus'],
            'sparse_penalty': score_info['sparse_penalty'],
            'clash_penalty':  score_info['clash_penalty'],
            'score_reason':   score_info['score_reason'],
            'dockq':          a.get('dockq', 0.0),
            'fnat':           a.get('fnat', 0.0),
            'lrms':           a.get('lrms', 0.0),
            'irms':           a.get('irms', 0.0),
            'classification': a.get('classification', 'unknown'),
            'identification': a.get('identification', ''),
        })
        for f in (rec_ply, lig_ply):
            try: os.remove(f)
            except: pass
    t_infer = time.time() - t0

    print(f"  surface={t_surf:.1f}s  infer={t_infer:.1f}s  n_models={len(results)}")
    return results


def main():
    p = argparse.ArgumentParser(description='CAPRI Scoreset v2022 评估（加速版）')
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--out',         default='results/capri_eval.csv')
    p.add_argument('--pos_metric',  default='classification',
                   choices=['classification', 'dockq', 'fnat'])
    p.add_argument('--pos_threshold', type=float, default=0.3)
    p.add_argument('--dockq_th', type=float, default=0.23,
                   help='DockQ threshold for hit-rate metrics')
    p.add_argument('--success_denominator', choices=['all', 'with_positives'],
                   default='with_positives',
                   help='Aggregate success denominator for *.aggregate.csv')
    p.add_argument('--score_type',  default='mdn', choices=['mdn', 'energy', 'fusion', 'enhanced'])
    p.add_argument('--fusion_alpha', type=float, default=0.5)
    p.add_argument('--contact_weight', type=float, default=0.003)
    p.add_argument('--close_contact_weight', type=float, default=0.002)
    p.add_argument('--sparse_weight', type=float, default=0.02)
    p.add_argument('--clash_weight', type=float, default=0.02)
    p.add_argument('--min_interface_pairs', type=int, default=25)
    p.add_argument('--close_threshold', type=float, default=5.0)
    p.add_argument('--clash_threshold', type=float, default=1.5)
    p.add_argument('--voxel_size',  type=float, default=3.5)
    p.add_argument('--max_models',  type=int, default=None)
    p.add_argument('--n_workers',   type=int, default=os.cpu_count() or 4,
                   help='surface 生成并行 worker 数')
    p.add_argument('--skip', type=int, default=0,
                   help='跳过排序后的前 N 个 target；跳过记录也会写入 summary CSV')
    p.add_argument('--append', action='store_true',
                   help='追加写入已有 CSV；默认 skip>0 时自动追加，否则覆盖输出')
    p.add_argument('--resume', action='store_true',
                   help='读取已有 summary CSV，跳过已记录完成/失败的 target，并追加写入')
    p.add_argument('--ramdisk',     type=str, default='/dev/shm',
                   help='临时文件目录，建议 /dev/shm（ramdisk）')
    p.add_argument('--targets_file', type=str, default=None,
                   help='只评估文件中列出的 target（每行一个，如 S-T047.1）')
    args = p.parse_args()

    pdb_files = sorted(glob.glob(os.path.join(args.data_dir, 'S-T*.pdb')))
    if args.targets_file:
        with open(args.targets_file, 'r', encoding='utf-8') as f:
            wanted = {
                line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')
            }
        before = len(pdb_files)
        pdb_files = [
            p for p in pdb_files
            if os.path.splitext(os.path.basename(p))[0] in wanted
        ]
        print(f"targets_file={args.targets_file}: {before} -> {len(pdb_files)} targets")
        missing = sorted(wanted - {os.path.splitext(os.path.basename(p))[0] for p in pdb_files})
        if missing:
            print(f"  missing in data_dir ({len(missing)}): {', '.join(missing[:12])}"
                  + (' ...' if len(missing) > 12 else ''))
    print(f"共 {len(pdb_files)} 个 target\n")

    # 临时目录优先用 /dev/shm
    if args.ramdisk and os.path.isdir(args.ramdisk):
        tmpdir = tempfile.mkdtemp(prefix='trad_eval_', dir=args.ramdisk)
    else:
        tmpdir = tempfile.mkdtemp(prefix='trad_eval_')
    print(f"临时目录: {tmpdir}\n")

    summary_rows = []
    t_global = time.time()
    append_mode = args.append or args.skip > 0 or args.resume
    summary_path = _summary_path(args.out)
    existing_targets = _load_existing_targets(summary_path) if args.resume else set()
    if existing_targets:
        print(f"断点续跑: 已有 {len(existing_targets)} 个 target 记录，将跳过")
    detail_f, detail_writer = _open_incremental_csv(
        args.out, DETAIL_FIELDS, append=append_mode
    )
    summary_f, summary_writer = _open_incremental_csv(
        summary_path, SUMMARY_FIELDS, append=append_mode
    )

    pool = None
    if args.n_workers > 1:
        pool = get_context('spawn').Pool(processes=args.n_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}  workers: {args.n_workers}  ramdisk: {args.ramdisk}")

    model, dist_threshold, in_channels = load_model(args.checkpoint, device)

    try:
        for i, pdb_path in enumerate(pdb_files, 1):
            step_t0 = time.time()
            base = os.path.splitext(os.path.basename(pdb_path))[0]
            if i <= args.skip:
                print(f"[{i}/{len(pdb_files)}] {base}: skip")
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'skipped_by_arg',
                    message=f'--skip {args.skip}',
                ))
                _flush_csv(summary_f)
                continue
            if base in existing_targets:
                print(f"[{i}/{len(pdb_files)}] {base}: resume skip")
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'skipped_existing',
                    message='already recorded in summary',
                ))
                _flush_csv(summary_f)
                continue

            csv_path = pdb_path.replace('.pdb', '.csv')
            if not os.path.exists(csv_path):
                print(f"[{i}/{len(pdb_files)}] {base}: 无 CSV 标注，跳过")
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'missing_csv',
                    message=f'missing {csv_path}',
                    elapsed_sec=time.time() - step_t0,
                ))
                _flush_csv(summary_f)
                continue
            print(f"[{i}/{len(pdb_files)}] {base}")

            try:
                results = eval_target_fast(
                    pdb_path, csv_path, model, dist_threshold, device, tmpdir,
                    pool, args.max_models, voxel_size=args.voxel_size,
                    score_type=args.score_type, fusion_alpha=args.fusion_alpha,
                    contact_weight=args.contact_weight,
                    close_contact_weight=args.close_contact_weight,
                    sparse_weight=args.sparse_weight,
                    clash_weight=args.clash_weight,
                    min_interface_pairs=args.min_interface_pairs,
                    close_threshold=args.close_threshold,
                    in_channels=in_channels,
                    clash_threshold=args.clash_threshold,
                )
            except Exception as exc:
                print(f"  [错误] {exc}")
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'error',
                    message=str(exc),
                    elapsed_sec=time.time() - step_t0,
                ))
                _flush_csv(summary_f)
                continue

            if not results:
                print("  无有效模型，跳过")
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'no_valid_models',
                    elapsed_sec=time.time() - step_t0,
                ))
                _flush_csv(summary_f)
                continue

            metrics = compute_metrics(results, pos_metric=args.pos_metric,
                                      pos_threshold=args.pos_threshold,
                                      dockq_th=args.dockq_th)
            if not metrics:
                print("  有结果但不足以计算指标，跳过汇总")
                for r in results:
                    detail_writer.writerow({'target_index': i, 'target_total': len(pdb_files),
                                            'target': base, **r})
                _flush_csv(detail_f)
                summary_writer.writerow(_summary_row(
                    i, len(pdb_files), base, 'no_metrics',
                    message='less than 2 valid models',
                    elapsed_sec=time.time() - step_t0,
                ))
                _flush_csv(summary_f)
                continue

            print(
                f"  AUC={metrics.get('auc',0):.3f}  "
                f"Spearman={metrics.get('spearman_r',0):.3f}  "
                f"Succ@1={metrics.get('success_top1',0)}  "
                f"Succ@5={metrics.get('success_top5',0)}  "
                f"Succ@10={metrics.get('success_top10',0)}  "
                f"Succ@20={metrics.get('success_top20',0)}"
            )

            for r in results:
                detail_writer.writerow({'target_index': i, 'target_total': len(pdb_files),
                                        'target': base, **r})
            _flush_csv(detail_f)

            summary_writer.writerow(_summary_row(
                i, len(pdb_files), base, 'done',
                elapsed_sec=time.time() - step_t0,
                metrics=metrics,
            ))
            _flush_csv(summary_f)
            summary_rows.append({'target': base, **metrics})
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        detail_f.close()
        summary_f.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n总耗时 {(time.time() - t_global)/60:.1f} 分钟")
    print(f"detail  -> {args.out}")
    print(f"summary -> {summary_path}")
    agg_path = _write_aggregate_summary(
        summary_path, args.pos_metric, args.pos_threshold,
        success_denominator=args.success_denominator,
    )
    if agg_path:
        print(f"aggregate -> {agg_path}")

    # 汇总
    if summary_rows:
        n = len(summary_rows)
        succ1  = sum(r['success_top1']   for r in summary_rows)
        succ2  = sum(r['success_top2']   for r in summary_rows)
        succ5  = sum(r['success_top5']   for r in summary_rows)
        succ10 = sum(r['success_top10']  for r in summary_rows)
        succ20 = sum(r['success_top20']  for r in summary_rows)
        succ100= sum(r['success_top100'] for r in summary_rows)
        aucs   = [r['auc'] for r in summary_rows if r['auc'] > 0]
        spearmans = [r['spearman_r'] for r in summary_rows]
        print("\n=== TransformerDock CAPRI Score_set 评估 ===")
        print(f"Targets: {n}    pos_metric={args.pos_metric}>{args.pos_threshold}")
        if aucs:
            print(f"Mean AUC:     {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"Mean Spearman: {np.mean(spearmans):.3f}")
        print(f"Success@1:   {100*succ1/n:.1f}%   ({succ1}/{n})")
        print(f"Success@2:   {100*succ2/n:.1f}%   ({succ2}/{n})")
        print(f"Success@5:   {100*succ5/n:.1f}%   ({succ5}/{n})")
        print(f"Success@10:  {100*succ10/n:.1f}%  ({succ10}/{n})")
        print(f"Success@20:  {100*succ20/n:.1f}%  ({succ20}/{n})")
        print(f"Success@100: {100*succ100/n:.1f}% ({succ100}/{n})")


if __name__ == '__main__':
    main()
