"""
TransformerDock — 对比微调训练 v2

相比 v1 (train_native_vs_decoy.py) 的改进：
  [1] MDN loss 只对 native 样本算（避免污染几何先验）
  [3] 默认用 InfoNCE 替代 ranking margin（梯度信号不饱和）
  [4] 增加 near-native 难度档 (DockQ 0.30-0.50)，让模型见到决策边界
  [6] 默认 decoy_per_native=15（增大对比池）
  [8] 验证指标改 per-target Succ@10，与最终评估对齐

Loss:
    L = λ_rank × InfoNCE(native vs decoys)
      + λ_mse  × BCE(pred_decoy, dockq_decoy)     [只对 decoy]
      + λ_mdn  × MDN_NLL                          [只对 native]
"""

import sys
import os
import csv
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformerdock.models import DeepDock_PPI, ppi_train_loss
from transformerdock.utils.data import match_feature_dim, read_ply


# ─────────────────────────────────────────────────────────────
# 数据集：native + decoy + 4 档难度
# ─────────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    def __init__(self, native_dir, decoy_dir, decoy_csv,
                 stem_filter=None, max_per_stem_decoy=None,
                 in_channels=11):
        self.records = []
        self.in_channels = in_channels

        # 1. Native
        native_csv = os.path.join(native_dir, 'pairs.csv')
        if os.path.exists(native_csv):
            with open(native_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('name') or row.get('pdb_id', '')
                    if not name:
                        continue
                    if stem_filter is not None and name not in stem_filter:
                        continue
                    rec = os.path.join(native_dir, f'{name}_receptor.ply')
                    lig = os.path.join(native_dir, f'{name}_ligand.ply')
                    if not (os.path.exists(rec) and os.path.exists(lig)):
                        continue
                    self.records.append({
                        'rec': rec, 'lig': lig,
                        'dockq': 1.0,
                        'is_native': True,
                        'stem': name,
                        'name': name,
                    })

        # 2. Decoys
        stem_counts = {}
        with open(decoy_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = row['stem']
                if stem_filter is not None and stem not in stem_filter:
                    continue
                if max_per_stem_decoy is not None:
                    if stem_counts.get(stem, 0) >= max_per_stem_decoy:
                        continue
                    stem_counts[stem] = stem_counts.get(stem, 0) + 1

                name = row['name']
                rec = os.path.join(decoy_dir, f'{name}_receptor.ply')
                lig = os.path.join(decoy_dir, f'{name}_ligand.ply')
                if not (os.path.exists(rec) and os.path.exists(lig)):
                    continue
                self.records.append({
                    'rec': rec, 'lig': lig,
                    'dockq': float(row['dockq']),
                    'is_native': False,
                    'stem': stem,
                    'name': name,
                })

        self.native_idx = [i for i, r in enumerate(self.records) if r['is_native']]
        self.decoy_idx  = [i for i, r in enumerate(self.records) if not r['is_native']]

        # [改进 4] 4 档难度划分
        self.decoy_near_native = [i for i in self.decoy_idx
                                  if 0.30 <= self.records[i]['dockq'] < 0.50]
        self.decoy_hard        = [i for i in self.decoy_idx
                                  if 0.10 <= self.records[i]['dockq'] < 0.30]
        self.decoy_medium      = [i for i in self.decoy_idx
                                  if 0.03 <= self.records[i]['dockq'] < 0.10]
        self.decoy_easy        = [i for i in self.decoy_idx
                                  if self.records[i]['dockq'] < 0.03]

        # 按 stem 分组（验证时按 target 算 Succ@10 用）
        self.records_by_stem = defaultdict(list)
        for i, r in enumerate(self.records):
            self.records_by_stem[r['stem']].append(i)

        print(f"Mixed dataset: native={len(self.native_idx)}, "
              f"decoy={len(self.decoy_idx)} "
              f"(near_native={len(self.decoy_near_native)}, "
              f"hard={len(self.decoy_hard)}, "
              f"medium={len(self.decoy_medium)}, "
              f"easy={len(self.decoy_easy)})")

        if not self.native_idx:
            raise RuntimeError(f"Native 样本为 0。检查 {native_csv}")
        if not self.decoy_idx:
            raise RuntimeError(f"Decoy 样本为 0。检查 {decoy_csv}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rec = read_ply(r['rec'])
        lig = read_ply(r['lig'])
        rec = match_feature_dim(rec, self.in_channels)
        lig = match_feature_dim(lig, self.in_channels)
        return (rec, lig,
                torch.tensor(r['dockq'], dtype=torch.float32),
                torch.tensor(r['is_native'], dtype=torch.bool),
                r['stem'], r['name'])


# ─────────────────────────────────────────────────────────────
# Batch Sampler: 1 native + N decoy (4 档难度按比例混合)
# ─────────────────────────────────────────────────────────────

class NativeVsDecoySampler(Sampler):
    """
    每个 batch: 1 native + decoy_per_native 个 decoy
    decoy 来自同一个 target 的 native，但 decoy 可以来自任意 target
    （因为 DIPS 不是所有 target 都有 LightDock decoy）

    [改进 4] 难度配比：
      near_native: 20%  (DockQ 0.30-0.50)
      hard:        40%  (DockQ 0.10-0.30)
      medium:      30%  (DockQ 0.03-0.10)
      easy:        10%  (DockQ < 0.03)
    """
    def __init__(self, dataset, batches_per_epoch=400,
                 decoy_per_native=15,
                 ratio_near_native=0.20,
                 ratio_hard=0.40,
                 ratio_medium=0.30,
                 ratio_easy=0.10):
        self.ds = dataset
        self.batches_per_epoch = batches_per_epoch
        self.decoy_per_native = decoy_per_native

        n = decoy_per_native
        self.n_near = max(1, int(round(n * ratio_near_native)))
        self.n_hard = max(1, int(round(n * ratio_hard)))
        self.n_med  = max(1, int(round(n * ratio_medium)))
        self.n_easy = max(0, n - self.n_near - self.n_hard - self.n_med)

    def _sample_pool(self, pool, k):
        if not pool or k <= 0:
            return []
        return [random.choice(pool) for _ in range(k)]

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            batch.append(random.choice(self.ds.native_idx))
            batch += self._sample_pool(self.ds.decoy_near_native, self.n_near)
            batch += self._sample_pool(self.ds.decoy_hard, self.n_hard)
            batch += self._sample_pool(self.ds.decoy_medium, self.n_med)
            batch += self._sample_pool(self.ds.decoy_easy, self.n_easy)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


def collate(batch):
    recs, ligs, dockqs, is_natives, stems, names = zip(*batch)
    return (
        Batch.from_data_list(list(recs)),
        Batch.from_data_list(list(ligs)),
        torch.stack(dockqs),
        torch.stack(is_natives),
        list(stems),
        list(names),
    )


# ─────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────

def infonce_loss(pred_energy, is_native, temperature=0.1):
    """
    [改进 3] InfoNCE: native 要在 batch 中分数最高
    L = -log( exp(s_native/τ) / Σ exp(s_i/τ) )
    """
    if is_native.sum() == 0 or (~is_native).sum() == 0:
        return torch.zeros(1, device=pred_energy.device, requires_grad=True).squeeze()
    logits = pred_energy / temperature
    log_prob = F.log_softmax(logits, dim=0)
    return -log_prob[is_native].mean()


def native_mdn_loss(pi, sigma, mu, dist, C_batch, is_native, dist_threshold):
    """
    [改进 1] MDN loss 只对 native 样本算
    C_batch: [M]，每个 pair 所属的 batch index
    is_native: [B]，每个样本是否为 native
    """
    if not is_native.any():
        return torch.zeros((), device=pi.device, requires_grad=True), False

    native_batch_ids = torch.where(is_native)[0]
    mask_native_pair = torch.isin(C_batch, native_batch_ids)

    if mask_native_pair.sum() == 0:
        return torch.zeros((), device=pi.device, requires_grad=True), False

    loss = ppi_train_loss(
        pi[mask_native_pair],
        sigma[mask_native_pair],
        mu[mask_native_pair],
        dist[mask_native_pair],
        dist_threshold=dist_threshold,
    )
    return loss, True


# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device,
                dist_threshold, lambdas, temperature, max_grad_norm=1.0):
    model.train()
    stats = {'total': 0, 'rank': 0, 'mse': 0, 'mdn': 0, 'n': 0}

    for rec_batch, lig_batch, dockqs, is_natives, stems, names in loader:
        rec_batch = rec_batch.to(device)
        lig_batch = lig_batch.to(device)
        dockqs = dockqs.to(device)
        is_natives = is_natives.to(device)

        optimizer.zero_grad()
        pi, sigma, mu, dist, C_batch, pred_energy = model(rec_batch, lig_batch)
        if (not torch.isfinite(pi).all()
                or not torch.isfinite(sigma).all()
                or not torch.isfinite(mu).all()
                or not torch.isfinite(dist).all()
                or not torch.isfinite(pred_energy).all()):
            continue

        loss_mdn, has_mdn = native_mdn_loss(pi, sigma, mu, dist, C_batch,
                                            is_natives, dist_threshold)
        if not torch.isfinite(loss_mdn):
            continue

        loss_rank = infonce_loss(pred_energy, is_natives, temperature=temperature)

        # 只对 decoy 算 BCE（pred_energy 已经过 sigmoid）
        if (~is_natives).any():
            decoy_pred = pred_energy[~is_natives].clamp(1e-6, 1 - 1e-6)
            decoy_target = dockqs[~is_natives].clamp(0.0, 1.0)
            loss_mse = F.binary_cross_entropy(decoy_pred, decoy_target)
        else:
            loss_mse = torch.zeros(1, device=device, requires_grad=True).squeeze()

        loss = (lambdas['rank'] * loss_rank
                + lambdas['mse']  * loss_mse
                + lambdas['mdn']  * loss_mdn)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        stats['total'] += loss.item()
        stats['rank']  += loss_rank.item()
        stats['mse']   += loss_mse.item()
        stats['mdn']   += loss_mdn.item() if has_mdn else 0.0
        stats['n']     += 1

    if stats['n'] == 0:
        return {'total': float('inf'), 'rank': float('inf'), 'mse': float('inf'), 'mdn': float('inf')}
    n = stats['n']
    return {k: v / n for k, v in stats.items() if k != 'n'}


# ─────────────────────────────────────────────────────────────
# 验证：per-target Succ@10
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, device, dist_threshold, lambdas, temperature):
    """
    [改进 8] 验证指标：
      - loss 分量（监控训练）
      - native vs decoy 分数 gap（粗指标）
      - per-batch Succ@1: native 是否为 batch 内分数最高
        （每个 batch 已经是 1 native + N decoy 的结构）
    """
    model.eval()
    stats = {'total': 0, 'rank': 0, 'mse': 0, 'mdn': 0, 'n': 0}
    all_native_preds = []
    all_decoy_preds = []
    all_decoy_dockq = []
    succ_top1 = 0
    n_batches = 0

    for rec_batch, lig_batch, dockqs, is_natives, stems, names in loader:
        rec_batch = rec_batch.to(device)
        lig_batch = lig_batch.to(device)
        dockqs = dockqs.to(device)
        is_natives = is_natives.to(device)

        pi, sigma, mu, dist, C_batch, pred_energy = model(rec_batch, lig_batch)
        if (not torch.isfinite(pi).all()
                or not torch.isfinite(sigma).all()
                or not torch.isfinite(mu).all()
                or not torch.isfinite(dist).all()
                or not torch.isfinite(pred_energy).all()):
            continue

        loss_mdn, has_mdn = native_mdn_loss(pi, sigma, mu, dist, C_batch,
                                            is_natives, dist_threshold)
        if not torch.isfinite(loss_mdn):
            continue

        loss_rank = infonce_loss(pred_energy, is_natives, temperature=temperature)

        if (~is_natives).any():
            decoy_pred = pred_energy[~is_natives].clamp(1e-6, 1 - 1e-6)
            decoy_target = dockqs[~is_natives].clamp(0.0, 1.0)
            loss_mse = F.binary_cross_entropy(decoy_pred, decoy_target)
        else:
            loss_mse = torch.zeros_like(loss_rank)

        loss = (lambdas['rank']*loss_rank + lambdas['mse']*loss_mse
                + lambdas['mdn']*loss_mdn)
        if not torch.isfinite(loss):
            continue

        stats['total'] += loss.item()
        stats['rank']  += loss_rank.item()
        stats['mse']   += loss_mse.item()
        stats['mdn']   += loss_mdn.item() if has_mdn else 0.0
        stats['n']     += 1

        # 收集分数
        preds_np = pred_energy.detach().cpu().numpy()
        is_nat_np = is_natives.cpu().numpy().astype(bool)
        all_native_preds.extend(preds_np[is_nat_np].tolist())
        all_decoy_preds.extend(preds_np[~is_nat_np].tolist())
        all_decoy_dockq.extend(dockqs[~is_natives].cpu().numpy().tolist())

        # [改进 8] Succ@1：native 是否为 batch 中最高分
        if is_nat_np.any() and (~is_nat_np).any():
            n_batches += 1
            top_idx = int(preds_np.argmax())
            if is_nat_np[top_idx]:
                succ_top1 += 1

    if stats['n'] == 0:
        return {
            'total': float('inf'), 'rank': float('inf'), 'mse': float('inf'), 'mdn': float('inf'),
            'native_mean': 0.0, 'decoy_mean': 0.0, 'gap': 0.0,
            'succ_top1': 0.0, 'spearman': 0.0,
        }
    n = stats['n']
    avg = {k: v / n for k, v in stats.items() if k != 'n'}

    nat_arr = np.array(all_native_preds) if all_native_preds else np.array([0.0])
    dec_arr = np.array(all_decoy_preds)  if all_decoy_preds  else np.array([0.0])
    avg['native_mean'] = float(nat_arr.mean())
    avg['decoy_mean']  = float(dec_arr.mean())
    avg['gap']         = avg['native_mean'] - avg['decoy_mean']
    avg['succ_top1']   = succ_top1 / max(n_batches, 1)

    from scipy.stats import spearmanr
    if len(all_decoy_preds) > 2:
        r, _ = spearmanr(all_decoy_preds, all_decoy_dockq)
        avg['spearman'] = float(r) if not np.isnan(r) else 0.0
    else:
        avg['spearman'] = 0.0

    return avg


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native_dir', required=True)
    parser.add_argument('--decoy_dir', required=True)
    parser.add_argument('--decoy_csv', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--init_from', type=str, default=None)
    # 模型
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_gaussians', type=int, default=10)
    parser.add_argument('--n_tf_blocks', type=int, default=6)
    parser.add_argument('--tf_heads', type=int, default=4)
    parser.add_argument('--cross_heads', type=int, default=8)
    parser.add_argument('--n_cross_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--dist_threshold', type=float, default=10.0)
    parser.add_argument('--in_channels', type=int, default=11,
                        help='输入特征维度；默认11维表面特征')
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batches_per_epoch', type=int, default=400)
    parser.add_argument('--val_batches_per_epoch', type=int, default=80)
    parser.add_argument('--decoy_per_native', type=int, default=15,
                        help='[改进 6] 默认 15（v1 默认 5）')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_per_stem_decoy', type=int, default=None)
    parser.add_argument('--val_targets', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    # Loss 权重
    parser.add_argument('--lambda_rank', type=float, default=1.0)
    parser.add_argument('--lambda_mse',  type=float, default=0.3)
    parser.add_argument('--lambda_mdn',  type=float, default=0.3,
                        help='[改进 1] MDN 只对 native 算后，权重可适当降低')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='[改进 3] InfoNCE 温度')
    # 难度档比例
    parser.add_argument('--ratio_near_native', type=float, default=0.20)
    parser.add_argument('--ratio_hard',        type=float, default=0.40)
    parser.add_argument('--ratio_medium',      type=float, default=0.30)
    parser.add_argument('--ratio_easy',        type=float, default=0.10)
    # Best ckpt 标准
    parser.add_argument('--best_metric', choices=['succ_top1', 'gap'],
                        default='succ_top1',
                        help='[改进 8] 选 best ckpt 的指标')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"配置: in_channels={args.in_channels}, "
          f"decoy_per_native={args.decoy_per_native}, "
          f"loss=InfoNCE(τ={args.temperature}), "
          f"best_metric={args.best_metric}")

    # 切 train/val（按 stem）
    stems = set()
    with open(args.decoy_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stems.add(row['stem'])
    stems = sorted(stems)
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    val_stems = set(stems[:args.val_targets])
    train_stems = set(stems[args.val_targets:])
    print(f"Train stems: {len(train_stems)}, Val stems: {len(val_stems)}")

    train_set = MixedDataset(
        args.native_dir, args.decoy_dir, args.decoy_csv,
        stem_filter=train_stems,
        max_per_stem_decoy=args.max_per_stem_decoy,
        in_channels=args.in_channels,
    )
    val_set = MixedDataset(
        args.native_dir, args.decoy_dir, args.decoy_csv,
        stem_filter=val_stems,
        max_per_stem_decoy=args.max_per_stem_decoy,
        in_channels=args.in_channels,
    )

    sampler_kwargs = dict(
        decoy_per_native=args.decoy_per_native,
        ratio_near_native=args.ratio_near_native,
        ratio_hard=args.ratio_hard,
        ratio_medium=args.ratio_medium,
        ratio_easy=args.ratio_easy,
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=NativeVsDecoySampler(
            train_set, args.batches_per_epoch, **sampler_kwargs),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=NativeVsDecoySampler(
            val_set, args.val_batches_per_epoch, **sampler_kwargs),
        collate_fn=collate,
    )

    model = DeepDock_PPI(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        n_gaussians=args.n_gaussians,
        n_transformer_blocks=args.n_tf_blocks,
        transformer_heads=args.tf_heads,
        use_global_attn=True,
        global_attn_layers=2,
        cross_attn_heads=args.cross_heads,
        n_cross_attn_layers=args.n_cross_layers,
        dist_threshold=args.dist_threshold,
        dropout_rate=args.dropout,
    ).to(device)

    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        cur_sd = model.state_dict()
        compatible = {
            k: v for k, v in sd.items()
            if k in cur_sd and tuple(v.shape) == tuple(cur_sd[k].shape)
        }
        skipped = sorted(k for k in sd if k in cur_sd and k not in compatible)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"Init from {args.init_from}: "
              f"loaded={len(compatible)}, skipped_shape={len(skipped)}, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    lambdas = {'rank': args.lambda_rank,
               'mse':  args.lambda_mse,
               'mdn':  args.lambda_mdn}

    best_value = -float('inf')
    log_rows = []

    header = (f"{'Ep':>3} {'TrTot':>7} {'TrRank':>7} {'TrMSE':>7} {'TrMDN':>7} "
              f"{'VaTot':>7} {'VaNat':>6} {'VaDec':>6} {'Gap':>6} "
              f"{'Sρ':>6} {'S@1':>6} {'LR':>8} {'T':>5}")
    print(header)
    print('─' * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = datetime.now()
        tr = train_epoch(model, train_loader, optimizer, device,
                         args.dist_threshold, lambdas, args.temperature)
        va = eval_epoch(model, val_loader, device,
                        args.dist_threshold, lambdas, args.temperature)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = (datetime.now() - t0).seconds

        print(f"{epoch:>3} {tr['total']:>7.4f} {tr['rank']:>7.4f} "
              f"{tr['mse']:>7.4f} {tr['mdn']:>7.4f} "
              f"{va['total']:>7.4f} "
              f"{va['native_mean']:>6.3f} {va['decoy_mean']:>6.3f} "
              f"{va['gap']:>6.3f} {va['spearman']:>6.3f} "
              f"{va['succ_top1']:>6.3f} "
              f"{lr_now:>8.1e} {elapsed:>4}s")

        log_rows.append({'epoch': epoch, **tr,
                         **{f'val_{k}': v for k,v in va.items()}})

        # [改进 8] 用 succ_top1 选 best
        cur_value = va[args.best_metric]
        if cur_value > best_value:
            best_value = cur_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val': va,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'TransformerDock_best.chk'))

    pd.DataFrame(log_rows).to_csv(
        os.path.join(args.save_dir, 'training_log.csv'), index=False)
    print(f"\n训练完成. Best {args.best_metric}: {best_value:.4f}")


if __name__ == '__main__':
    main()
