"""
TransformerDock — 模型训练脚本

用法：
    python train.py                    # 合成数据测试
    python train.py --data_dir ../data # 真实数据训练
    python train.py --help             # 查看所有参数
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import transformerdock
from transformerdock.models import DeepDock_PPI, ppi_train_loss, ppi_score_diff
from transformerdock.utils.data import PPI_Dataset, ppi_collate


# ─────────────────────────────────────────────────────────────
# 合成数据（无真实数据时用于测试）
# ─────────────────────────────────────────────────────────────

def make_fake_surface(n_nodes=200, n_features=11, seed=None):
    """生成随机表面网格数据，用于验证模型能正常运行。"""
    from torch_geometric.data import Data
    if seed is not None:
        torch.manual_seed(seed)
    pos = torch.randn(n_nodes, 3) * 10
    x = torch.randn(n_nodes, n_features)
    k = 8
    src = torch.arange(n_nodes).repeat_interleave(k)
    dst = torch.randint(0, n_nodes, (n_nodes * k,))
    edge_index = torch.stack([src, dst], dim=0)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, pos=pos, edge_index=edge_index, batch=batch)


def test_forward_backward(model, device, in_channels):
    """验证前向传播和反向传播是否正常。"""
    print("── 合成数据测试 ──")
    rec = make_fake_surface(150, n_features=in_channels, seed=1).to(device)
    lig = make_fake_surface(100, n_features=in_channels, seed=2).to(device)

    # 前向传播
    model.eval()
    with torch.no_grad():
        pi, sigma, mu, dist, C_batch, _ = model(rec, lig)
    print(f"前向传播成功")
    print(f"  pi shape:     {pi.shape}")
    print(f"  dist shape:   {dist.shape}")
    print(f"  有效节点对数: {pi.shape[0]}")

    # 反向传播
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    pi, sigma, mu, dist, _, _ = model(rec, lig)
    loss = ppi_train_loss(pi, sigma, mu, dist, dist_threshold=10.0)
    loss.backward()
    optimizer.step()
    print(f"反向传播成功，Loss = {loss.item():.4f}")
    print()


# ─────────────────────────────────────────────────────────────
# 训练 / 验证函数
# ─────────────────────────────────────────────────────────────

def _random_rotation_matrix(device):
    """生成均匀随机 3x3 旋转矩阵（QR 分解法）。"""
    a = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(a)
    # 修正反射，保证 det=1
    d = torch.diag(torch.sign(torch.diagonal(r)))
    q = q @ d
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def perturb_ligand(lig_batch, t_min=5.0, t_max=15.0):
    """
    对 ligand batch 做随机刚体扰动（绕质心旋转 + 平移）。
    返回扰动后的 batch（克隆，不修改原 batch）。
    """
    perturbed = lig_batch.clone()
    pos = perturbed.pos
    device = pos.device
    batch = perturbed.batch if hasattr(perturbed, 'batch') and perturbed.batch is not None \
            else torch.zeros(pos.size(0), dtype=torch.long, device=device)

    n_graphs = int(batch.max().item()) + 1
    new_pos = pos.clone()
    for g in range(n_graphs):
        mask = (batch == g)
        sub = pos[mask]
        # 旋转(绕质心)
        center = sub.mean(dim=0, keepdim=True)
        R = _random_rotation_matrix(device)
        rotated = (sub - center) @ R.T + center
        # 平移
        direction = torch.randn(3, device=device)
        direction = direction / (direction.norm() + 1e-8)
        magnitude = (t_max - t_min) * torch.rand(1, device=device) + t_min
        translation = direction * magnitude
        new_pos[mask] = rotated + translation
    perturbed.pos = new_pos
    return perturbed


def train_epoch(model, loader, optimizer, device, dist_threshold,
                contrast_weight=0.0, contrast_margin=0.5,
                perturb_min=5.0, perturb_max=15.0, max_grad_norm=1.0,
                scaler=None, amp_dtype=None):
    """scaler/amp_dtype 非 None 时启用 torch.cuda.amp 混合精度。"""
    use_amp = scaler is not None and amp_dtype is not None
    model.train()
    total_loss = 0.0
    total_mdn = 0.0
    total_contrast = 0.0
    n_batches = 0
    skipped_nan = 0
    skipped_no_interface = 0
    for rec_batch, lig_batch, labels, pdb_ids in loader:
        rec_batch = rec_batch.to(device, non_blocking=True)
        lig_batch = lig_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # ── 1. native 上的 MDN loss ──
            pi, sigma, mu, dist, _, _ = model(rec_batch, lig_batch)
            # 提前检测模型输出
            if (not torch.isfinite(pi).all()
                    or not torch.isfinite(sigma).all()
                    or not torch.isfinite(mu).all()
                    or not torch.isfinite(dist).all()):
                print(f"  [警告] 模型输出NaN，跳过batch: {pdb_ids}")
                skipped_nan += 1
                continue
            if (dist.squeeze(1) <= dist_threshold).sum() == 0:
                skipped_no_interface += 1
                continue
            loss_mdn = ppi_train_loss(pi, sigma, mu, dist, dist_threshold=dist_threshold)
        if not torch.isfinite(loss_mdn):
            print(f"  [警告] 跳过 NaN/Inf MDN loss（batch: {pdb_ids}）")
            skipped_nan += 1
            continue

        loss = loss_mdn
        loss_c_val = 0.0

        # ── 2. contrastive：扰动 ligand 后,评分应该比 native 低 ──
        if contrast_weight > 0:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                score_native = ppi_score_diff(pi, sigma, mu, dist, dist_threshold)
                lig_perturbed = perturb_ligand(lig_batch, t_min=perturb_min, t_max=perturb_max)
                pi_p, sigma_p, mu_p, dist_p, _, _ = model(rec_batch, lig_perturbed)
                if (torch.isfinite(pi_p).all()
                        and torch.isfinite(sigma_p).all()
                        and torch.isfinite(mu_p).all()
                        and torch.isfinite(dist_p).all()):
                    score_perturbed = ppi_score_diff(pi_p, sigma_p, mu_p, dist_p, dist_threshold)
                    loss_contrast = torch.relu(score_perturbed - score_native + contrast_margin)
                else:
                    loss_contrast = torch.full((), float('nan'), device=device)
            if torch.isfinite(loss_contrast):
                loss = loss + contrast_weight * loss_contrast
                loss_c_val = loss_contrast.item()

        if use_amp:
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        total_loss += loss.item()
        total_mdn += loss_mdn.item()
        total_contrast += loss_c_val
        n_batches += 1
    if n_batches == 0:
        print(f"  [警告] 本轮训练没有有效 batch（NaN={skipped_nan}, no_interface={skipped_no_interface}）")
        return float('inf'), float('inf'), 0.0
    if skipped_nan or skipped_no_interface:
        print(f"  [提示] 训练跳过 batch: NaN/Inf={skipped_nan}, no_interface={skipped_no_interface}")
    nb = n_batches
    return total_loss / nb, total_mdn / nb, total_contrast / nb


@torch.no_grad()
def eval_epoch(model, loader, device, dist_threshold, amp_dtype=None):
    """amp_dtype 非 None 时 forward 走 autocast（推理本就无 backward, 不需要 scaler）。"""
    use_amp = amp_dtype is not None
    model.eval()
    total_loss = 0.0
    n_batches = 0
    skipped_nan = 0
    skipped_no_interface = 0
    for rec_batch, lig_batch, labels, pdb_ids in loader:
        rec_batch = rec_batch.to(device, non_blocking=True)
        lig_batch = lig_batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            pi, sigma, mu, dist, _, _ = model(rec_batch, lig_batch)
            # 检测模型输出NaN
            if (not torch.isfinite(pi).all()
                    or not torch.isfinite(sigma).all()
                    or not torch.isfinite(mu).all()
                    or not torch.isfinite(dist).all()):
                skipped_nan += 1
                continue
            if (dist.squeeze(1) <= dist_threshold).sum() == 0:
                skipped_no_interface += 1
                continue
            loss = ppi_train_loss(pi, sigma, mu, dist, dist_threshold=dist_threshold)
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        else:
            skipped_nan += 1
    if n_batches == 0:
        print(f"  [警告] 验证没有有效 batch（NaN/Inf={skipped_nan}, no_interface={skipped_no_interface}），返回 inf")
        return float('inf')
    if skipped_nan or skipped_no_interface:
        print(f"  [提示] 验证跳过 batch: NaN/Inf={skipped_nan}, no_interface={skipped_no_interface}")
    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TransformerDock 训练脚本')
    parser.add_argument('--data_dir',       type=str,   default=None,
                        help='数据目录（含 .ply 文件和 pairs.csv）')
    parser.add_argument('--pairs_csv',      type=str,   default=None,
                        help='可选 pairs.csv 路径；默认使用 data_dir/pairs.csv')
    parser.add_argument('--save_dir',       type=str,   default='../Trained_models',
                        help='模型保存目录')
    parser.add_argument('--hidden_dim',     type=int,   default=128)
    parser.add_argument('--n_gaussians',    type=int,   default=10)
    parser.add_argument('--n_tf_blocks',    type=int,   default=6,
                        help='TransformerResBlock 层数')
    parser.add_argument('--tf_heads',       type=int,   default=4,
                        help='TransformerResBlock 注意力头数')
    parser.add_argument('--cross_heads',    type=int,   default=8,
                        help='Cross-Attention 注意力头数')
    parser.add_argument('--n_cross_layers', type=int,   default=2,
                        help='Cross-Attention 层数')
    parser.add_argument('--dropout',        type=float, default=0.15)
    parser.add_argument('--dist_threshold', type=float, default=10.0,
                        help='MDN 距离阈值（Å）')
    parser.add_argument('--in_channels',    type=int,   default=11,
                        help='输入特征维度；默认11维表面特征')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--weight_decay',   type=float, default=1e-5)
    parser.add_argument('--save_every',     type=int,   default=10)
    parser.add_argument('--resume',         type=str,   default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--init_from',      type=str,   default=None,
                        help='从已有 checkpoint 初始化兼容权重；形状不匹配的层会跳过')
    parser.add_argument('--test_only',      action='store_true',
                        help='只运行合成数据测试，不训练')
    parser.add_argument('--amp',            type=str, default='off',
                        choices=['off', 'fp16', 'bf16'],
                        help='混合精度模式：off=fp32 默认；fp16=GradScaler；bf16=直接 autocast 不需要 scaler')
    parser.add_argument('--num_workers',    type=int, default=4,
                        help='DataLoader 并行 worker 数')
    parser.add_argument('--max_pairs',      type=int, default=None,
                        help='最多使用多少个 pair 训练；用于快速验证 19 维特征')
    parser.add_argument('--contrast_weight', type=float, default=0.0,
                        help='contrastive 辅助损失权重（0=关闭）')
    parser.add_argument('--contrast_margin', type=float, default=0.5,
                        help='contrastive margin')
    parser.add_argument('--perturb_min',    type=float, default=5.0,
                        help='扰动平移下限（Å）')
    parser.add_argument('--perturb_max',    type=float, default=15.0,
                        help='扰动平移上限（Å）')
    args = parser.parse_args()

    # ── 设备 ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"TransformerDock v{transformerdock.__version__}")
    print(f"PyTorch {torch.__version__} | 设备: {device}")
    print()

    # ── 模型 ──
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}  in_channels={args.in_channels}")

    # ── 合成数据测试 ──
    test_forward_backward(model, device, args.in_channels)

    if args.test_only:
        print("测试完成（--test_only 模式）")
        return

    # ── 数据集 ──
    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("[跳过训练] 未指定数据目录，请用 --data_dir 指定。")
        print("示例：python train.py --data_dir ../data")
        return

    pairs_csv = args.pairs_csv or os.path.join(args.data_dir, 'pairs.csv')
    if not os.path.exists(pairs_csv):
        print(f"[错误] 未找到 {pairs_csv}")
        print("请创建 pairs.csv，格式：name,label")
        return

    dataset = PPI_Dataset(args.data_dir, pairs_csv=pairs_csv, in_channels=args.in_channels)
    n = len(dataset)
    # 按固定 seed 打乱 pairs 顺序，避免按字母序切分 train/test 导致分布偏置
    import random as _random
    _rng = _random.Random(42)
    _shuffled = list(dataset.pairs)
    _rng.shuffle(_shuffled)
    if args.max_pairs is not None and args.max_pairs > 0:
        _shuffled = _shuffled[:args.max_pairs]
    dataset.pairs = _shuffled
    n = len(dataset)
    n_train = int(n * 0.9)
    train_set = dataset[:n_train]
    test_set  = dataset[n_train:]
    print(f"数据集: {n} 个复合物（训练 {len(train_set)} / 测试 {len(test_set)}）")

    loader_train = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, collate_fn=ppi_collate,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    loader_test = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, collate_fn=ppi_collate,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # ── 优化器 ──
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── 恢复训练 ──
    start_epoch = 1
    losses = []
    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        cur_sd = model.state_dict()
        compatible = {
            k: v for k, v in sd.items()
            if k in cur_sd and tuple(v.shape) == tuple(cur_sd[k].shape)
        }
        skipped = sorted(k for k in sd if k in cur_sd and k not in compatible)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"从 {args.init_from} 初始化: loaded={len(compatible)}, "
              f"skipped_shape={len(skipped)}, missing={len(missing)}, "
              f"unexpected={len(unexpected)}")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"从 epoch {ckpt['epoch']} 恢复训练")

    # ── 保存目录 ──
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 混合精度 ──
    amp_dtype, scaler = None, None
    if args.amp != 'off' and device == 'cuda':
        amp_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
        # fp16 需要 GradScaler；bf16 数值范围足够，传 enabled=False 的 scaler 走纯 autocast 路径
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == 'fp16'))
        print(f"混合精度: {args.amp} (dtype={amp_dtype}, scaler={'on' if scaler.is_enabled() else 'off'})")

    # ── 训练循环 ──
    best_loss = float('inf')
    print(f"\n开始训练（epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}）")
    if args.contrast_weight > 0:
        print(f"Contrastive: weight={args.contrast_weight}, margin={args.contrast_margin}, "
              f"perturb=[{args.perturb_min},{args.perturb_max}]Å")
    print(f"{'Epoch':>6}  {'Train':>10}  {'MDN':>10}  {'Cont':>10}  {'Test':>10}  {'LR':>10}  {'Time':>8}")
    print("─" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = datetime.now()
        train_loss, train_mdn, train_cont = train_epoch(
            model, loader_train, optimizer, device, args.dist_threshold,
            contrast_weight=args.contrast_weight,
            contrast_margin=args.contrast_margin,
            perturb_min=args.perturb_min,
            perturb_max=args.perturb_max,
            scaler=scaler, amp_dtype=amp_dtype,
        )
        test_loss  = eval_epoch(model, loader_test, device, args.dist_threshold,
                                amp_dtype=amp_dtype)
        scheduler.step()
        elapsed = (datetime.now() - t0).seconds
        lr_now = optimizer.param_groups[0]['lr']

        losses.append([train_loss, train_mdn, train_cont, test_loss])
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_mdn:>10.4f}  {train_cont:>10.4f}  "
              f"{test_loss:>10.4f}  {lr_now:>10.2e}  {elapsed:>6}s")

        # 保存最优模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'TransformerDock_best.chk'))

        # 定期保存
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, os.path.join(args.save_dir, f'TransformerDock_epoch_{epoch:03d}.chk'))

    # ── 保存 loss 曲线 ──
    df = pd.DataFrame(losses, columns=['train_loss', 'train_mdn', 'train_contrast', 'test_loss'])
    csv_path = os.path.join(args.save_dir, 'training_loss.csv')
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(ax=ax, title='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MDN Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    plt.close()

    print(f"\n训练完成！最优测试 Loss: {best_loss:.4f}")
    print(f"模型保存至: {args.save_dir}")
    print(f"Loss 曲线: {csv_path}")


if __name__ == '__main__':
    main()
