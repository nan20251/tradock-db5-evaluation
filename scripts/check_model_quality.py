#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check checkpoint metadata and run a small inference smoke test."""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformerdock.models import DeepDock_PPI
from transformerdock.utils.data import prepare_complex


DEFAULT_DATA_DIR = os.environ.get('DIPS_SURFACES', '/root/autodl-tmp/dips_with_sasa_full')
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'Trained_models/pretrain_with_sasa/TransformerDock_best.chk'
DEFAULT_LOG = PROJECT_ROOT / 'Trained_models/pretrain_sasa.log'


def parse_args():
    parser = argparse.ArgumentParser(description='检查训练好的模型质量')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                        help='DIPS surfaces 目录，默认读取 DIPS_SURFACES 或 AutoDL 默认目录')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT),
                        help='模型 checkpoint')
    parser.add_argument('--log', default=str(DEFAULT_LOG),
                        help='训练日志，可不存在')
    parser.add_argument('--limit', type=int, default=20, help='最多抽样多少个 receptor/ligand 对')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = DeepDock_PPI(
        in_channels=11,
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
    return model, ckpt


def print_log_summary(log_path):
    print("\n2. 训练过程 NaN/跳过统计:")
    print("-" * 60)
    if not os.path.exists(log_path):
        print(f"  [跳过] 未找到训练日志: {log_path}")
        return

    with open(log_path, 'r', errors='ignore') as f:
        log_content = f.read()
    nan_warnings = (
        log_content.count('模型输出NaN')
        + log_content.count('NaN/Inf MDN loss')
        + log_content.count('NaN/Inf=')
    )
    no_interface = log_content.count('no_interface=')
    print(f"  NaN/Inf 相关记录: {nan_warnings}")
    print(f"  no_interface 相关记录: {no_interface}")


def collect_pairs(data_dir, limit):
    rec_files = sorted(glob.glob(os.path.join(data_dir, '*_receptor.ply')))
    pairs = []
    for rec_file in rec_files:
        lig_file = rec_file.replace('_receptor.ply', '_ligand.ply')
        if os.path.exists(lig_file):
            name = os.path.basename(rec_file).replace('_receptor.ply', '')
            pairs.append((name, rec_file, lig_file))
        if len(pairs) >= limit:
            break
    return pairs


@torch.no_grad()
def run_inference_check(model, device, data_dir, limit):
    print("\n3. 测试模型推理能力:")
    print("-" * 60)

    pairs = collect_pairs(data_dir, limit)
    if not pairs:
        print(f"  [跳过] 未找到 receptor/ligand PLY 对: {data_dir}")
        print("  AutoDL 上请设置 DIPS_SURFACES=/path/to/surfaces。")
        return None

    normal_count = 0
    bad_count = 0
    print(f"\n  测试 {len(pairs)} 个样本:")
    for i, (name, rec_file, lig_file) in enumerate(pairs, 1):
        try:
            rec, lig = prepare_complex(rec_file, lig_file, in_channels=11)
            rec = rec.to(device)
            lig = lig.to(device)
            pi, sigma, mu, dist, _, pred_energy = model(rec, lig)
            has_bad = (
                not torch.isfinite(pi).all()
                or not torch.isfinite(mu).all()
                or not torch.isfinite(sigma).all()
                or not torch.isfinite(dist).all()
                or not torch.isfinite(pred_energy).all()
            )
            if has_bad:
                print(f"    {i:2d}. {name:24s} - [错误] NaN/Inf")
                bad_count += 1
            else:
                print(f"    {i:2d}. {name:24s} - [OK]")
                normal_count += 1
        except Exception as exc:
            print(f"    {i:2d}. {name:24s} - [错误] {exc}")
            bad_count += 1

    print()
    total = max(1, len(pairs))
    print(f"  正常样本: {normal_count}/{len(pairs)} ({normal_count / total * 100:.1f}%)")
    print(f"  异常样本: {bad_count}/{len(pairs)} ({bad_count / total * 100:.1f}%)")
    return normal_count / total


def main():
    args = parse_args()

    print("=" * 60)
    print("检查训练好的模型质量")
    print("=" * 60)
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"数据目录: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    print("\n1. 最佳模型信息:")
    print("-" * 60)
    if not os.path.exists(args.checkpoint):
        print(f"  [错误] 未找到 checkpoint: {args.checkpoint}")
        return 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_model(args.checkpoint, device)
    print(f"  设备: {device}")
    print(f"  epoch: {ckpt.get('epoch', '?')}")
    loss = ckpt.get('loss', None)
    if loss is not None:
        print(f"  loss: {float(loss):.4f}")
    print("  [OK] 模型加载成功")

    print_log_summary(args.log)
    ratio = run_inference_check(model, device, args.data_dir, args.limit)

    print()
    print("=" * 60)
    print("结论:")
    print("=" * 60)
    if ratio is None:
        print("[跳过] 未做数据推理检查；请在 AutoDL 完整数据目录上运行。")
    elif ratio > 0.8:
        print("[OK] 抽样推理质量正常，可以继续 CAPRI 113 fast 评估。")
    elif ratio > 0.5:
        print("[警告] 抽样推理部分异常，建议先运行 check_data_range.py 和 check_nan_samples.py。")
    else:
        print("[错误] 抽样推理大部分异常，建议先检查数据特征和 checkpoint 是否匹配。")
        return 1
    print("=" * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
