#!/bin/bash
# TraDock 一键流程运行
# 适配：RTX 4090D (24GB) + 60GB 内存 + 18核 CPU + CUDA 12.6
# 当前流程：11维 DIPS 预训练 + CAPRI 评估
#
# 用法：
#   nohup bash scripts/run_all.sh > /root/autodl-tmp/run_all.log 2>&1 &
#   tail -f /root/autodl-tmp/run_all.log

set -e
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4

echo "=========================================="
echo "TraDock 全流程开始: $(date)"
echo "=========================================="

bash scripts/run_step2_pretrain.sh
bash scripts/run_step7_eval.sh

echo "=========================================="
echo "TraDock 全流程完成: $(date)"
echo "=========================================="
