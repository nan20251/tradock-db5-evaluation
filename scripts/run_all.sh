#!/bin/bash
# TraDock 一键全流程运行
# 适配：RTX 4090D (24GB) + 60GB 内存 + 18核 CPU + CUDA 12.6
# 预计总耗时：~18-24 小时
#
# 用法：
#   nohup bash run_all.sh > /root/autodl-tmp/run_all.log 2>&1 &
#   tail -f /root/autodl-tmp/run_all.log

set -e
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4

echo "=========================================="
echo "TraDock 全流程开始: $(date)"
echo "=========================================="

run_optional_step() {
    local script="$1"
    if [ -f "scripts/$script" ]; then
        bash "scripts/$script"
    elif [ -f "$script" ]; then
        bash "$script"
    else
        echo "[跳过] 未找到 $script"
    fi
}

run_optional_step run_step0_step1.sh
run_optional_step run_step1_prep_full.sh
bash scripts/run_step2_pretrain.sh
run_optional_step run_step3_step4.sh
run_optional_step run_step5_decoy_surface.sh
run_optional_step run_step6_finetune.sh
bash scripts/run_step7_eval.sh

echo "=========================================="
echo "TraDock 全流程完成: $(date)"
echo "=========================================="
