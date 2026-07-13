#!/bin/bash
# Step 2: 预训练（RTX 4090D 24GB，~10-12 小时）
# max_nodes=1500 限制每个表面最大节点数，避免 OOM

set -e
PROJECT_ROOT="${TRADOCK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DIPS_SURFACES="${DIPS_SURFACES:-/root/autodl-tmp/dips_with_sasa_full}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "examples/train.py" ]; then
    echo "[错误] 未找到 examples/train.py，项目目录不正确: $PROJECT_ROOT" >&2
    exit 1
fi
if [ ! -f "$DIPS_SURFACES/pairs.csv" ]; then
    echo "[错误] 未找到训练数据: $DIPS_SURFACES/pairs.csv" >&2
    echo "请设置 DIPS_SURFACES=/path/to/surfaces，或在 AutoDL 准备 /root/autodl-tmp/dips_with_sasa_full" >&2
    exit 1
fi

mkdir -p Trained_models results

PAIRS_CSV="${PAIRS_CSV:-results/dips_with_sasa_full.filtered_pairs.csv}"
python scripts/filter_bad_dips_pairs.py \
    --input "$DIPS_SURFACES/pairs.csv" \
    --output "$PAIRS_CSV" \
    --exclude 1u0c_A_B 1yk0_A_B \
    --exclude_file data/dips/exclude_capri.txt

python examples/train.py \
    --data_dir "$DIPS_SURFACES" \
    --pairs_csv "$PAIRS_CSV" \
    --save_dir Trained_models/pretrain_with_sasa \
    --epochs 30 \
    --batch_size 2 \
    --lr 1e-4 \
    --contrast_weight 0.0

echo "=== Step 2 预训练完成 ==="
echo "产物: Trained_models/pretrain_with_sasa/TransformerDock_best.chk"
