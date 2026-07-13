# TraDock 项目结构

## 目录布局

```
TraDock/
├── README.md                  # 快速开始和常用流程
├── PROJECT_GUIDE.md           # 当前项目约定和检查顺序
├── RUNBOOK.md                 # 远端运行记录和续跑命令
├── FAQ.md                     # 常见问题与解决方案
├── environment.yml            # Conda 环境定义（torch 2.4.1, cuda 12.1）
├── requirements.txt           # pip 依赖清单
├── .gitignore                 # Git 忽略规则
│
├── transformerdock/           # 核心 Python 包
│   ├── __init__.py
│   ├── models.py              # 模型定义（DeepDock_PPI 等）
│   ├── prepare_target/        # 靶蛋白预处理
│   │   ├── __init__.py
│   │   └── computeSurface.py  # 表面积计算（含 FreeSASA）
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   └── data.py            # 11维表面特征数据加载和预处理
│   ├── data/                  # 数据文件（可选）
│   └── models/                # 预训练模型权重存储
│
├── examples/                  # 示例和评估脚本
│   ├── train.py               # 主训练脚本
│   ├── train_native_vs_decoy_v2.py
│   ├── eval_capri_fast.py     # CAPRI 113 fast 评估
│   ├── eval_db5_paper_tradock.py
│   ├── surface_gen.py         # 表面特征生成示例
│   ├── extract_capri_sequences.py
│   ├── prep_dips.py           # DIPS 数据集准备
│   ├── prep_dockground_decoys.py
│   └── ... (更多脚本)
│
├── data/                      # 轻量数据集元数据和参考
│   ├── capri_exclude/         # CAPRI 排除列表
│   ├── capri113_single_targets/
│   └── dips/                  # DIPS 数据集
│       ├── exclude_capri.txt
│       └── metadata.csv
│
├── results/                   # 输出和日志
│   ├── capri_113_w1.log       # 训练或评估日志示例
│   └── ... (更多结果)
│
├── scripts/                   # 辅助脚本
│   ├── quick_check.py         # 环境自检脚本（Python）
│   ├── quick_check_remote.sh  # 远端环境检查（Bash）
│   ├── run_step2_pretrain.sh  # 11维 DIPS 预训练
│   ├── run_step7_eval.sh      # CAPRI 评估
│   ├── run_step8_eval_db5_paper.sh
│   ├── fix_pdb_elements.py    # PDB 元素列修复工具
│   └── install_pyg_remote.sh  # 远端 PyG 安装脚本
│
├── Trained_models/            # 保存的模型检查点
├── results/                   # 本地评估结果
├── results_remote_*           # 从远端同步的评估结果
├── backups/                   # 小范围代码备份
├── eval_packages/             # 打包导出的评估代码/结果
│
└── ... (其他辅助文件)
```

## 当前约定

- 活跃训练和评估路径使用 11 维表面特征。
- 19 维 pair-aware 实验入口和产物已从当前项目中移除。
- `Trained_models/pretrain_with_sasa/TransformerDock_best.chk` 是当前默认 checkpoint。
- DIPS 完整表面数据通常不放入仓库，通过 `DIPS_SURFACES` 指向。
- CAPRI 完整数据库通过 `CAPRI_DIR` 指向；本地只保留轻量元数据和结果。

## 关键模块说明

### `transformerdock/models.py`
包含模型类定义，如基于 Transformer 的 dock 预测器。

### `transformerdock/prepare_target/computeSurface.py`
使用 FreeSASA 库计算蛋白质表面积特征。警告注意：确保 PDB 文件包含正确的元素列（见 FAQ）。

### `transformerdock/utils/data.py`
数据加载和图构造工具，与 PyTorch Geometric 集成；默认对齐 11 维表面特征。

### `examples/`
训练和评估的入口脚本。`train.py` 是主要训练脚本；`eval_capri_fast.py` 用于评估模型在 CAPRI 数据集上的表现。

## 运行工作流

1. **准备环境**：
   ```bash
   conda env create -f environment.yml
   conda activate tradock
   # 然后按 README.md 中的 PyG 安装指南安装原生扩展
   ```

2. **自检**：
   ```bash
   python scripts/quick_check.py
   ```

3. **准备数据**（可选）：
   ```bash
   python examples/prep_dips.py
   # 或其他数据准备脚本
   ```

4. **训练**：
   ```bash
   DIPS_SURFACES=/path/to/dips_with_sasa_full bash scripts/run_step2_pretrain.sh
   ```

5. **评估**：
   ```bash
   CAPRI_DIR=/path/to/database \
   CHECKPOINT=Trained_models/pretrain_with_sasa/TransformerDock_best.chk \
   bash scripts/run_step7_eval.sh
   ```

## 常见问题

如遇到 PyG 导入错误、FreeSASA 警告等问题，请参考 `FAQ.md`。
