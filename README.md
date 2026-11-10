# DPMG

DPMG 面向组织病理图像的目标基因空间表达预测。模型联合使用 H&E 图像块、空间坐标和邻域图，通过全基因预训练学习组织表征，再将编码器及目标基因对应的预测头参数迁移到目标基因微调任务。

微调阶段使用 **FeatureGate 特征门控**、**基因加权回归**和**基因关系先验约束**，同时更新编码器与预测头。默认目标面板包含 **GNAS、FASN、SCD、MYL12B 和 FN1**。目标基因及先验权重分别由 TXT、CSV 文件配置。

```text
H&E 图像块 + 空间坐标 + 邻域图
                 ↓
         全基因监督预训练
                 ↓
    迁移编码器与目标基因预测头参数
                 ↓
      FeatureGate + 目标基因微调
      基因加权回归 + 基因关系约束
                 ↓
         目标基因空间表达预测
```

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `train_dpmg_two_stage.py` | 主训练入口：全基因预训练与目标基因微调 |
| `train_dpmg_folds.py` | 按指定折次依次运行完整 DPMG 模型 |
| `train_dpmg.py` | 可选的直接训练入口，支持基因、基因子集或通路预测 |
| `dpmg_model.py` | DPMG 模型、FeatureGate 和训练损失 |
| `dpmg_data.py` | 数据读取、表达量处理和图像块提取 |
| `dpmg_evaluation.py` | 数据集构建和训练后评估 |
| `dpmg_attention.py`, `dpmg_graph_layers.py`, `dpmg_graph.py` | 注意力、图网络与邻域图构建 |
| `dpmg_count_losses.py` | 可选的计数分布损失 |
| `dpmg_data_utils.py`, `dpmg_targets.py`, `dpmg_priors.py` | 数据处理、目标定义和先验工具 |
| `recommended_gene_list_her2st.txt` | 默认目标基因列表 |
| `disease_gene_prior_her2st1.csv`, `disease_gene_relation_her2st1.csv` | 人工指定的基因权重与基因关系权重 |

## 环境与数据

使用已安装匹配 PyTorch / torchvision 的 Python 环境，然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

已检查的版本组合为 PyTorch 2.8.0、torchvision 0.23.0 和 PyTorch Lightning 2.6.0。`requirements.txt` 记录其余直接依赖版本。正式 GPU 训练需要安装适配设备的 CUDA 版 PyTorch。

将 HER2ST 数据及建模基因列表放在以下目录，或通过 `--data_root` 指定实际位置：

```text
data/
  her_hvg_cut_1000.npy
  her2st/data/
    ST-cnts/<切片>.tsv
    ST-imgs/<患者>/<切片>/<图像文件>
    ST-spotfiles/<切片>_selection.tsv
    ST-pat/lbl/<切片>_labeled_coordinates.tsv
```

`her_hvg_cut_1000.npy` 需使用与实验匹配的基因列表；当前主配置使用 785 个基因。表达量先在建模基因集上做文库大小归一化和 log 转换，再选取目标基因。

## 训练

在仓库根目录运行。以下命令均为单行，可用于 PowerShell 或 Bash；将 `--data_root ./data` 替换为实际数据目录。

单折训练：

```bash
python train_dpmg_folds.py --folds 0 --gpu 0 --data_root ./data --output_dir results/dpmg_main
```

完整 32 折训练：

```bash
python train_dpmg_folds.py --folds 0-31 --gpu 0 --data_root ./data --output_dir results/dpmg_main_32folds
```

预览每折训练命令：

```bash
python train_dpmg_folds.py --folds 0-31 --data_root ./data --dry_run
```

主配置使用随机种子 12000、学习率 1e-5、预训练 100 轮和微调 100 轮。预训练学习全基因表达表征；微调启用 FeatureGate、基因加权回归及权重为 0.001 的基因关系损失，普通相关性损失权重为 0。

直接调用两阶段入口并显式设置参数：

```bash
python train_dpmg_two_stage.py --data her2st --gpu 0 --fold 0 --seed 12000 --data_root ./data --gene_list_file recommended_gene_list_her2st.txt --pretrain_epochs 100 --finetune_epochs 100 --lr 1e-5 --dropout 0.2 --zinb 0 --nb F --bake 0 --corr_lambda 0 --use_feature_gate T --attn_reduction 4 --use_disease_prior T --disease_prior_file disease_gene_prior_her2st1.csv --disease_relation_file disease_gene_relation_her2st1.csv --prior_corr_lambda 0.001 --neighbor 4 --prune Grid --policy mean --tag 5-7-2-8-4-16-32 --save_model T --output_dir results/dpmg_main
```

可选的直接训练：

```bash
python train_dpmg.py --data her2st --fold 0 --gpu 0 --seed 12000 --epochs 200 --data_root ./data --task_type gene_subset --gene_list_file recommended_gene_list_her2st.txt --corr_lambda 0 --use_feature_gate T --use_disease_prior T --prior_corr_lambda 0.001 --save_model T --logger results/direct
```

## 结果与模型

两阶段训练结果保存在 `output_dir/fold_XX/`：

- `final_model.ckpt`：最终微调模型及构造参数、优化器状态。
- `run_config.json`：本次训练参数。
- `summary.csv`、`per_gene_pearson.csv`：总体与逐基因相关性指标。
- `pred_matrix.csv`、`gt_matrix.csv`：预测和真实表达矩阵。
- `gene_expression_diagnostics.csv`：表达量统计。

两阶段使用同一训练划分，训练结束后在留出的测试切片上评估。保存的是最终模型；测试结果不参与模型参数选择。`--save_model F` 可关闭模型保存。

加载模型：

```python
from dpmg_model import DPMG

model = DPMG.load_from_checkpoint(
    "results/dpmg_main/fold_00/final_model.ckpt", map_location="cpu"
)
model.eval()
```

直接训练的输出位于 TensorBoard 创建的 `version_*` 目录下，其中 `results_final/` 保存评估结果。
多折入口遇到非空的折输出目录会停止；中断后可用 `--folds` 指定剩余折，失败折使用新的输出目录。
数据、训练结果和模型权重由 `.gitignore` 排除。

## 致谢

编码器实现参考 [Hist2ST](https://github.com/biomed-AI/Hist2ST)，实验数据来自 [HER2ST](https://github.com/almaan/her2st)。使用相关代码和数据时，请遵守原项目许可并引用相应工作。
