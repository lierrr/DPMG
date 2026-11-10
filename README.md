# DPMG 训练代码

基于 Hist2ST 的全基因预训练与目标基因微调。默认目标基因为 GNAS、FASN、SCD、MYL12B、FN1。
此目录只包含训练、数据读取、模型和训练后评估所需代码。

## 文件

| 文件 | 用途 |
| --- | --- |
| `HIST2ST_pretrain_finetune.py` | 主训练入口：全基因预训练 → 五基因微调 |
| `run_train_folds.py` | 按折顺序运行完整 DPMG 模型，无消融组合 |
| `HIST2ST_train.py` | 可选：从头直接训练基因、基因子集或通路预测模型 |
| `HIST2ST.py` | 模型、FeatureGate 和损失函数 |
| `dataset.py`, `predict.py` | 数据读取及训练后测试；predict.py 是训练依赖，需保留 |
| `gcn.py`, `transformer.py`, `graph_construction.py`, `NB_module.py` | 网络组件 |
| `utils.py`, `pathway_utils.py`, `disease_prior_utils.py` | 数据及先验工具；均为训练依赖 |
| `recommended_gene_list_her2st.txt`, 两个 `disease_*.csv` | 目标基因和人工指定的先验权重 |

## 环境与数据

本机可用环境为 `D:\Anaconda\envs\hist2st\python.exe`。默认 `python` 指向 Anaconda 基础环境，缺少训练依赖。
检查时 hist2st 使用 PyTorch 2.8.0+cpu、torchvision 0.23.0、Lightning 2.6.0；当前无法使用 CUDA。
`requirements.txt` 记录本机直接依赖版本，不是完整的跨平台环境锁文件。新环境先按目标设备安装匹配的 PyTorch 与 torchvision，再运行 `python -m pip install -r requirements.txt`。

`--data_root` 指向以下目录，不需要把数据复制进代码目录：

```text
data_root/
  her_hvg_cut_1000.npy
  her2st/data/
    ST-cnts/<切片>.tsv
    ST-imgs/<患者>/<切片>/<图像文件>
    ST-spotfiles/<切片>_selection.tsv
    ST-pat/lbl/<切片>_labeled_coordinates.tsv
```

本机数据目录为 `D:\Hist2ST-main_gene_subset_pathway_pl2_ready\Hist2ST-main\data`。
全基因数以 npy 内容为准，不根据文件名推断。表达量先在全基因集上进行文库大小归一化及 log 转换，再取目标基因。

## 训练指令（Windows PowerShell）

进入本目录，激活已有环境：

```powershell
Set-Location 'D:\Hist2ST-main_gene_subset_pathway_pl2_ready\新建文件夹'
conda activate hist2st
```

主实验，单折（fold 0）：

```powershell
python run_train_folds.py --folds 0 --gpu 0 --data_root '../Hist2ST-main/data' --output_dir results/dpmg_main
```

完整 32 折，依次运行相同模型：

```powershell
python run_train_folds.py --folds 0-31 --gpu 0 --data_root '../Hist2ST-main/data' --output_dir results/dpmg_main_32folds
```

只检查将执行的命令，不加载数据、不训练：

```powershell
python run_train_folds.py --folds 0-31 --data_root '../Hist2ST-main/data' --dry_run
```

如果 `conda activate` 不可用，将上述 `python` 换成 `& 'D:\Anaconda\envs\hist2st\python.exe'`。
在 Linux 服务器上进入代码目录，使用已装好依赖的 `python`，将 `--data_root` 改为服务器实际数据目录。

主实验入口的等价完整参数如下（单行可在 PowerShell 或 Bash 执行）：

```text
python HIST2ST_pretrain_finetune.py --data her2st --gpu 0 --fold 0 --seed 12000 --data_root ../Hist2ST-main/data --gene_list_file recommended_gene_list_her2st.txt --pretrain_epochs 100 --finetune_epochs 100 --lr 1e-5 --dropout 0.2 --zinb 0 --nb F --bake 0 --corr_lambda 0 --use_feature_gate T --attn_reduction 4 --use_disease_prior T --disease_prior_file disease_gene_prior_her2st1.csv --disease_relation_file disease_gene_relation_her2st1.csv --prior_corr_lambda 0.001 --neighbor 4 --prune Grid --policy mean --tag 5-7-2-8-4-16-32 --save_model T --output_dir results/dpmg_main
```

主配置：预训练 100 轮、微调 100 轮，学习率 1e-5，seed=12000。预训练关闭 FeatureGate、相关性损失及疾病先验；微调启用 FeatureGate、基因加权回归及权重为 0.001 的基因关系损失。普通相关性损失权重为 0。两阶段都只使用训练切片，最后在留出的测试切片评估，无验证集选择最佳权重。

可选直接训练（不经过全基因预训练，因此与主实验是不同训练流程）：

```text
python HIST2ST_train.py --data her2st --fold 0 --gpu 0 --seed 12000 --epochs 200 --data_root ../Hist2ST-main/data --task_type gene_subset --gene_list_file recommended_gene_list_her2st.txt --corr_lambda 0 --use_feature_gate T --use_disease_prior T --prior_corr_lambda 0.001 --save_model T --logger results/direct
```

## 划分和输出

HER2ST 固定排除 A5、F1、F2、F3；fold 0–31 对应：

```text
A1 A2 A3 A4 A6 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6
D1 D2 D3 D4 D5 D6 E1 E2 E3 G1 G2 G3 H1 H2 H3
```

每折训练 31 张切片、测试 1 张；预训练和微调使用同一划分。这是按切片留一，不是按患者留一。
`HER2ST_SPLIT_MODE` 环境变量不会改变此划分。

两阶段训练结果位于 `output_dir/fold_XX/`，包括 `final_model.ckpt`、`run_config.json`、`summary.csv`、`per_gene_pearson.csv`、预测和真值矩阵、表达诊断。
默认保存最终微调模型；`--save_model F` 可关闭保存。checkpoint 为包含模型参数、构造参数和优化器状态的 Lightning 格式。测试结果是训练结束的评估，不参与参数选择。
直接训练的输出位于 TensorBoard 创建的 `version_*` 目录下，指标在 `results_final/`，checkpoint 在对应版本目录下。

多折入口遇到非空折输出目录会停止，避免覆盖；中断后用 `--folds` 指定尚未运行的折，失败折使用新的输出目录。它不支持从中断的 epoch 恢复。

数据、训练结果和模型权重已通过 `.gitignore` 排除，适合后续上传源代码。

## 来源

模型基于 [Hist2ST](https://github.com/biomed-AI/Hist2ST)，数据来自 [HER2ST](https://github.com/almaan/her2st)。使用代码与数据时请遵守原项目许可并引用原作者工作。
