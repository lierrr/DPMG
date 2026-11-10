import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from dpmg_priors import load_gene_weights, load_gene_relation_matrix
from dpmg_data_utils import *
from dpmg_model import *
from dpmg_evaluation import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='the id of gpu.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=12000, help='random seed.')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs.')
parser.add_argument('--name', type=str, default='dpmg', help='prefix name.')
parser.add_argument('--data', type=str, default='her2st', help='dataset name: {"her2st","cscc"}.')
parser.add_argument('--logger', type=str, default='results/direct', help='logger path.')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout.')

parser.add_argument('--bake', type=int, default=0, help='the number of augmented images.')
parser.add_argument('--lamb', type=float, default=0.5, help='the loss coef of self-distillation.')

parser.add_argument('--nb', type=str, default='F', help='zinb or nb loss.')
parser.add_argument('--zinb', type=float, default=0, help='the loss coef of zinb.')

parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge: {"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN.')
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')

parser.add_argument('--task_type', type=str, default='gene', choices=['gene', 'gene_subset', 'pathway'],
                    help='Predict all genes (original), a selected gene subset, or pathway scores.')
parser.add_argument('--pathway_file', type=str, default=None,
                    help='JSON/CSV/TSV/GMT pathway definition file. Required when task_type=pathway.')
parser.add_argument('--gene_list_file', type=str, default=None,
                    help='TXT/CSV/TSV/JSON gene list file. Required when task_type=gene_subset.')
parser.add_argument('--pathway_score_method', type=str, default='mean', choices=['mean', 'zscore_mean'],
                    help='How to aggregate genes into pathway scores.')

parser.add_argument('--corr_lambda', type=float, default=0.1,
                    help='weight of correlation loss.')
parser.add_argument('--use_feature_gate', type=str, default='T', choices=['T', 'F'],
                    help='whether to use lightweight feature gate attention.')
parser.add_argument('--attn_reduction', type=int, default=4,
                    help='reduction ratio for lightweight attention.')

parser.add_argument('--prior_corr_lambda', type=float, default=0.0,
                    help='weight of disease-aware gene-gene correlation prior loss.')
parser.add_argument('--use_disease_prior', type=str, default='F', choices=['T', 'F'],
                    help='whether to use disease-aware gene weights and gene-gene prior.')
parser.add_argument('--disease_prior_file', type=str, default='disease_gene_prior_her2st1.csv',
                    help='CSV file with columns gene,disease_weight.')
parser.add_argument('--disease_relation_file', type=str, default='disease_gene_relation_her2st1.csv',
                    help='CSV file with columns gene_a,gene_b,relation_weight.')
parser.add_argument('--save_model', type=str, default='T', choices=['T', 'F'],
                    help='whether to save final model weights.')

parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32',
                    help='hyper params: kernel-patch-depth1-depth2-depth3-heads-channel,'
                         'depth1-depth2-depth3 are the depth of Convmixer, Multi-head layer in Transformer, and GNN, respectively; '
                         'patch is the value of kernel_size and stride in the path embedding layer of Convmixer; '
                         'kernel is the kernel_size in the depthwise of Convmixer module; '
                         'heads are the number of attention heads in the Multi-head layer; '
                         'channel is the value of the input and output channel of depthwise and pointwise.')

parser.add_argument('--data_root', default='data', help='Directory containing her_hvg_cut_1000.npy and her2st/.')

args = parser.parse_args()
os.environ['DPMG_DATA_ROOT'] = os.path.abspath(args.data_root)

# -------------------------
# Reproducibility
# -------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')

kernel, patch, depth1, depth2, depth3, heads, channel = map(int, args.tag.split('-'))

if args.task_type == 'pathway' and not args.pathway_file:
    raise ValueError('--pathway_file is required when --task_type pathway')
if args.task_type == 'gene_subset' and not args.gene_list_file:
    raise ValueError('--gene_list_file is required when --task_type gene_subset')

ori_flag = args.task_type in ['gene', 'gene_subset']

# gene_subset 默认关闭 bake；zinb 保持可调
if args.task_type == 'gene':
    zinb_coef = args.zinb
    bake_num = args.bake
elif args.task_type == 'gene_subset':
    zinb_coef = args.zinb
    bake_num = 0
else:  # pathway
    zinb_coef = 0.0
    bake_num = 0
args.effective_zinb = zinb_coef
args.effective_bake = bake_num

# =========================
# 1. Load official train/test only
# =========================
trainset = pk_load(
    args.fold, 'train', False, args.data, neighs=args.neighbor, prune=args.prune,
    ori=ori_flag, target_mode=args.task_type,
    target_definitions_path=(args.pathway_file if args.task_type == 'pathway' else args.gene_list_file),
    target_score_method=args.pathway_score_method,
)

testset = pk_load(
    args.fold, 'test', False, args.data, neighs=args.neighbor, prune=args.prune,
    ori=ori_flag, target_mode=args.task_type,
    target_definitions_path=(args.pathway_file if args.task_type == 'pathway' else args.gene_list_file),
    target_score_method=args.pathway_score_method,
)

train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

# =========================
# 2. Prepare labels / target names
# =========================
label = None
if args.task_type == 'gene' and args.fold in [5, 11, 17, 23, 26, 30] and args.data == 'her2st':
    label = testset.label[testset.names[0]]

target_names = getattr(trainset, 'target_names', None)
if target_names is None:
    raise RuntimeError('Failed to resolve target names from the dataset.')

target_names = [str(x) for x in target_names]

test_target_names = getattr(testset, 'target_names', None)
if test_target_names is None:
    raise RuntimeError('Failed to resolve target names from the test dataset.')

test_target_names = [str(x) for x in test_target_names]

if target_names != test_target_names:
    raise RuntimeError(
        'Train/test target gene order mismatch!\n'
        f'Train targets: {target_names}\n'
        f'Test targets: {test_target_names}'
    )

num_targets = len(target_names)

# =========================
# 2.1 Optional disease-aware prior
# =========================
gene_weights = None
gene_relation_matrix = None

if args.use_disease_prior == 'T':
    if args.task_type != 'gene_subset':
        raise ValueError('--use_disease_prior T is currently recommended only for --task_type gene_subset')

    gene_weights = load_gene_weights(target_names, args.disease_prior_file)
    gene_relation_matrix = load_gene_relation_matrix(target_names, args.disease_relation_file)

    print('=' * 100)
    print('Disease-aware prior enabled.')
    print('Disease gene weights, normalized to mean=1:')
    for g, w in zip(target_names, gene_weights):
        print(f'{g}: {float(w):.6f}')
    print('Disease gene relation matrix:')
    print(gene_relation_matrix)
    print(f'prior_corr_lambda: {args.prior_corr_lambda}')
    print('=' * 100)

print('Gene order check passed.')
for idx, gene in enumerate(target_names):
    print(f'{idx + 1}: {gene}')

# =========================
# 3. Experiment naming
# =========================
if args.data == 'cscc':
    args.name += '_cscc'
if args.task_type == 'pathway':
    args.name += '_pathway'
elif args.task_type == 'gene_subset':
    args.name += '_gene_subset'

log_name = ''
if zinb_coef > 0:
    if args.nb == 'T':
        args.name += '_nb'
    else:
        args.name += '_zinb'
    log_name += f'-{zinb_coef}'
if bake_num > 0:
    args.name += '_bake'
    log_name += f'-{bake_num}-{args.lamb}'
if args.task_type == 'pathway':
    pathway_stub = os.path.splitext(os.path.basename(args.pathway_file))[0]
    log_name += f'-{pathway_stub}-{args.pathway_score_method}'
elif args.task_type == 'gene_subset':
    subset_stub = os.path.splitext(os.path.basename(args.gene_list_file))[0]
    log_name += f'-{subset_stub}'

if args.use_disease_prior == 'T':
    args.name += '_diseaseprior'
    log_name += f'-dprior-{args.prior_corr_lambda}'

log_name = f'{args.fold}-{args.name}-{args.tag}' + log_name + f'-{args.policy}-{args.neighbor}'

logger = TensorBoardLogger(args.logger, name=log_name)

os.makedirs(logger.log_dir, exist_ok=True)
with open(os.path.join(logger.log_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, indent=2, ensure_ascii=False)

gene_order_df = pd.DataFrame({
    'column_index': list(range(1, len(target_names) + 1)),
    'target_name': target_names
})
gene_order_df.to_csv(os.path.join(logger.log_dir, 'gene_order_check.csv'), index=False)

try:
    logger.log_hyperparams(vars(args))
except Exception:
    pass

print(log_name)
print(f'Task type: {args.task_type}')
print(f'Number of targets: {num_targets}')
print(f'Train size: {len(trainset)}')
print(f'Test size: {len(testset)}')
if args.task_type in ['pathway', 'gene_subset']:
    print('Targets:', target_names)

# =========================
# 4. Model
# =========================
model = DPMG(
    depth1=depth1, depth2=depth2, depth3=depth3,
    n_genes=num_targets, learning_rate=args.lr, label=label,
    kernel_size=kernel, patch_size=patch,
    heads=heads, channel=channel, dropout=args.dropout,
    zinb=zinb_coef, nb=args.nb == 'T',
    bake=bake_num, lamb=args.lamb,
    policy=args.policy, task_type=args.task_type, target_names=target_names,
    corr_lambda=args.corr_lambda,
    use_feature_gate=(args.use_feature_gate == 'T'),
    attn_reduction=args.attn_reduction,
    gene_weights=gene_weights,
    gene_relation_matrix=gene_relation_matrix,
    prior_corr_lambda=args.prior_corr_lambda,
)

pl_version_major = int(pl.__version__.split('.')[0])

# =========================
# 5. Trainer (no validation, no checkpoint selection)
# =========================
if torch.cuda.is_available():
    if pl_version_major >= 2:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[args.gpu],
            max_epochs=args.epochs,
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
    else:
        trainer = pl.Trainer(
            gpus=[args.gpu],
            max_epochs=args.epochs,
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
else:
    if pl_version_major >= 2:
        trainer = pl.Trainer(
            accelerator='cpu',
            devices=1,
            max_epochs=args.epochs,
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

# =========================
# 6. Train using official train only
# =========================
if pl_version_major >= 2:
    trainer.fit(model, train_dataloaders=train_loader)
else:
    trainer.fit(model, train_loader)

# =========================
# 7. Final evaluation on official test
# =========================
pred, gt = test(model, test_loader, f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
R = get_R(pred, gt)[0]
mean_r = np.nanmean(R)

print('Final Mean Pearson Correlation:', mean_r)

# =========================
# 8. Save final results
# =========================
result_dir = os.path.join(logger.log_dir, 'results_final')
os.makedirs(result_dir, exist_ok=True)

pearson_df = pd.DataFrame({
    'gene': target_names,
    'pearson': R
})
pearson_df.to_csv(os.path.join(result_dir, 'per_gene_pearson.csv'), index=False)
def concat_dataset_exp(dataset):
    arrs = []
    for name in dataset.names:
        arrs.append(np.asarray(dataset.exp_dict[name]))
    return np.concatenate(arrs, axis=0)

train_exp_all = concat_dataset_exp(trainset)
test_exp_all = concat_dataset_exp(testset)

if train_exp_all.shape[1] != len(target_names):
    raise RuntimeError(
        f'Train expression column number mismatch: '
        f'{train_exp_all.shape[1]} vs {len(target_names)}'
    )

if test_exp_all.shape[1] != len(target_names):
    raise RuntimeError(
        f'Test expression column number mismatch: '
        f'{test_exp_all.shape[1]} vs {len(target_names)}'
    )

diagnostic_df = pd.DataFrame({
    'gene': target_names,
    'pearson': R,
    'train_mean': train_exp_all.mean(axis=0),
    'train_std': train_exp_all.std(axis=0),
    'test_mean': test_exp_all.mean(axis=0),
    'test_std': test_exp_all.std(axis=0),
    'train_nonzero_ratio': (np.abs(train_exp_all) > 1e-8).mean(axis=0),
    'test_nonzero_ratio': (np.abs(test_exp_all) > 1e-8).mean(axis=0),
})

diagnostic_df['abs_train_test_mean_diff'] = np.abs(
    diagnostic_df['train_mean'] - diagnostic_df['test_mean']
)

diagnostic_df['test_std_too_low'] = diagnostic_df['test_std'] < 1e-4

diagnostic_df.to_csv(
    os.path.join(result_dir, 'gene_expression_diagnostics.csv'),
    index=False
)

print('Gene expression diagnostics saved to:',
      os.path.join(result_dir, 'gene_expression_diagnostics.csv'))
summary_df = pd.DataFrame({
    'metric': ['mean_pearson'],
    'value': [mean_r]
})
summary_df.to_csv(os.path.join(result_dir, 'summary.csv'), index=False)

pred_df = pd.DataFrame(pred.X, columns=target_names)
gt_df = pd.DataFrame(gt.X, columns=target_names)

if list(pred_df.columns) != list(gt_df.columns):
    raise RuntimeError(
        'pred_matrix and gt_matrix column order mismatch!\n'
        f'pred columns: {list(pred_df.columns)}\n'
        f'gt columns: {list(gt_df.columns)}'
    )

matrix_order_df = pd.DataFrame({
    'column_index': list(range(1, len(target_names) + 1)),
    'pred_column': list(pred_df.columns),
    'gt_column': list(gt_df.columns),
    'same': [a == b for a, b in zip(pred_df.columns, gt_df.columns)]
})
matrix_order_df.to_csv(os.path.join(result_dir, 'pred_gt_column_check.csv'), index=False)

pred_df.to_csv(os.path.join(result_dir, 'pred_matrix.csv'), index=False)
gt_df.to_csv(os.path.join(result_dir, 'gt_matrix.csv'), index=False)
print('Results saved to:', result_dir)

# =========================
# 9. Save final model weights
# =========================
if args.save_model == 'T':
    final_model_path = os.path.join(logger.log_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    print('Final model checkpoint saved to:', final_model_path)
else:
    print('Final model weights not saved. Set --save_model T if needed.')

# =========================
# 10. Cluster eval only for original full-gene task
# =========================
if args.task_type == 'gene' and label is not None:
    clus, ARI = cluster(pred, label)
    print('ARI:', ARI)
else:
    print('Cluster evaluation skipped for non-full-gene tasks.')