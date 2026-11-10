import os
import json
import inspect
import random
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from utils import *
from HIST2ST import *
from predict import *
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_target_names(dataset):
    target_names = getattr(dataset, "target_names", None)
    if target_names is not None:
        return [str(x) for x in target_names]

    gene_set = getattr(dataset, "gene_set", None)
    if gene_set is not None:
        return [str(x) for x in gene_set]

    raise RuntimeError("Cannot resolve target names from dataset.")


def build_model(model_kwargs):
    """
    兼容不同版本 HIST2ST.py：
    如果 Hist2ST 不支持某个参数，就自动忽略。
    """
    sig = inspect.signature(Hist2ST.__init__)
    valid = set(sig.parameters.keys())
    final_kwargs = {k: v for k, v in model_kwargs.items() if k in valid}
    ignored = sorted(set(model_kwargs.keys()) - set(final_kwargs.keys()))
    if ignored:
        print("[INFO] Ignored unsupported Hist2ST args:", ignored)
    return Hist2ST(**final_kwargs)


def make_trainer(max_epochs, gpu):
    pl_version_major = int(pl.__version__.split(".")[0])

    common = dict(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )

    if torch.cuda.is_available():
        if pl_version_major >= 2:
            return pl.Trainer(accelerator="gpu", devices=[gpu], **common)
        else:
            return pl.Trainer(gpus=[gpu], **common)
    else:
        if pl_version_major >= 2:
            return pl.Trainer(accelerator="cpu", devices=1, **common)
        else:
            return pl.Trainer(**common)


def copy_pretrained_weights(full_model, target_model, full_gene_names, target_gene_names):
    """
    Copy full-gene pretrained weights into the target-gene model.

    1. Copy encoder modules:
       patch_embedding / x_embed / y_embed / vit

    2. Copy gene_head:
       - LayerNorm copied directly
       - Linear layer rows copied according to target gene indices
    """

    # ========== 1. Copy encoder ==========
    modules = ["patch_embedding", "x_embed", "y_embed", "vit"]

    for name in modules:
        if hasattr(full_model, name) and hasattr(target_model, name):
            getattr(target_model, name).load_state_dict(
                getattr(full_model, name).state_dict()
            )
            print(f"[PRETRAIN INIT] Loaded encoder module: {name}")
        else:
            print(f"[WARNING] Missing module, skip: {name}")

    # ========== 2. Copy target gene head ==========
    full_gene_names = [str(x) for x in full_gene_names]
    target_gene_names = [str(x) for x in target_gene_names]

    missing = [g for g in target_gene_names if g not in full_gene_names]
    if missing:
        print("[WARNING] Missing target genes in full-gene head, skip head initialization:", missing)
        return

    target_indices = [full_gene_names.index(g) for g in target_gene_names]

    try:
        # gene_head = nn.Sequential(LayerNorm(dim), Linear(dim, n_genes))
        full_ln = full_model.gene_head[0]
        full_linear = full_model.gene_head[1]

        target_ln = target_model.gene_head[0]
        target_linear = target_model.gene_head[1]

        # copy LayerNorm
        target_ln.load_state_dict(full_ln.state_dict())

        # copy only target gene rows from full 785 head
        with torch.no_grad():
            target_linear.weight.copy_(full_linear.weight[target_indices, :])
            target_linear.bias.copy_(full_linear.bias[target_indices])

        print("[PRETRAIN INIT] Loaded target gene head rows:")
        for g, idx in zip(target_gene_names, target_indices):
            print(f"  {g} <- full785 head row {idx}")

    except Exception as e:
        print("[WARNING] Failed to copy target gene head rows.")
        print(repr(e))


def concat_dataset_exp(dataset):
    arrs = []
    for name in dataset.names:
        arrs.append(np.asarray(dataset.exp_dict[name]))
    return np.concatenate(arrs, axis=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12000)

    parser.add_argument("--data", type=str, default="her2st")
    parser.add_argument("--gene_list_file", type=str, default="recommended_gene_list_her2st.txt")

    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--finetune_epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--zinb", type=float, default=0.0)
    parser.add_argument("--nb", type=str, default="F")
    parser.add_argument("--bake", type=int, default=0)
    parser.add_argument("--lamb", type=float, default=0.5)

    parser.add_argument("--corr_lambda", type=float, default=0.0)
    parser.add_argument("--use_feature_gate", type=str, default="T", choices=["T", "F"])
    parser.add_argument("--attn_reduction", type=int, default=4)

    parser.add_argument("--use_disease_prior", type=str, default="T", choices=["T", "F"])
    parser.add_argument("--disease_prior_file", type=str, default="disease_gene_prior_her2st1.csv")
    parser.add_argument("--disease_relation_file", type=str, default="disease_gene_relation_her2st1.csv")
    parser.add_argument("--prior_corr_lambda", type=float, default=0.001)

    parser.add_argument("--prune", type=str, default="Grid")
    parser.add_argument("--policy", type=str, default="mean")
    parser.add_argument("--neighbor", type=int, default=4)
    parser.add_argument("--tag", type=str, default="5-7-2-8-4-16-32")

    parser.add_argument("--output_dir", type=str, default="pretrain_finetune_results")
    parser.add_argument("--name", type=str, default="full785_pretrain_finetune_5gene")

    parser.add_argument('--data_root', default='data', help='Directory containing her_hvg_cut_1000.npy and her2st/.')

    parser.add_argument("--save_model", choices=["T", "F"], default="T")

    args = parser.parse_args()
    os.environ['HIST2ST_DATA_ROOT'] = os.path.abspath(args.data_root)

    seed_everything(args.seed)

    kernel, patch, depth1, depth2, depth3, heads, channel = map(int, args.tag.split("-"))

    print("=" * 100)
    print("FULL785 PRETRAIN -> 5GENE FINETUNE")
    print("fold:", args.fold)
    print("gene_list_file:", args.gene_list_file)
    print("pretrain_epochs:", args.pretrain_epochs)
    print("finetune_epochs:", args.finetune_epochs)
    print("corr_lambda:", args.corr_lambda)
    print("use_feature_gate:", args.use_feature_gate)
    print("use_disease_prior:", args.use_disease_prior)
    print("prior_corr_lambda:", args.prior_corr_lambda)
    print("HER2ST_SPLIT_MODE:", "custom32" if args.data == "her2st" else "cscc")
    print("=" * 100)

    # =========================================================
    # 1. full785 pretrain dataset
    # =========================================================
    full_trainset = pk_load(
        args.fold,
        "train",
        False,
        args.data,
        neighs=args.neighbor,
        prune=args.prune,
        ori=True,
        target_mode="gene",
        target_definitions_path=None,
        target_score_method="mean",
    )

    full_train_loader = DataLoader(full_trainset, batch_size=1, num_workers=0, shuffle=True)

    full_target_names = get_target_names(full_trainset)
    full_n_genes = len(full_target_names)

    print("[FULL PRETRAIN] n_genes:", full_n_genes)

    full_model_kwargs = dict(
        depth1=depth1,
        depth2=depth2,
        depth3=depth3,
        n_genes=full_n_genes,
        learning_rate=args.lr,
        label=None,
        kernel_size=kernel,
        patch_size=patch,
        heads=heads,
        channel=channel,
        dropout=args.dropout,
        zinb=args.zinb,
        nb=args.nb == "T",
        bake=args.bake,
        lamb=args.lamb,
        policy=args.policy,
        task_type="gene",
        target_names=full_target_names,
        corr_lambda=0.0,
        use_feature_gate=False,
        attn_reduction=args.attn_reduction,
    )

    full_model = build_model(full_model_kwargs)

    pretrainer = make_trainer(args.pretrain_epochs, args.gpu)
    pretrainer.fit(full_model, train_dataloaders=full_train_loader)

    # =========================================================
    # 2. target 5gene finetune dataset
    # =========================================================
    target_trainset = pk_load(
        args.fold,
        "train",
        False,
        args.data,
        neighs=args.neighbor,
        prune=args.prune,
        ori=True,
        target_mode="gene_subset",
        target_definitions_path=args.gene_list_file,
        target_score_method="mean",
    )

    target_testset = pk_load(
        args.fold,
        "test",
        False,
        args.data,
        neighs=args.neighbor,
        prune=args.prune,
        ori=True,
        target_mode="gene_subset",
        target_definitions_path=args.gene_list_file,
        target_score_method="mean",
    )

    target_train_loader = DataLoader(target_trainset, batch_size=1, num_workers=0, shuffle=True)
    target_test_loader = DataLoader(target_testset, batch_size=1, num_workers=0, shuffle=False)

    target_names = get_target_names(target_trainset)
    test_target_names = get_target_names(target_testset)

    if target_names != test_target_names:
        raise RuntimeError(f"Target order mismatch: train={target_names}, test={test_target_names}")

    n_targets = len(target_names)

    print("[TARGET FINETUNE] target genes:", target_names)

    # =========================================================
    # 3. optional disease prior
    # =========================================================
    gene_weights = None
    gene_relation_matrix = None

    if args.use_disease_prior == "T":
        from disease_prior_utils import load_gene_weights, load_gene_relation_matrix
        gene_weights = load_gene_weights(target_names, args.disease_prior_file)
        gene_relation_matrix = load_gene_relation_matrix(target_names, args.disease_relation_file)
        print("[DISEASE PRIOR] enabled")
        print("[DISEASE PRIOR] gene_weights:", gene_weights)
        print("[DISEASE PRIOR] gene_relation_matrix:")
        print(gene_relation_matrix)

    target_model_kwargs = dict(
        depth1=depth1,
        depth2=depth2,
        depth3=depth3,
        n_genes=n_targets,
        learning_rate=args.lr,
        label=None,
        kernel_size=kernel,
        patch_size=patch,
        heads=heads,
        channel=channel,
        dropout=args.dropout,
        zinb=0.0,
        nb=False,
        bake=0,
        lamb=args.lamb,
        policy=args.policy,
        task_type="gene_subset",
        target_names=target_names,
        corr_lambda=args.corr_lambda,
        use_feature_gate=args.use_feature_gate == "T",
        attn_reduction=args.attn_reduction,
        gene_weights=gene_weights,
        gene_relation_matrix=gene_relation_matrix,
        prior_corr_lambda=args.prior_corr_lambda,
    )

    target_model = build_model(target_model_kwargs)

    # copy encoder from full785 model
    copy_pretrained_weights(
        full_model,
        target_model,
        full_target_names,
        target_names
    )

    # release full model before finetuning
    del pretrainer, full_train_loader, full_trainset, full_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetrainer = make_trainer(args.finetune_epochs, args.gpu)
    finetrainer.fit(target_model, train_dataloaders=target_train_loader)

    # =========================================================
    # 4. final test
    # =========================================================
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    pred, gt = test(target_model, target_test_loader, device)

    R = np.asarray(get_R(pred, gt)[0], dtype=float)
    mean_r = float(np.nanmean(R))
    median_r = float(np.nanmedian(R))

    print("=" * 100)
    print("Final Mean Pearson Correlation:", mean_r)
    print("Final Median Pearson Correlation:", median_r)
    for g, r in zip(target_names, R):
        print(f"{g}: pearson = {r}")
    print("=" * 100)

    # =========================================================
    # 5. Save final checkpoint and evaluation outputs
    # =========================================================
    out_dir = os.path.join(args.output_dir, f"fold_{args.fold:02d}")
    os.makedirs(out_dir, exist_ok=True)

    summary_df = pd.DataFrame([{
        "fold": args.fold,
        "test_slice": target_testset.names[0],
        "mean_pearson": mean_r,
        "median_pearson": median_r,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "corr_lambda": args.corr_lambda,
        "use_feature_gate": args.use_feature_gate,
        "use_disease_prior": args.use_disease_prior,
        "prior_corr_lambda": args.prior_corr_lambda,
        "gene_list_file": args.gene_list_file,
        "split_mode": "custom32" if args.data == "her2st" else "cscc",
    }])
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    per_gene_df = pd.DataFrame({
        "fold": args.fold,
        "test_slice": target_testset.names[0],
        "gene": target_names,
        "pearson": R,
    })
    per_gene_df.to_csv(os.path.join(out_dir, "per_gene_pearson.csv"), index=False)

    train_exp_all = concat_dataset_exp(target_trainset)
    test_exp_all = concat_dataset_exp(target_testset)

    diagnostic_df = pd.DataFrame({
        "gene": target_names,
        "pearson": R,
        "train_mean": train_exp_all.mean(axis=0),
        "train_std": train_exp_all.std(axis=0),
        "test_mean": test_exp_all.mean(axis=0),
        "test_std": test_exp_all.std(axis=0),
        "train_nonzero_ratio": (np.abs(train_exp_all) > 1e-8).mean(axis=0),
        "test_nonzero_ratio": (np.abs(test_exp_all) > 1e-8).mean(axis=0),
    })
    diagnostic_df.to_csv(os.path.join(out_dir, "gene_expression_diagnostics.csv"), index=False)

    pred_df = pd.DataFrame(pred.X, columns=target_names)
    gt_df = pd.DataFrame(gt.X, columns=target_names)
    pred_df.to_csv(os.path.join(out_dir, "pred_matrix.csv"), index=False)
    gt_df.to_csv(os.path.join(out_dir, "gt_matrix.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("Results saved to:", out_dir)
    if args.save_model == "T":
        checkpoint_path = os.path.join(out_dir, "final_model.ckpt")
        finetrainer.save_checkpoint(checkpoint_path)
        print("Final model checkpoint saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
