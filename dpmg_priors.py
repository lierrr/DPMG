import numpy as np
import pandas as pd
import torch


def load_gene_weights(target_genes, prior_file):
    """
    Load gene-level disease weights.
    Return numpy array ordered by target_genes.
    """
    prior = pd.read_csv(prior_file)

    if "gene" not in prior.columns or "disease_weight" not in prior.columns:
        raise ValueError("prior_file must contain columns: gene,disease_weight")

    weight_map = dict(zip(prior["gene"].astype(str), prior["disease_weight"]))

    weights = []
    missing = []

    for g in target_genes:
        g = str(g)
        if g in weight_map:
            weights.append(float(weight_map[g]))
        else:
            missing.append(g)
            weights.append(1.0)

    if len(missing) > 0:
        print("[WARNING] Missing genes in disease prior file:", missing)

    weights = np.asarray(weights, dtype=np.float32)

    # Normalize to mean 1.0 to avoid changing total loss scale too much
    weights = weights / (weights.mean() + 1e-8)

    return weights


def load_gene_relation_matrix(target_genes, relation_file):
    """
    Load disease-aware gene-gene relation matrix.
    Return G x G numpy matrix ordered by target_genes.
    """
    rel = pd.read_csv(relation_file)

    required = {"gene_a", "gene_b", "relation_weight"}
    if not required.issubset(set(rel.columns)):
        raise ValueError(
            "relation_file must contain columns: gene_a,gene_b,relation_weight"
        )

    target_genes = [str(g) for g in target_genes]
    g2i = {g: i for i, g in enumerate(target_genes)}
    G = len(target_genes)

    mat = np.zeros((G, G), dtype=np.float32)

    for _, row in rel.iterrows():
        a = str(row["gene_a"])
        b = str(row["gene_b"])
        w = float(row["relation_weight"])

        if a in g2i and b in g2i:
            i = g2i[a]
            j = g2i[b]
            mat[i, j] = w
            mat[j, i] = w

    np.fill_diagonal(mat, 0.0)

    return mat


def weighted_gene_mse(pred, target, gene_weights=None):
    """
    Weighted expression reconstruction loss.

    pred, target: [N, G] or [1, N, G]
    gene_weights: [G]
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)

    if gene_weights is None:
        return torch.mean((pred - target) ** 2)

    if not torch.is_tensor(gene_weights):
        gene_weights = torch.tensor(gene_weights, dtype=torch.float32, device=pred.device)
    else:
        gene_weights = gene_weights.to(pred.device).float()

    w = gene_weights.view(1, -1)

    return torch.mean(((pred - target) ** 2) * w)


def _corr_matrix(x, eps=1e-6):
    """
    x: [N, G]
    Return gene-gene Pearson correlation matrix [G, G].
    """
    if x.dim() == 3:
        x = x.squeeze(0)

    x = x.float()

    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    x = x / (std + eps)

    n = x.shape[0]
    if n <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device)

    return torch.matmul(x.t(), x) / max(n - 1, 1)


def weighted_pair_corr_loss(pred, target, relation_matrix=None, eps=1e-6, base_pair_weight=0.10):
    """
    Disease-aware pairwise gene correlation loss.

    This loss encourages the predicted gene-gene correlation matrix to match
    the true gene-gene correlation matrix. Disease-related gene pairs can be
    emphasized by relation_matrix.

    pred, target: [N, G] or [1, N, G]
    relation_matrix: [G, G]
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)

    if pred.shape[0] < 3:
        return pred.sum() * 0.0

    G = pred.shape[1]

    pred_corr = _corr_matrix(pred, eps=eps)
    target_corr = _corr_matrix(target, eps=eps)

    diff = (pred_corr - target_corr) ** 2

    mask = torch.triu(torch.ones((G, G), device=pred.device), diagonal=1)

    if relation_matrix is None:
        weight = mask
    else:
        if not torch.is_tensor(relation_matrix):
            relation_matrix = torch.tensor(
                relation_matrix, dtype=torch.float32, device=pred.device
            )
        else:
            relation_matrix = relation_matrix.to(pred.device).float()

        # Keep a small base weight for all gene pairs,
        # and add disease relation weights for prior-related pairs.
        weight = base_pair_weight * mask + relation_matrix * mask

    loss = (diff * weight).sum() / (weight.sum() + eps)

    if not torch.isfinite(loss):
        return pred.sum() * 0.0

    return loss


# Compatibility alias. If another file imports weighted_corr_loss, it will also work.
def weighted_corr_loss(pred, target, relation_matrix=None, eps=1e-6):
    return weighted_pair_corr_loss(
        pred,
        target,
        relation_matrix=relation_matrix,
        eps=eps,
    )
