import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf

from dpmg_graph_layers import *
from dpmg_count_losses import *
from dpmg_attention import *
from dpmg_priors import weighted_gene_mse, weighted_pair_corr_loss

from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans

try:
    import scanpy as sc
    import anndata as ann
except Exception:
    sc = None
    ann = None


class PearsonCorrLoss(nn.Module):
    def __init__(self, eps=1e-8, min_target_std=1e-4):
        super().__init__()
        self.eps = eps
        self.min_target_std = min_target_std
        self.last_valid_ratio = 0.0
        self.last_valid_num = 0

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)

        target_std_raw = target.std(dim=0, unbiased=False)
        valid_gene_mask = target_std_raw > self.min_target_std

        self.last_valid_num = int(valid_gene_mask.sum().detach().cpu().item())
        self.last_valid_ratio = float(valid_gene_mask.float().mean().detach().cpu().item())

        if valid_gene_mask.sum() == 0:
            return pred.sum() * 0.0

        pred = pred[:, valid_gene_mask]
        target = target[:, valid_gene_mask]

        pred = pred - pred.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)

        pred_norm = torch.sqrt((pred ** 2).sum(dim=0) + self.eps)
        target_norm = torch.sqrt((target ** 2).sum(dim=0) + self.eps)

        corr = (pred * target).sum(dim=0) / (pred_norm * target_norm + self.eps)

        valid_corr_mask = torch.isfinite(corr)
        if valid_corr_mask.sum() == 0:
            return pred.sum() * 0.0

        corr = corr[valid_corr_mask]
        return 1.0 - corr.mean()


class FeatureGate(nn.Module):
    """
    轻量注意力门控：
    输入支持 [N, D] 或 [B, N, D]
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
        self.gate_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])     # [M, D]
        gate = self.mlp(x2)                    # [M, D]
        # out = x2 + x2 * gate
        out = x2 + self.gate_scale * x2 * gate
        spot_score = gate.mean(dim=-1)         # [M]
        return out.reshape(orig_shape), gate.reshape(orig_shape), spot_score.reshape(orig_shape[:-1])


class convmixer_block(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.dw(x) + x
        x = self.pw(x)
        return x


class mixer_transformer(nn.Module):
    def __init__(self, channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, depth3=4,
                 heads=8, dim_head=64, mlp_dim=1024, dropout=0.,
                 policy='mean', gcn=True):
        super().__init__()
        self.layer1 = nn.Sequential(
            *[convmixer_block(channel, kernel_size) for _ in range(depth1)],
        )
        self.layer2 = nn.Sequential(
            *[attn_block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth2)]
        )
        self.layer3 = nn.ModuleList([gs_block(dim, dim, policy, gcn) for _ in range(depth3)])
        self.jknet = nn.Sequential(
            nn.LSTM(dim, dim, 2),
            SelectItem(0),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x, ct, adj):
        x = self.down(self.layer1(x))
        g = x.unsqueeze(0)
        g = self.layer2(g + ct).squeeze(0)
        jk = []
        for layer in self.layer3:
            g = layer(g, adj)
            jk.append(g.unsqueeze(0))
        g = torch.cat(jk, 0)
        g = self.jknet(g).mean(0)
        return g


class ViT(nn.Module):
    def __init__(self, channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, depth3=4,
                 heads=8, mlp_dim=1024, dim_head=64, dropout=0.,
                 policy='mean', gcn=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = mixer_transformer(
            channel, kernel_size, dim,
            depth1, depth2, depth3,
            heads, dim_head, mlp_dim, dropout,
            policy, gcn,
        )

    def forward(self, x, ct, adj):
        x = self.dropout(x)
        x = self.transformer(x, ct, adj)
        return x


class DPMG(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, fig_size=112, label=None,
                 dropout=0.2, n_pos=64, kernel_size=5, patch_size=7, n_genes=785,
                 depth1=2, depth2=8, depth3=4, heads=16, channel=32,
                 zinb=0, nb=False, bake=0, lamb=0, policy='mean',
                 task_type='gene', target_names=None,
                 corr_lambda=0.2, use_feature_gate=True, attn_reduction=4,
                 gene_weights=None, gene_relation_matrix=None, prior_corr_lambda=0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["label"])
        # Plain lists keep constructor metadata compatible with safe checkpoint loading.
        if gene_weights is not None:
            self.hparams.gene_weights = torch.as_tensor(gene_weights).tolist()
        if gene_relation_matrix is not None:
            self.hparams.gene_relation_matrix = torch.as_tensor(gene_relation_matrix).tolist()
        dim = (fig_size // patch_size) ** 2 * channel // 8
        self.learning_rate = learning_rate

        self.nb = nb
        self.zinb = zinb

        self.bake = bake
        self.lamb = lamb

        self.label = label
        self.task_type = task_type
        self.target_names = target_names if target_names is not None else list(range(n_genes))

        self.corr_lambda = corr_lambda
        self.use_feature_gate = use_feature_gate
        self.prior_corr_lambda = prior_corr_lambda

        if gene_weights is not None:
            gene_weights = torch.as_tensor(gene_weights, dtype=torch.float32)
            if gene_weights.numel() != n_genes:
                raise ValueError(
                    f'gene_weights length mismatch: {gene_weights.numel()} vs n_genes={n_genes}'
                )
            self.register_buffer('gene_weights', gene_weights)
        else:
            self.gene_weights = None

        if gene_relation_matrix is not None:
            gene_relation_matrix = torch.as_tensor(gene_relation_matrix, dtype=torch.float32)
            if tuple(gene_relation_matrix.shape) != (n_genes, n_genes):
                raise ValueError(
                    f'gene_relation_matrix shape mismatch: {tuple(gene_relation_matrix.shape)} vs {(n_genes, n_genes)}'
                )
            self.register_buffer('gene_relation_matrix', gene_relation_matrix)
        else:
            self.gene_relation_matrix = None

        self.patch_embedding = nn.Conv2d(3, channel, patch_size, patch_size)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, heads=heads,
            dim=dim, depth1=depth1, depth2=depth2, depth3=depth3,
            mlp_dim=dim, dropout=dropout, policy=policy, gcn=True,
        )

        self.channel = channel
        self.patch_size = patch_size
        self.n_genes = n_genes

        if self.zinb > 0:
            if self.nb:
                self.hr = nn.Linear(dim, n_genes)
                self.hp = nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())

        if self.bake > 0:
            self.coef = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )

        self.feature_gate = FeatureGate(dim, reduction=attn_reduction)
        self.corr_loss_fn = PearsonCorrLoss()

        self.latest_gate = None
        self.latest_spot_score = None

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )

        self.tf = tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2),
        ])

    def forward(self, patches, centers, adj, aug=False):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        patches = self.patch_embedding(patches)

        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        ct = centers_x + centers_y

        h = self.vit(patches, ct, adj)

        if self.use_feature_gate:
            h, gate_w, spot_score = self.feature_gate(h)
            self.latest_gate = gate_w
            self.latest_spot_score = spot_score
        else:
            self.latest_gate = None
            self.latest_spot_score = None

        x = self.gene_head(h)

        extra = None
        if self.zinb > 0:
            if self.nb:
                r = self.hr(h)
                p = self.hp(h)
                extra = (r, p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra = (m, d, p)

        if aug:
            h = self.coef(h)

        return x, extra, h

    def aug(self, patch, center, adj):
        bake_x = []
        for _ in range(self.bake):
            new_patch = self.tf(patch.squeeze(0)).unsqueeze(0)
            x, _, h = self(new_patch, center, adj, True)
            bake_x.append((x.unsqueeze(0), h.unsqueeze(0)))
        return bake_x

    def distillation(self, bake_x):
        new_x, coef = zip(*bake_x)
        coef = torch.cat(coef, 0)
        new_x = torch.cat(new_x, 0)
        coef = F.softmax(coef, dim=0)
        new_x = (new_x * coef).sum(0)
        return new_x

    def get_latest_spot_score(self):
        return self.latest_spot_score

    def _unpack_batch(self, batch):
        patch, center, exp, adj, *rest = batch
        payload = {
            'patch': patch,
            'center': center,
            'exp': exp,
            'adj': adj.squeeze(0),
            'oris': None,
            'sfs': None,
        }
        if len(rest) >= 2:
            payload['oris'] = rest[0]
            payload['sfs'] = rest[1]
        return payload

    def training_step(self, batch, batch_idx):
        batch = self._unpack_batch(batch)
        patch, center, exp, adj = batch['patch'], batch['center'], batch['exp'], batch['adj']
        oris, sfs = batch['oris'], batch['sfs']

        exp = exp.squeeze(0)
        pred, extra, h = self(patch, center, adj)

        raw_mse_loss = F.mse_loss(pred, exp)
        mse_loss = weighted_gene_mse(pred, exp, self.gene_weights)
        corr_loss = self.corr_loss_fn(pred, exp)
        prior_corr_loss = weighted_pair_corr_loss(pred, exp, self.gene_relation_matrix)

        self.log('raw_mse_loss', raw_mse_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('corr_loss', corr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('prior_corr_loss', prior_corr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        corr_valid_ratio = torch.tensor(self.corr_loss_fn.last_valid_ratio, device=self.device)
        corr_valid_num = torch.tensor(self.corr_loss_fn.last_valid_num, device=self.device)

        self.log('corr_valid_ratio', corr_valid_ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('corr_valid_num', corr_valid_num, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        bake_loss = 0
        if self.bake > 0:
            bake_x = self.aug(patch, center, adj)
            new_pred = self.distillation(bake_x)
            bake_loss = F.mse_loss(new_pred, pred)
            self.log('bake_loss', bake_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        zinb_loss = 0
        if self.zinb > 0 and oris is not None and sfs is not None:
            if self.nb:
                r, p = extra
                zinb_loss = NB_loss(oris.squeeze(0), r, p)
            else:
                m, d, p = extra
                zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))
            self.log('zinb_loss', zinb_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        loss = (
            mse_loss
            + self.corr_lambda * corr_loss
            + self.prior_corr_lambda * prior_corr_loss
            + self.zinb * zinb_loss
            + self.lamb * bake_loss
        )
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._unpack_batch(batch)
        patch, center, exp, adj = batch['patch'], batch['center'], batch['exp'], batch['adj']

        def cluster(pred, cls):
            sc.pp.pca(pred)
            sc.tl.tsne(pred)
            kmeans = KMeans(n_clusters=cls, init="k-means++", random_state=0).fit(pred.obsm['X_pca'])
            pred.obs['kmeans'] = kmeans.labels_.astype(str)
            p = pred.obs['kmeans'].to_numpy()
            return p

        pred, extra, h = self(patch, center, adj)

        if self.label is not None and self.task_type == 'gene' and sc is not None and ann is not None:
            adata = ann.AnnData(pred.squeeze().cpu().numpy())
            idx = self.label != 'undetermined'
            cls = len(set(self.label))
            x = adata[idx]
            l = self.label[idx]
            predlbl = cluster(x, cls - 1)
            self.log('nmi', nmi_score(predlbl, l))
            self.log('ari', ari_score(predlbl, l))

        pred_val = pred.squeeze(0)
        exp_val = exp.squeeze(0)
        raw_mse_loss = F.mse_loss(pred_val, exp_val)
        mse_loss = weighted_gene_mse(pred_val, exp_val, self.gene_weights)
        corr_loss = self.corr_loss_fn(pred_val, exp_val)
        prior_corr_loss = weighted_pair_corr_loss(pred_val, exp_val, self.gene_relation_matrix)
        total_val_loss = mse_loss + self.corr_lambda * corr_loss + self.prior_corr_lambda * prior_corr_loss

        val_corr_valid_ratio = torch.tensor(self.corr_loss_fn.last_valid_ratio, device=self.device)
        val_corr_valid_num = torch.tensor(self.corr_loss_fn.last_valid_num, device=self.device)

        self.log('val_corr_valid_ratio', val_corr_valid_ratio, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_corr_valid_num', val_corr_valid_num, on_epoch=True, prog_bar=False, logger=True)

        self.log('val_raw_mse_loss', raw_mse_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log('valid_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_corr_loss', corr_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prior_corr_loss', prior_corr_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_total_loss', total_val_loss, on_epoch=True, prog_bar=False, logger=True)

        pred_np = pred.squeeze(0).detach().cpu().numpy()
        exp_np = exp.squeeze(0).detach().cpu().numpy()

        pearsons = []
        spearmans = []
        for g in range(self.n_genes):
            try:
                pearsons.append(pearsonr(pred_np[:, g], exp_np[:, g])[0])
            except Exception:
                pearsons.append(np.nan)
            try:
                spearmans.append(spearmanr(pred_np[:, g], exp_np[:, g])[0])
            except Exception:
                spearmans.append(np.nan)

        R = torch.tensor(np.nanmean(pearsons), device=self.device)
        S = torch.tensor(np.nanmean(spearmans), device=self.device)

        self.log('R', R, on_epoch=True, prog_bar=True, logger=True)
        self.log('S', S, on_epoch=True, prog_bar=False, logger=True)

        return total_val_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
