import torch
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from dpmg_data import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4,
            target_mode='gene', target_definitions_path=None, target_score_method='mean'):
    assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r,
            target_mode=target_mode, target_definitions_path=target_definitions_path,
            target_score_method=target_score_method
        )
    elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r,
            target_mode=target_mode, target_definitions_path=target_definitions_path,
            target_score_method=target_score_method
        )
    return dataset
def test(model, test, device='cuda'):
    model = model.to(device)
    model.eval()
    predictions, targets, centers = [], [], []
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position = patch.to(device), position.to(device)
            adj = adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            predictions.append(pred.reshape(-1, model.n_genes).cpu().numpy())
            targets.append(exp.reshape(-1, model.n_genes).cpu().numpy())
            centers.append(center.reshape(-1, 2).cpu().numpy())
    if not predictions:
        raise ValueError('The test loader is empty.')
    adata = ad.AnnData(np.concatenate(predictions, axis=0))
    adata_gt = ad.AnnData(np.concatenate(targets, axis=0))
    adata.obsm['spatial'] = np.concatenate(centers, axis=0)
    adata_gt.obsm['spatial'] = adata.obsm['spatial'].copy()
    return adata, adata_gt

def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)
def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1


