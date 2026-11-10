import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from pathway_utils import load_gene_list, load_pathway_definitions, resolve_gene_subset, score_pathways
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _build_expression_dataframe(meta_df, gene_columns):
    expr = meta_df[gene_columns].copy()
    expr = expr.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return expr


def _log_normalize_expression(expr_df):
    expr = expr_df.to_numpy(dtype=np.float32)
    norm = scp.normalize.library_size_normalize(expr)
    norm = scp.transform.log(norm)
    return pd.DataFrame(norm, index=expr_df.index, columns=expr_df.columns)


def _resolve_targets(meta_df, gene_columns, target_mode='gene', target_definitions=None, target_score_method='mean'):
    expr_df = _build_expression_dataframe(meta_df, gene_columns)
    log_expr_df = _log_normalize_expression(expr_df)
    if target_mode == 'gene':
        target_names = list(gene_columns)
        targets = log_expr_df[target_names].to_numpy(dtype=np.float32)
        return targets, target_names, expr_df[target_names].to_numpy(dtype=np.float32), None
    if target_mode == 'gene_subset':
        if target_definitions is None:
            raise ValueError('target_definitions is required when target_mode="gene_subset"')
        subset_genes = resolve_gene_subset(log_expr_df, target_definitions)
        targets = log_expr_df[subset_genes].to_numpy(dtype=np.float32)
        return targets, subset_genes, expr_df[subset_genes].to_numpy(dtype=np.float32), subset_genes
    if target_definitions is None:
        raise ValueError('target_definitions is required when target_mode="pathway"')
    pathway_scores, target_names, used_genes = score_pathways(log_expr_df, target_definitions, method=target_score_method)
    return pathway_scores, target_names, None, used_genes

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4,
                 target_mode='gene', target_definitions_path=None, target_score_method='mean'):
        super(ViT_HER2ST, self).__init__()
        
        data_root = os.environ.get('HIST2ST_DATA_ROOT', 'data')
        self.cnt_dir = os.path.join(data_root, 'her2st/data/ST-cnts')
        self.img_dir = os.path.join(data_root, 'her2st/data/ST-imgs')
        self.pos_dir = os.path.join(data_root, 'her2st/data/ST-spotfiles')
        self.lbl_dir = os.path.join(data_root, 'her2st/data/ST-pat/lbl')
        self.r = 224//r
        self.target_mode = target_mode
        self.target_score_method = target_score_method
        if target_mode == 'pathway':
            self.target_definitions = load_pathway_definitions(target_definitions_path)
        elif target_mode == 'gene_subset':
            self.target_definitions = load_gene_list(target_definitions_path)
        else:
            self.target_definitions = None
        self.pathway_gene_usage = {}

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load(os.path.join(data_root, 'her_hvg_cut_1000.npy'),allow_pickle=True))
        self.gene_list = gene_list

        all_samples = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
            'E1', 'E2', 'E3',
            'F1', 'F2', 'F3',
            'G1', 'G2', 'G3',
            'H1', 'H2', 'H3'
        ]

        low_spot_exclude = {'A5', 'F1', 'F2', 'F3'}

        samples = [s for s in all_samples if s not in low_spot_exclude]


        self.train = train
        self.ori = ori
        self.adj = adj


        if fold < 0 or fold >= len(samples):
            raise ValueError(
                f"Invalid fold={fold}. Available fold range: 0 to {len(samples) - 1}. "
                f"Current samples={samples}"
            )

        te_names = [samples[fold]]
        tr_names = [x for x in samples if x not in te_names]

        print("=" * 80)
        print("[HER2ST SPLIT DEBUG]")
        print("fold =", fold)
        print("all original samples =", all_samples)
        print("excluded low-spot samples =", sorted(list(low_spot_exclude)))
        print("final samples =", samples)
        print("num final samples =", len(samples))
        print("test names =", te_names)
        print("train names =", tr_names)
        print("=" * 80)
        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        self.exp_dict = {}
        self.target_names = None
        self.used_genes = {}
        if self.ori:
            self.ori_dict = {}
            self.counts_dict={}
        for i, m in self.meta_dict.items():
            targets, target_names, ori_vals, used_genes = _resolve_targets(
                m, self.gene_set, target_mode=self.target_mode,
                target_definitions=self.target_definitions,
                target_score_method=self.target_score_method,
            )
            self.exp_dict[i] = targets
            if self.target_names is None:
                self.target_names = target_names
            if used_genes is not None:
                self.used_genes[i] = used_genes
            if self.ori:
                self.ori_dict[i] = ori_vals
                n_counts = ori_vals.sum(1)
                sf = n_counts / max(np.median(n_counts), 1e-8)
                self.counts_dict[i] = sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        #df = pd.read_csv(path,sep='\t',index_col=0)
        #print("[READ CNT]", name, path)
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4,
                 target_mode='gene', target_definitions_path=None, target_score_method='mean'):
        super(ViT_SKIN, self).__init__()

        data_root = os.environ.get('HIST2ST_DATA_ROOT', 'data')
        self.dir = os.path.join(data_root, 'GSE144240_RAW') + os.sep
        self.r = 224//r
        self.target_mode = target_mode
        self.target_score_method = target_score_method
        if target_mode == 'pathway':
            self.target_definitions = load_pathway_definitions(target_definitions_path)
        elif target_mode == 'gene_subset':
            self.target_definitions = load_gene_list(target_definitions_path)
        else:
            self.target_definitions = None
        self.pathway_gene_usage = {}

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load(os.path.join(data_root, 'skin_hvg_cut_1000.npy'),allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        self.exp_dict = {}
        self.target_names = None
        self.used_genes = {}
        if self.ori:
            self.ori_dict = {}
            self.counts_dict={}
        for i, m in self.meta_dict.items():
            targets, target_names, ori_vals, used_genes = _resolve_targets(
                m, self.gene_set, target_mode=self.target_mode,
                target_definitions=self.target_definitions,
                target_score_method=self.target_score_method,
            )
            if self.norm and self.target_mode == 'gene':
                mu = targets.mean(axis=0, keepdims=True)
                sigma = targets.std(axis=0, keepdims=True)
                sigma[sigma < 1e-8] = 1.0
                targets = (targets - mu) / sigma
            self.exp_dict[i] = targets
            if self.target_names is None:
                self.target_names = target_names
            if used_genes is not None:
                self.used_genes[i] = used_genes
            if self.ori:
                self.ori_dict[i] = ori_vals
                n_counts = ori_vals.sum(1)
                sf = n_counts / max(np.median(n_counts), 1e-8)
                self.counts_dict[i] = sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)