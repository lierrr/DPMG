import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def load_pathway_definitions(path: str) -> Dict[str, List[str]]:
    """Load pathway definitions from JSON/CSV/GMT.

    Supported formats:
      - JSON: {"PathwayA": ["GENE1", "GENE2"], ...}
      - CSV/TSV: two columns named pathway,gene
      - GMT: pathway_name <tab> description <tab> gene1 <tab> gene2 ...
    """
    if path is None:
        raise ValueError("pathway definition file is required for pathway prediction mode")
    if not os.path.exists(path):
        raise FileNotFoundError(f"pathway definition file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    definitions: Dict[str, List[str]] = {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("JSON pathway file must be a dict: {pathway: [genes]}")
        for pathway, genes in raw.items():
            if isinstance(genes, str):
                genes = [genes]
            definitions[str(pathway)] = [str(g).strip() for g in genes if str(g).strip()]
    elif ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        cols = {c.lower(): c for c in df.columns}
        if "pathway" not in cols or "gene" not in cols:
            raise ValueError("CSV/TSV pathway file must contain columns: pathway, gene")
        for pathway, sub_df in df.groupby(cols["pathway"]):
            definitions[str(pathway)] = [str(g).strip() for g in sub_df[cols["gene"]].tolist() if str(g).strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                pathway = parts[0]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                definitions[pathway] = genes

    cleaned = {}
    for pathway, genes in definitions.items():
        unique_genes = []
        seen = set()
        for gene in genes:
            if gene not in seen:
                seen.add(gene)
                unique_genes.append(gene)
        if unique_genes:
            cleaned[pathway] = unique_genes

    if not cleaned:
        raise ValueError("No valid pathways were loaded from the pathway definition file")
    return cleaned


def load_gene_list(path: str) -> List[str]:
    """Load a gene subset file.

    Supported formats:
      - TXT: one gene per line
      - CSV/TSV: either a single-column file or a column named gene
      - JSON: ["GENE1", "GENE2"] or {"genes": [...]} 
    """
    if path is None:
        raise ValueError("gene list file is required for gene subset prediction mode")
    if not os.path.exists(path):
        raise FileNotFoundError(f"gene list file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    genes: List[str] = []

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if "genes" not in raw:
                raise ValueError('JSON gene file must be a list or a dict with key "genes"')
            raw = raw["genes"]
        if not isinstance(raw, list):
            raise ValueError("JSON gene file must contain a list of genes")
        genes = [str(g).strip() for g in raw if str(g).strip()]
    elif ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] == 1:
            genes = [str(g).strip() for g in df.iloc[:, 0].tolist() if str(g).strip()]
        else:
            cols = {c.lower(): c for c in df.columns}
            if "gene" not in cols:
                raise ValueError("CSV/TSV gene file must have a single column or a column named 'gene'")
            genes = [str(g).strip() for g in df[cols["gene"]].tolist() if str(g).strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            genes = [line.strip() for line in f if line.strip()]

    cleaned = []
    seen = set()
    for gene in genes:
        if gene not in seen:
            seen.add(gene)
            cleaned.append(gene)

    if not cleaned:
        raise ValueError("No valid genes were loaded from the gene list file")
    return cleaned


def resolve_gene_subset(expr_df: pd.DataFrame, selected_genes: List[str]):
    """Resolve a requested gene subset against a spot x gene expression table."""
    if not isinstance(expr_df, pd.DataFrame):
        expr_df = pd.DataFrame(expr_df)

    columns = list(expr_df.columns)
    gene_to_col = {str(g): g for g in columns}
    overlap = [gene_to_col[g] for g in selected_genes if g in gene_to_col]
    if not overlap:
        raise ValueError(
            "No requested subset genes overlap with the current dataset. "
            "Please check the gene symbols in your gene list file."
        )
    return overlap


def score_pathways(expr_df: pd.DataFrame, pathway_defs: Dict[str, List[str]], method: str = "mean"):
    """Compute pathway scores from a spot x gene expression table.

    Args:
        expr_df: normalized/log-transformed expression, rows are spots and columns are genes.
        pathway_defs: pathway -> gene list.
        method: 'mean' or 'zscore_mean'.

    Returns:
        scores: np.ndarray of shape [n_spots, n_pathways]
        target_names: list[str]
        used_genes: dict[pathway] -> list[str]
    """
    if not isinstance(expr_df, pd.DataFrame):
        expr_df = pd.DataFrame(expr_df)

    method = method.lower()
    if method not in {"mean", "zscore_mean"}:
        raise ValueError("Unsupported pathway score method. Use 'mean' or 'zscore_mean'.")

    columns = list(expr_df.columns)
    gene_to_col = {str(g): g for g in columns}

    target_names = []
    score_list = []
    used_genes = {}

    for pathway, genes in pathway_defs.items():
        overlap = [gene_to_col[g] for g in genes if g in gene_to_col]
        if not overlap:
            continue
        sub = expr_df[overlap].to_numpy(dtype=np.float32)
        if method == "zscore_mean":
            mu = sub.mean(axis=0, keepdims=True)
            sigma = sub.std(axis=0, keepdims=True)
            sigma[sigma < 1e-8] = 1.0
            sub = (sub - mu) / sigma
        score = sub.mean(axis=1)
        target_names.append(pathway)
        score_list.append(score)
        used_genes[pathway] = overlap

    if not score_list:
        raise ValueError(
            "No pathway genes overlap with the current dataset. "
            "Please check gene symbols in your pathway file."
        )

    scores = np.stack(score_list, axis=1).astype(np.float32)
    return scores, target_names, used_genes
