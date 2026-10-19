"""Sparse-friendly, low-memory pattern for concatenating + integrating two AnnData
cohorts with Harmony on a machine with limited RAM (developed on 16GB).

Four fixes bundled here, each targeting a specific MemoryError:

1. Pre-align to common genes, in the same order, BEFORE ad.concat. Concatenating
   AnnDatas with mismatched var order makes ad.concat do an internal sparse
   reindex-matmul (`el @ idxmtx`) per object, which can blow up to hundreds of
   millions of nonzeros on real gene-symbol mismatches.
2. Stratified subsampling per batch (e.g. patient x cohort) BEFORE any heavy step,
   capped at MAX_CELLS_PER_BATCH. This is a trade-off for constrained hardware:
   rare cell populations lose cells first, so check their counts after subsampling,
   and raise the cap or drop the step on a bigger machine.
3. PCA on the HVG subset, kept sparse. With a sparse matrix and svd_solver="arpack",
   Scanpy centres implicitly and never builds the dense (cells x HVG) array, so the
   centred PCA adds little memory (+187 MB on 200,000 cells x 2,000 genes). Do not
   switch to zero_center=False to save RAM: at that size it used more (+443 MB) and
   changed the components (benchmarks/pca_centering.py, results in
   benchmarks/results/). Densifying the matrix first is what costs memory (+5 GB).
4. `sc.tl.leiden(..., flavor="igraph")` instead of the `leidenalg` default. The
   leidenalg wrapper builds its own C-level graph representation, which runs out
   of memory on graphs with several hundred thousand nodes well before igraph does.

Usage: load your two cohorts as raw-count AnnData objects and call
`integrate(a1, a2, batch_key=...)`, where batch_key is an obs column present in both
(for example patient or sample). Each step is also usable on its own.
"""
from __future__ import annotations

import gc

import anndata as ad
import numpy as np
import scanpy as sc

N_HVG = 2000
N_PCS = 50
MAX_CELLS_PER_BATCH = 2000
RNG_SEED = 0


def align_and_concat(a1: ad.AnnData, a2: ad.AnnData, batch_label: str = "_concat_batch") -> ad.AnnData:
    common_genes = sorted(set(a1.var_names) & set(a2.var_names))
    print(f"common genes: {len(common_genes)}")
    a1 = a1[:, common_genes].copy()
    gc.collect()
    a2 = a2[:, common_genes].copy()
    gc.collect()
    adata = ad.concat([a1, a2], join="outer", merge="unique", label=batch_label)
    del a1, a2
    gc.collect()
    return adata


def stratified_subsample(adata: ad.AnnData, batch_key: str, max_per_batch: int = MAX_CELLS_PER_BATCH,
                          seed: int = RNG_SEED) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_before = adata.n_obs
    keep_idx = []
    for _, group in adata.obs.groupby(batch_key).indices.items():
        if len(group) > max_per_batch:
            group = rng.choice(group, size=max_per_batch, replace=False)
        keep_idx.append(group)
    keep_idx = np.sort(np.concatenate(keep_idx))
    adata = adata[keep_idx].copy()
    gc.collect()
    print(f"cells {n_before} -> {adata.n_obs} (stratified subsample, max {max_per_batch}/batch)")
    return adata


def sparse_pca_harmony(adata: ad.AnnData, batch_key: str, n_hvg: int = N_HVG, n_pcs: int = N_PCS) -> None:
    """Adds `X_pca` and `X_pca_harmony` to adata.obsm in place, keeping the HVG
    matrix sparse throughout (no dense scaling step). Requires harmonypy."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key=batch_key, flavor="seurat")
    hvg = adata[:, adata.var["highly_variable"]].copy()
    # sparse input + arpack: Scanpy centres implicitly, no dense copy is made
    sc.tl.pca(hvg, n_comps=n_pcs, svd_solver="arpack", random_state=RNG_SEED)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    del hvg
    gc.collect()
    sc.external.pp.harmony_integrate(adata, key=batch_key, basis="X_pca")


def cluster_lowmem(adata: ad.AnnData, use_rep: str = "X_pca_harmony", n_pcs: int = N_PCS,
                    resolution: float = 1.0) -> None:
    sc.pp.neighbors(adata, use_rep=use_rep, n_pcs=n_pcs)
    sc.tl.umap(adata, random_state=RNG_SEED)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False,
                 random_state=RNG_SEED)


# --- pipeline starts here ---
def integrate(a1: ad.AnnData, a2: ad.AnnData, batch_key: str) -> ad.AnnData:
    adata = align_and_concat(a1, a2)
    adata = stratified_subsample(adata, batch_key=batch_key)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sparse_pca_harmony(adata, batch_key=batch_key)
    cluster_lowmem(adata)
    return adata
