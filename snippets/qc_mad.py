"""MAD-based (median absolute deviation) QC filtering for scRNA-seq, sc-best-practices style.

Fixed thresholds (e.g. "keep cells with >200 genes") break down across a scRNA-seq atlas
built from many samples/batches processed over years, where the *typical* library size and
gene count vary batch to batch. MAD-based outlier detection adapts to each metric's own
distribution instead. Adapted from the sc-best-practices QC chapter
(https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html).

Usage:
    adata = qc_filter(adata, label="my_sample")
"""
from __future__ import annotations

import gc

import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation

MAD_THRESH = 5
MT_MAD_THRESH = 3
MT_PCT_HARD_MAX = 20.0
MIN_GENES = 200
MIN_CELLS_PER_GENE = 3


def is_outlier(values: np.ndarray, n_mads: float) -> np.ndarray:
    med = np.median(values)
    mad = median_abs_deviation(values)
    return (values < med - n_mads * mad) | (values > med + n_mads * mad)


def qc_filter(adata: sc.AnnData, label: str = "sample") -> sc.AnnData:
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=[20], log1p=True, inplace=True)

    outlier = (
        is_outlier(adata.obs["log1p_total_counts"].values, MAD_THRESH)
        | is_outlier(adata.obs["log1p_n_genes_by_counts"].values, MAD_THRESH)
        | is_outlier(adata.obs["pct_counts_in_top_20_genes"].values, MAD_THRESH)
    )
    mt_outlier = is_outlier(adata.obs["pct_counts_mt"].values, MT_MAD_THRESH) | (
        adata.obs["pct_counts_mt"].values > MT_PCT_HARD_MAX
    )

    n_before, n_genes_before = adata.n_obs, adata.n_vars
    keep_cells = ((~outlier) & (~mt_outlier) & (adata.obs["n_genes_by_counts"] >= MIN_GENES)).values

    # Gene mask computed on the cell-subset (no copy yet), so min_cells is counted
    # only among cells actually being kept.
    n_cells_per_gene = np.asarray((adata.X[keep_cells] > 0).sum(axis=0)).ravel()
    keep_genes = n_cells_per_gene >= MIN_CELLS_PER_GENE

    # A single combined materialization instead of two sequential `.copy()` calls:
    # sc.pp.filter_genes does its own internal copy, which briefly doubles peak
    # memory on top of an already-materialized cell-filtered object.
    adata = adata[keep_cells, keep_genes].copy()
    gc.collect()

    print(
        f"[{label}] cells {n_before} -> {adata.n_obs} ({n_before - adata.n_obs} removed); "
        f"genes {n_genes_before} -> {adata.n_vars}"
    )
    return adata
