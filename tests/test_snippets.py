"""Small, network-free tests for the snippets. Run with:  python -m pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))

from infercnvpy_pandas3_patch import force_object_dtype  # noqa: E402
from integrate_sparse import align_and_concat, stratified_subsample  # noqa: E402
from qc_mad import is_outlier, qc_filter  # noqa: E402


def counts_adata(n_cells=400, n_genes=300, seed=0, genes=None):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(np.float32))
    genes = genes if genes is not None else [f"G{i}" for i in range(n_genes - 5)] + [f"MT-{i}" for i in range(5)]
    return ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
                      var=pd.DataFrame(index=genes))


# ---------------------------------------------------------------- qc_mad
def test_is_outlier_flags_extremes_only():
    values = np.r_[np.full(99, 10.0) + np.linspace(-1, 1, 99), 100.0]
    flags = is_outlier(values, n_mads=5)
    assert flags[-1] and flags[:-1].sum() == 0


def test_qc_filter_removes_injected_bad_cells():
    adata = counts_adata()
    X = adata.X.toarray()
    X[:5, :] = 0
    X[:5, :3] = 1                      # 5 near-empty cells
    X[5:10, -5:] = 500                 # 5 cells dominated by mitochondrial counts
    adata.X = sp.csr_matrix(X)
    out = qc_filter(adata, label="test")
    kept = set(out.obs_names)
    assert not any(f"c{i}" in kept for i in range(10))
    assert out.n_obs >= 380            # healthy cells are kept
    assert sp.issparse(out.X)


# ---------------------------------------------------------- integrate_sparse
def test_align_and_concat_keeps_only_shared_genes_in_one_order():
    a1 = counts_adata(n_cells=50, n_genes=6, genes=["A", "B", "C", "D", "E", "F"])
    a2 = counts_adata(n_cells=40, n_genes=5, seed=1, genes=["F", "C", "A", "X", "B"])
    a2.obs_names = [f"d{i}" for i in range(40)]
    merged = align_and_concat(a1, a2)
    assert list(merged.var_names) == ["A", "B", "C", "F"]
    assert merged.n_obs == 90
    # values follow their gene, not their original column position
    np.testing.assert_array_equal(merged[a2.obs_names, "F"].X.toarray(), a2[:, "F"].X.toarray())


def test_stratified_subsample_caps_each_batch_and_is_reproducible():
    adata = counts_adata(n_cells=500)
    adata.obs["batch"] = pd.Categorical(["a"] * 300 + ["b"] * 150 + ["c"] * 50)
    out1 = stratified_subsample(adata, "batch", max_per_batch=100, seed=1)
    out2 = stratified_subsample(adata, "batch", max_per_batch=100, seed=1)
    counts = out1.obs["batch"].value_counts()
    assert counts["a"] == 100 and counts["b"] == 100 and counts["c"] == 50
    assert list(out1.obs_names) == list(out2.obs_names)


def test_sparse_centred_pca_matches_dense_pca():
    """The reason the PCA step stays centred: sparse + arpack equals the exact PCA."""
    adata = counts_adata(n_cells=600, n_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    dense = adata.copy()
    dense.X = dense.X.toarray()
    sc.tl.pca(adata, n_comps=10, svd_solver="arpack", random_state=0)
    sc.tl.pca(dense, n_comps=10, svd_solver="arpack", random_state=0)
    a, b = adata.obsm["X_pca"], dense.obsm["X_pca"]
    cos = np.abs(np.sum(a * b, axis=0)) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    assert np.all(cos > 0.999)


def test_sparse_pca_harmony_runs_end_to_end():
    pytest.importorskip("harmonypy")
    from integrate_sparse import sparse_pca_harmony

    adata = counts_adata(n_cells=300, n_genes=400)
    adata.obs["batch"] = pd.Categorical(np.repeat(["a", "b"], 150))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sparse_pca_harmony(adata, batch_key="batch", n_hvg=200, n_pcs=10)
    assert adata.obsm["X_pca_harmony"].shape == (300, 10)


# ------------------------------------------------------ infercnvpy patch
def test_force_object_dtype_converts_gene_names_and_chromosome():
    adata = counts_adata(n_cells=10, n_genes=8)
    adata.var["chromosome"] = pd.array(["chr1"] * 8, dtype="string")
    force_object_dtype(adata)
    assert adata.var.index.dtype == object
    assert adata.var["chromosome"].dtype == object
