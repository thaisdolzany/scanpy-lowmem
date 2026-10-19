"""Benchmark: does PCA on a sparse HVG matrix need zero_center=False to fit in RAM?

The matrix is built once and saved to disk. Each PCA variant then runs in a fresh
subprocess that only loads it, so the process peak (peak working set on Windows,
max RSS elsewhere) before PCA is the loaded matrix and not the temporary arrays used
to generate it, which would otherwise hide any PCA allocation below that level. Also checks how far each result
is from an exact, centred PCA computed on a small subsample.

Usage:
    python benchmarks/pca_centering.py --cells 200000 --genes 2000 --density 0.08

Output: one CSV row per variant in benchmarks/results/pca_centering.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "centred_sparse_arpack": dict(zero_center=True, svd_solver="arpack"),
    "uncentred_truncated_svd": dict(zero_center=False, svd_solver="arpack"),
    "centred_dense": dict(zero_center=True, svd_solver="arpack", densify=True),
}


def make_matrix(n_cells: int, n_genes: int, density: float, seed: int = 0):
    """Synthetic log-normalised counts with a sparse, skewed structure like scRNA-seq."""
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=density, format="csr", dtype=np.float32, random_state=rng,
                  data_rvs=lambda n: np.log1p(rng.negative_binomial(2, 0.3, size=n) + 1).astype(np.float32))
    # two latent groups so the leading PC has real structure to find
    group = rng.random(n_cells) < 0.3
    X = X.tocoo()
    X.data[group[X.row] & (X.col < 50)] *= 3
    return X.tocsr()


def peak_memory_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
    if peak is None:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return peak / 1e6


def run_variant(name: str, matrix_path: Path, n_pcs: int) -> dict:
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    X = sp.load_npz(matrix_path)
    n_cells, n_genes = X.shape
    density = round(X.nnz / (n_cells * n_genes), 4)
    base_mb = peak_memory_mb()
    params = dict(VARIANTS[name])
    densify = params.pop("densify", False)
    adata = ad.AnnData(X.toarray() if densify else X)
    t0 = time.perf_counter()
    sc.tl.pca(adata, n_comps=n_pcs, random_state=0, **params)
    elapsed = time.perf_counter() - t0
    np.save(HERE / "results" / f"_pcs_{name}.npy", adata.obsm["X_pca"][:5000, :10])
    return {"variant": name, "cells": n_cells, "genes": n_genes, "density": density,
            "peak_mb": round(peak_memory_mb(), 1), "peak_before_pca_mb": round(base_mb, 1),
            "pca_added_mb": round(peak_memory_mb() - base_mb, 1),
            "seconds": round(elapsed, 1)}


def subspace_agreement(a, b) -> float:
    """Overlap of the subspaces spanned by the first 10 PCs (mean squared cosine of the
    principal angles): 1 = same subspace, 0 = orthogonal. Invariant to sign and to the
    order of the components, so a shifted first component is not penalised."""
    import numpy as np

    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s ** 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=200_000)
    p.add_argument("--genes", type=int, default=2000)
    p.add_argument("--density", type=float, default=0.08)
    p.add_argument("--pcs", type=int, default=50)
    p.add_argument("--variant", help=argparse.SUPPRESS)
    p.add_argument("--matrix", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--skip-dense", action="store_true", help="skip the dense reference if RAM is short")
    args = p.parse_args()

    (HERE / "results").mkdir(exist_ok=True)
    if args.variant:
        print(json.dumps(run_variant(args.variant, args.matrix, args.pcs)))
        return

    import scipy.sparse as sp

    matrix_path = HERE / "results" / "_matrix.npz"
    sp.save_npz(matrix_path, make_matrix(args.cells, args.genes, args.density), compressed=False)

    rows = []
    for name in VARIANTS:
        if name == "centred_dense" and args.skip_dense:
            continue
        out = subprocess.run([sys.executable, __file__, "--variant", name, "--matrix", str(matrix_path),
                              "--pcs", str(args.pcs)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{name} failed:\n{out.stderr[-1500:]}", file=sys.stderr)
            continue
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1])

    import numpy as np
    ref_path = HERE / "results" / "_pcs_centred_dense.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
        for r in rows:
            r["agreement_with_dense_pca"] = round(subspace_agreement(np.load(HERE / "results" / f"_pcs_{r['variant']}.npy"), ref), 4)
    for f in [*(HERE / "results").glob("_pcs_*.npy"), matrix_path]:
        f.unlink()

    env = {"python": platform.python_version(), "scanpy": version("scanpy"), "anndata": version("anndata"),
           "numpy": version("numpy"), "scipy": version("scipy"), "machine": platform.platform()}
    out_csv = HERE / "results" / "pca_centering.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + list(env.keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, **env})
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
