# scanpy-lowmem

Patterns and patches for running Scanpy, Harmony, pySCENIC and infercnvpy on single-cell
RNA-seq cohorts of a few hundred thousand cells on a 16 GB laptop, without a cluster.

They come from a multi-cohort breast cancer integration (about 486,000 cells before
subsampling). Each snippet fixed a real `MemoryError` or library incompatibility, and its
docstring explains the cause, not only the fix.

## Contents

| File | What it addresses |
|---|---|
| `snippets/qc_mad.py` | Cell QC with MAD-based outlier thresholds instead of fixed cut-offs, for atlases where library size and gene counts vary between samples |
| `snippets/integrate_sparse.py` | Concatenating cohorts with mismatched gene order, a centred PCA that stays sparse, `leidenalg` running out of memory on large graphs, and per-batch subsampling when nothing else fits |
| `snippets/infercnvpy_pandas3_patch.py` | infercnvpy 0.6.1 failing under the pandas 3 string dtype (2D fancy-indexing, h5ad writing, worker processes) |
| `snippets/upsetplot_patches.md` | upsetplot 0.9.0 under pandas Copy-on-Write and newer matplotlib |

## Benchmark: centred PCA on a sparse matrix

A common workaround for memory is `sc.tl.pca(..., zero_center=False)`. Measured on a
synthetic log-normalised matrix of 200,000 cells x 2,000 genes (8% non-zero). The matrix is
generated once and saved; each variant runs in its own process that loads it and records the
rise in peak memory during PCA:

| Variant | Extra memory for PCA | Time | Agreement with exact PCA |
|---|---|---|---|
| Centred, sparse, `svd_solver="arpack"` | +187 MB | 17 s | 1.000 |
| Uncentred (`zero_center=False`) | +443 MB | 19 s | 0.899 |
| Centred, densified first | +5,249 MB | 17 s | 1.000 |

Agreement is the overlap between the subspaces of the first 10 components (1 = identical).
With sparse input and the arpack solver, Scanpy centres implicitly and never builds the
dense matrix, so the centred PCA adds under 200 MB and gives the exact result. Skipping
centring saved neither memory nor time here, and it changes the components.
`integrate_sparse.py` therefore keeps the PCA centred. The densified row includes the
1.6 GB dense copy of the matrix. Raw numbers: `benchmarks/results/pca_centering.csv`.

```bash
python benchmarks/pca_centering.py --cells 200000 --genes 2000 --density 0.08
```

Measured on Windows 11, 17 GB RAM, Python 3.13, Scanpy 1.12.4, anndata 0.13.4.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

The tests use small synthetic data and no network. They check that QC removes injected
low-quality and high-mitochondrial cells, that concatenation keeps each value with its
gene, that subsampling caps every batch and is reproducible with a seed, that the sparse
centred PCA matches the dense one, and that the infercnvpy patch converts the dtypes. The
Harmony test is skipped when `harmonypy` is not installed (it did not build on Python 3.13
for Windows at the time of testing).

## Using the snippets

The snippets are standalone functions, not an installable package. Copy what you need and
set the parameters (`N_HVG`, `MAX_CELLS_PER_BATCH` and so on) for your data and memory.

Things to keep in mind:

- Run `qc_filter` per sample. MAD thresholds are computed on whatever object you pass in,
  so running it on a merged atlas mixes the distributions it is meant to separate.
- The mitochondrial prefix is `MT-` (human). Change it for mouse (`mt-`).
- Subsampling per batch removes cells from rare populations first. Check their counts
  after subsampling and report the cap you used.
- The infercnvpy and upsetplot fixes target specific releases (0.6.1 and 0.9.0). Check
  whether a newer release already fixes the issue before applying them.

## License

MIT. See `LICENSE`.
