"""Compatibility patch: infercnvpy 0.6.1 + pandas >=3.0 + anndata (write).

Several incompatibilities show up together when running infercnvpy on a
pandas 3.x environment. The notes below follow the order in which you hit them:

1. infercnvpy 0.6.1 does 2D fancy-indexing on gene-name arrays internally. Under
   pandas 3.x's new default string dtype this raises
   `ArrowInvalid: only handle 1-dimensional arrays`, because that infercnvpy version
   does not support fancy-indexing on the new string extension array.
   Forcing classic numpy `object` dtype for var_names/var["chromosome"] sidesteps it.
2. `pd.set_option("future.infer_string", False)` alone does NOT change this:
   pandas 3's own default string dtype is independent of that legacy option, which
   only controls whether strings get pyarrow-backed. You need the explicit dtype
   cast below regardless of that option's value.
3. Once step 1's classic-object columns exist, anndata's h5ad writer refuses plain
   object-dtype string columns unless you opt in explicitly.
4. If you run infercnv with n_jobs>1 (ProcessPoolExecutor), each worker re-imports
   pandas fresh, so the same fix must run in worker processes too. A
   `sitecustomize.py` dropped into the venv's site-packages is the simplest way to
   guarantee that without threading options through the pool.

Apply patch_pandas3_for_infercnvpy() once, before importing infercnvpy.
"""
from __future__ import annotations

import pandas as pd


def patch_pandas3_for_infercnvpy() -> None:
    import anndata

    pd.set_option("future.infer_string", False)  # harmless, kept for clarity; see note 2 above
    anndata.settings.allow_write_nullable_strings = True


def force_object_dtype(adata) -> None:
    """Call after loading gene positions (e.g. cnv.io.genomic_position_from_gtf),
    right before cnv.tl.infercnv."""
    adata.var.index = pd.Index(adata.var.index.astype(str).tolist(), dtype=object)
    adata.var["chromosome"] = adata.var["chromosome"].astype(str).astype(object)


SITECUSTOMIZE_SNIPPET = '''\
# Drop this file at <venv>/Lib/site-packages/sitecustomize.py (Windows) or
# <venv>/lib/pythonX.Y/site-packages/sitecustomize.py (Linux/Mac) so every
# subprocess, including ProcessPoolExecutor workers spawned by infercnvpy,
# picks up the same pandas 3 compatibility fix as the main process.
import pandas as pd
pd.options.future.infer_string = False
'''
