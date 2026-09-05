# `upsetplot` 0.9.0 patches for pandas Copy-on-Write + newer matplotlib

Two small patches to `upsetplot/plotting.py` (site-packages), needed together on
recent pandas (Copy-on-Write enabled) and matplotlib:

1. **`fillna(inplace=True)` on a copy silently no-ops under pandas CoW.**
   `upsetplot` calls `.fillna(0, inplace=True)` on an intermediate DataFrame that,
   under Copy-on-Write, is already a copy, so the fill never reaches the object
   the rest of the function reads from. Replace the in-place call with a
   reassignment: `df = df.fillna(0)`.

2. **`np.diff()` on a 2-element array no longer returns a bare scalar.**
   A newer matplotlib/numpy pairing returns a length-1 array from `np.diff` where
   `upsetplot` expects a scalar for a spacing calculation, breaking indexing
   downstream. Wrap the call: `float(np.diff(...))` (or index `[0]`) at that call
   site.

Both are one-line fixes at the exact call sites `pip show -f upsetplot` points you
to; no patch file is included here since line numbers vary by release: search
`plotting.py` for `fillna(` and `np.diff(` respectively.
