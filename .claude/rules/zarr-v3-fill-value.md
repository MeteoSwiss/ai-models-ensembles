---
name: zarr-v3-fill-value
description: Zarr v3 templates built via np.empty + to_zarr(compute=False) silently leak zeros (not NaN) into unwritten chunks; fill_value in encoding is ignored. Use dask.array.full(..., NaN). np.isfinite idempotency checks then false-positive every slot.
paths:
  - "ai_models_ensembles/e2s_inference.py"
  - "ai_models_ensembles/e2s_data.py"
  - "tools/reshard_zarr.py"
  - "tools/*reshard*.py"
---

When creating a zarr v3 sharded store as a template for region-writes (one init/time slot at a time),
do NOT use `np.empty(shape, dtype="float32")` as the initial array:

```python
# WRONG -- leaks zeros; fill_value in encoding is ignored
ds["v"] = (dims, np.empty(shape, dtype="float32"))
encoding["v"] = {"chunks": chunks, "fill_value": np.float32("nan")}
ds.to_zarr(path, mode="w-", encoding=encoding, compute=False, zarr_format=3)
# Read back: all 0.0, NOT NaN.
```

xarray writes the np.empty values (zeroed kernel pages on Linux) into the chunks at `to_zarr` time even
with `compute=False`, and the encoding fill_value is then irrelevant. Use `dask.array.full(..., NaN)` --
lazy (no RAM), compute=False writes metadata only, and fill_value=NaN takes effect:

```python
import dask.array as da
nan32 = np.float32("nan")
ds["v"] = (dims, da.full(shape, nan32, dtype="float32", chunks=chunks))
encoding["v"] = {"chunks": chunks, "fill_value": nan32}
ds.to_zarr(path, mode="w-", encoding=encoding, compute=False, zarr_format=3)
```

**Why it matters:** if the idempotent "skip already-written" check is `np.all(np.isfinite(arr))`, zeros
pass and every slot is marked done. That is how the 2026-05-29 IFS perturbed-IC download "SKIPPED"
224/224 inits against a zero-fill template (no MARS requests made). Defense in depth: also test
`np.any(arr)` (reject all-zero - a real 721x1440 geophysical field is never exactly zero everywhere).
Verify any new template: `assert np.all(np.isnan(ds.v.isel(<unwritten slot>).values))`.
