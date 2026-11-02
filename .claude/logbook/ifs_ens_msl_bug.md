---
name: ifs-ens-msl-bug
description: 14 inits (2024-07-02..08) in ifs_ens.zarr had mean_sea_level_pressure = 0 (failed Google Cloud Storage download). Patched in-place 2026-05-28 to NaN so NaN-aware aggregation skips them.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

`/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr` had a data quality bug in the `mean_sea_level_pressure` variable for **14 consecutive inits** (indices 84-97, valid times **2024-07-02T00 through 2024-07-08T12**). The entire (init x 50 ens x 61 leads x 721 x 1440) slab was filled with zeros instead of valid pressure data -- a failed Google Cloud Storage download where the pipeline wrote zeros instead of NaN/raising. Other variables (2m_temperature, u_component_of_wind, 10m_u_wind) had a partial NaN overlap on these inits but no zeros.

Symptom in eval: ifs_ens var-averaged CRPS @ 240h came out at 766 (vs ~38 for AI baselines). The ~14000 Pa per-variable CRPS for MSL alone dragged the mean up.

Math check:
```
expected per-init MSL CRPS (valid inits)   ≈   250 Pa
zero-init contribution                      = 101000 Pa (forecasting 0 vs reality 101000)
weighted mean over 112 inits = (83 × 250 + 14 × 101000) / 97 = 14790 Pa  ✓ matches observed 14805
```

**Why: other variables were fine.** Investigation 2026-05-28 traced this to the ifs_ens model-side zarr, NOT the WB2 reference. WB2 needs `consolidated=True` to open MSL correctly; with that flag both WB2 zarrs (2022_2023 and 2024_2025) work fine.

**Fix applied (option 2, in-place 2026-05-28)**: ran a python script that loads each affected init slice, applies `where(msl != 0)` to mask zeros to NaN, writes back via `to_zarr(region={init_time: slice(i, i+1)})`. Only the MSL variable was touched; only the 14 bad inits. Verification confirms inits 84-97 now have NaN_frac=1.0 at lead 240h. Other inits untouched.

How to apply: if you see similar all-zero contamination on a model zarr from a cloud-download pipeline, the safe in-place fix is:
1. Per-init load + `where(var != 0)` + region-write with `drop_vars` on coords without the region dim.
2. Verify NaN_frac for the patched inits, then verify the other inits are untouched.
3. Tell the upstream data pipeline about the bug so the next ingest doesn't repeat it.

After this patch the headline 7-way intercomp can be re-aggregated and ifs_ens MSL drops back to a sensible ~250 Pa CRPS, putting the overall ifs_ens var-averaged CRPS back where it should be (right between aifsens and the other trained baselines).
