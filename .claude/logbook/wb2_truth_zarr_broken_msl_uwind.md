---
name: wb2-truth-zarr-broken-msl-uwind
description: "RESOLVED 2026-06-01: the WB2 'broken metadata' for MSL + u_wind is a `consolidated=False` artefact only. The consolidated .zmetadata path has both vars. Standing rule: ALWAYS open WB2 zarrs with `consolidated=True` (matches SwissClim's existing CRPS/SSR convention). Below is the original symptom write-up; treat as historical."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## TL;DR (post-resolution, 2026-06-01)

The 2 vars are present in the consolidated metadata even though they're absent from the raw zarr hierarchy. So `xr.open_zarr(<path>, consolidated=True)` loads ALL 7 paper vars cleanly from the 2022-2023 + 2024-2025 splits -- no need to fall back to weatherbench2_original.

Standing rule for any new WB2 diagnostic: open with `consolidated=True`.

Patches applied 2026-06-01:
- `tools/spatial_mean_ssr.py`: truth open switched to `consolidated=True` (with explanatory comment). 6 Tier 1b SSR jobs relaunched (2452169-74) to repopulate MSL + u_wind columns that were previously NaN.
- `tools/plot_7way_spatial_mean_ssr.py`: restored 7-var 2x4 layout (2m_t, MSL, gpot, t, u, v, q) -- 8th panel is the legend.
- `/iopsstor/scratch/cscs/sadamov/crpss_ablation.py` (ablation CRPSS climatology): WB2 2022-2023 + 2024-2025 with `consolidated=True`.

## Original symptom (historical)

## Symptom

When `tools/spatial_mean_ssr.py` opens `weatherbench2_2022_2023.zarr` and accesses `truth["mean_sea_level_pressure"]` or `truth["u_component_of_wind"]`:

- `xr.open_zarr(..., consolidated=False)` raises a `ZarrUserWarning: Object at <varname> is not recognized as a component of a Zarr hierarchy`.
- The variable does NOT appear in `truth.data_vars` -- so `truth[variable]` returns a coordinate-like NaN object (or raises). The script's `t_scalar = _weighted_spatial_mean(ta, lat)` returns NaN.
- Downstream the CSV ends up with `spread` computed correctly (from forecast.zarr) but `error = NaN`, `ssr = NaN`.

All other vars work: `2m_temperature`, `geopotential`, `temperature`, `v_component_of_wind`, `specific_humidity` are fine in both 2022-2023 and 2024-2025 splits.

## Root cause

The 2 affected vars have chunk directories under the zarr root (`<store>/mean_sea_level_pressure/<chunk>` files exist) but the variable-level `zarr.json` / `.zarray` metadata is malformed or missing. zarr-python skips them silently with a `ZarrUserWarning` rather than raising. Verified 2026-06-01 with `xr.open_zarr(path, consolidated=False)` -- the warning fires twice, and the two vars are absent from `data_vars`.

The 2024-2025 zarr is similarly broken for the same two vars.

## Alternate truth zarrs we checked

- `/capstor/store/cscs/swissai/weatherbench/weatherbench2_original`: HAS MSL + u_wind working but time range is **1959-2021 ONLY** -- does not cover our 2022-2024 forecast inits.
- `/capstor/store/cscs/swissai/weatherbench/weatherbench2_2019_2020.zarr3`: time 2019-2020, has neither var.
- `/capstor/store/cscs/swissai/weatherbench/weatherbench2_additionalvariables.zarr`: opens with `ValueError: conflicting sizes for dimension 'time'`.
- Various other zarrs in `/capstor/store/cscs/swissai/weatherbench/`: none cover 2022-2024 with both MSL + u_wind working.

So for any diagnostic that needs ERA5 truth in our forecast date range, you have 5 working vars only:
`2m_temperature, geopotential, temperature, v_component_of_wind, specific_humidity`.

## Workarounds

1. **Drop MSL + u_wind from the plot panel set.** Done 2026-06-01 in `tools/plot_7way_spatial_mean_ssr.py` -- VARS_PLOT now has 5 vars + last panel turned off via `ax.axis("off")`.
2. **Compute wind_speed from u + v** (v_component_of_wind works) if you need a "wind" panel.
3. **Fix the WB2 zarr metadata** -- would require re-loading those 2 vars from elsewhere (Google Cloud bucket ERA5? CDS?) and re-shipping the zarr. Not on critical path for the paper.

## Note for evaluate_baselines.sh / SwissClim

This bug bites `tools/spatial_mean_ssr.py` because it does explicit `truth[variable]` lookup. SwissClim's main CRPS / SSR / etc. modules use `consolidated=True` (per [[ifs-ens-msl-bug]] memo), which masks the bug -- they get values for MSL via the consolidated metadata path. So the existing baseline + ablation intercomp CSVs are fine; this only affects the standalone spatial-mean SSR script that opens the zarr `consolidated=False`.

If you write a new diagnostic against WB2 truth, default to `consolidated=True` to be safe.
