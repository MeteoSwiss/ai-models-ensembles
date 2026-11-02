---
name: paper-full-surface-vars-goal
description: "Final-paper goal -- include the full surface variable set we have (T2m, MSL, 10m_u, 10m_v, 100m_u, 100m_v) in the headline CRPSS variable-mean, not just T2m + MSL. IFS-ENS NaN gaps are OK; skipna handles it."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Headline variable mean: full surface set (final-paper target)

User said 2026-06-09: ultimately we want the **full surface variable set we have** in the headline CRPSS variable mean, not just T2m + MSL. IFS-ENS missing values are acceptable -- skipna averaging handles it.

## Current state (2026-06-09)

`tools/plot_headline_crpss_vs_lead_8way.py` and `tools/headline_8way_table.py`:
```python
VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = ["geopotential", "temperature", "u_component_of_wind",
           "v_component_of_wind", "specific_humidity"]
```

7 paper vars: T2m + MSL + Z/T/u/v/q at 500+850 hPa.

## Final-paper target

Add the surface vars we have CRPS CSVs for:
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`

(and possibly 100m winds if/when they're in the CSV outputs).

The CSV files exist for these in each baseline's `eval/probabilistic/`:
```
crps_line_10m_u_component_of_wind_by_lead_ensprob.csv
crps_line_10m_v_component_of_wind_by_lead_ensprob.csv
```

These follow the same SwissClim naming convention so `_load_perbase` will pick them up unchanged once they're added to `VARS_2D`.

## What to do when picking this up

1. Verify the 10m_u / 10m_v CRPS CSVs exist for all baselines listed in MODELS (especially the combined-zarr ones aifsens, atlas, fcn3, ifs_ens via `temporal_metrics_combined.csv`).
2. Add them to `VARS_2D` in both `plot_headline_crpss_vs_lead_8way.py` and `headline_8way_table.py`.
3. Add a sigma_clim entry for them in `sigma_clim_1990_2019.json` if it's not already there (check first; the 1990-2019 file has only the 7 paper vars currently).
   - If sigma_clim is missing for `10m_u_component_of_wind` / `10m_v_component_of_wind`, the clim recompute (`tools/compute_climatology_1990_2019.py`) needs the surface var list extended. ~12h slurm job per added variable on the production-grid clim window.
4. Regen the headline table + figure.
5. Update the caption: "7 paper variables" -> "9 paper variables" (or whatever the final count is), expand the variable list.
6. Verify the IFS-ENS row is still defensible -- check the WB2 NaN audit for 10m_u/v (per [[ifs-ens-10m-wind-nan-bug]] it's ~20% NaN for those too, so the skipna pattern applies).

## Why not now?

Current scope is mid-Phase-5 (3 of 4 perturbed baselines still evaluating). The headline figure has been through many iterations today already. Adding 10m winds now would invalidate every CRPSS-vs-lead comparison we've staged in the paper. Defer to a clean post-Phase-5 revision round.

## Related

- [[ifs-ens-10m-wind-nan-bug]] -- WB2 NaN audit for IFS-ENS surface vars
- [[paper-revision-round-todos]] -- the existing revision-round queue (add this there too)
- [[clim1990-crps-perf-fix]] -- the 30-yr clim recompute pipeline (would need extension if sigma_clim for 10m winds is missing)
