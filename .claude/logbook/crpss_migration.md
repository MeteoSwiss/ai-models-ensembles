---
name: crpss-migration
description: "2026-06-01 evening: Tab. 3 ablation calibration table being migrated from CRPS to CRPSS because the variable-mean CRPS is dominated by geopotential (different units across vars). CRPSS = 1 - CRPS / CRPS_clim normalises each (var, level) so 7 vars contribute equally. Climatology from WB2 2022-2023 + 2024-2025 (consolidated=True), per-pixel temporal std then cos-weighted spatial mean. WORK IN FLIGHT."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why CRPSS (the user caught this)

The Tab. 3 "CRPS" column was nominally variable-mean across 8 paper-headline vars, but the raw CRPS values are in mixed units (geopotential ~50-100 m^2/s^2, temperature ~K, specific humidity ~0.0001 kg/kg). The arithmetic mean is dominated >90% by geopotential, so "variable-mean CRPS" is effectively just geopotential CRPS with rounding noise from the others.

Within-model rankings could in principle be misleading: if a phase has lower humidity CRPS but higher geopotential CRPS, the variable-mean would still favour the geopotential-better phase even if it loses on most variables. The user confirms they've eyeballed enough cross-phase plots that the picks "probably hold" but wants the table to be honest about it.

## Methodology (matches `headline_7way_crpss.md`)

1. **Climatology sigma** per (var, level): WB2 2022-2023 + 2024-2025 (consolidated=True), 4-step time subsample, per-pixel temporal stddev, then cos(lat)-weighted spatial mean. Saved to `/iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json`.
2. **CRPS_clim** = sigma_clim / sqrt(pi) (analytical Gaussian).
3. **Per-(var, level) CRPSS** = 1 - CRPS / CRPS_clim using the existing per-baseline CRPS from `crps_summary_combined.csv`.
4. **Per-variable CRPSS** = mean over (500, 850) levels for 3D vars, single value for 2D vars (2m_t, MSL). This treats each variable as one unit regardless of how many levels we have for it.
5. **Table cell** = mean across 7 variables (2m_t, MSL, gpot, t, u, v, q) -- equal weight per variable.

This gives a unitless 0-1 score per cell where 1 = perfect, 0 = climatology, <0 = worse than climatology.

## What changes

- Tab. 3 ([figures/calibration_basis_table.tex](../../../../pyprojects/ai-models-ensembles/figures/calibration_basis_table.tex)): replace the CRPS 4-lead block with CRPSS. Direction flips (HIGHER is better, argmax bolds). Caption updated to drop the geopotential-dominance footgun.
- Tab. 1 ablation grid (experiments_tables.tex): the bolded "winner" within each phase is currently argmin CRPS @ 240h. If any winner changes under CRPSS, the bolding has to move. **The user wants to be told if this happens.**
- Tab. 4 inference cost table: no CRPS values, no change needed (only `AIFS-CRPS` / `ATLAS-CRPS` model name acronyms appear, which stay).

The headline 7-way baseline table (`headline_7way_crpss.md`) already uses CRPSS, so it's untouched.

## Status: COMPLETE 2026-06-02 ~11:00 (commit `901a039`)

- 4 OOM attempts (2454183 / 2454331 / 2454580 / `chunks={"level":1}` fix) before landing on the right chunking
- Root cause of the OOMs: WB2 stores geopotential with chunks=(time=24, level=full=37, lat=full, lon=full) ~ 3.7 GB per chunk. `.sel(level=500).values` materialised all 37 levels into 64 dask workers in parallel; peak >200 GB, busting 800G
- Fix: open zarr with `chunks={"time": 100, "level": 1, ...}` so `.sel(level=500)` reads a single-level chunk (~414 MB)
- Job 2454617 COMPLETED 50:23
- All 4 production picks REMAIN within-model argmax CRPSS at lead 240h
- Tab. 3 first block updated CRPS -> CRPSS, bolds recomputed for argmax direction, caption rewritten
- Tab. 1 caption updated: "Phase 3 hook beats Phase 3b on CRPSS@240h" with new numbers
- Outputs: `/iopsstor/scratch/cscs/sadamov/{sigma_clim_ablation.json, crpss_ablation_table3.csv}` -- both safe to keep for further analysis

## Status history (2026-06-01 evening, in flight)

- Climatology script: `/iopsstor/scratch/cscs/sadamov/crpss_ablation.py` (PID 250866 at time of writing) is computing the 12 sigma_clim values (2 2D vars + 5 3D vars at 2 levels). 2m_t done (sigma = 4.53), MSL in progress. Each var takes a few minutes due to per-pixel std across ~7800 daily timesteps at 0.25 deg.
- Once it finishes:
  - Output `/iopsstor/scratch/cscs/sadamov/crpss_ablation_table3.csv` has per (model_phase, lead) mean CRPSS plus per-variable detail columns.
  - I need to compare new (CRPSS-based) ranking vs old (CRPS-based) ranking for each phase per model to flag any pick changes.
  - Then edit calibration_basis_table.tex: replace the CRPS block values, header, caption, bold direction.
  - If any per-phase winner changes (e.g. Aurora Phase 2 mag_0.025 was bold for CRPS@24 -- might shift), update Tab. 1 bolds and TELL THE USER.

## Subtle averaging detail to NOT mess up

For 3D vars we average the 500 and 850 CRPSS values first (per-variable), THEN take the cross-variable mean. NOT per-(var, level) entry mean (which would double-count 3D vars). With 5 3D vars at 2 levels + 2 2D vars, per-entry averaging would weight 3D vars 2x = 10/12 of the mean. Per-variable averaging gives each of the 7 vars 1/7 weight. The script does the right thing -- noted here so I don't regress.

## Scripts

- Climatology + CRPSS: `/iopsstor/scratch/cscs/sadamov/crpss_ablation.py`
- Sigma cache: `/iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json`
- Output table: `/iopsstor/scratch/cscs/sadamov/crpss_ablation_table3.csv`

## Related

- [[wb2-truth-zarr-broken-msl-uwind]] -- the consolidated=True fix that enables MSL + u_wind in the climatology.
- [[headline-7way-crpss]] -- the 7-way baseline CRPSS table, methodology source.
- [[calibration-winners]] -- the production-baseline picks, which Tab. 3 documents. Picks themselves shouldn't change (multi-metric criterion has 5 other scale-invariant metrics) but the table presentation does.
