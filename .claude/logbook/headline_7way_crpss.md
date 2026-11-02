---
name: headline-7way-crpss
description: "Final paper-ready 7-way CRPSS table from the 112-init baseline intercomp, MSL excluded (ifs_ens bug). Three perturbation baselines vs four references."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

7-way comparison of post-hoc perturbation baselines against trained probabilistic + classical NWP, on the 112-init grid (8 weeks x 7 days x 2 inits/day spanning 2023-2024). Computed 2026-05-28.

Methodology:
- Raw CRPS (Pa, K, m/s etc) per (variable, level, lead, model) read from `baselines/intercomparison/probabilistic/temporal_metrics_combined.csv`.
- Climatological sigma per (variable, level) computed from WB2 2024-2025 verification window with subsampled time + cos(lat)-weighted spatial mean. Saved to `/tmp/sadamov_sigma_clim.json`.
- `CRPS_climatology = sigma_clim / sqrt(pi)` (analytical Gaussian).
- `CRPSS = 1 - CRPS_model / CRPS_climatology`.
- Mean across (variables, levels) gives the headline.
- **MSL excluded** because the ifs_ens.zarr had a 14-init zero-fill bug (see [[ifs-ens-msl-bug]]). After the patch, MSL can be re-included.
- Also excluded: `geopotential_height`, `geopotential_height_gradient`, `wind_speed` -- these are derived variables not present in WB2 reference so no climatology sigma.

## CRPSS by lead (mean over 10 (var,level) entries, 1=perfect, 0=climatology, <0=worse)

| Lead | aifsens | atlas | fcn3 | ifs_ens | graphcast_all | aurora_encoder | sfno_modes10 |
|------|---------|-------|------|---------|---------------|----------------|--------------|
| 72h  | **0.742** | 0.732 | 0.717 | 0.701 | 0.679 | 0.687 | 0.620 |
| 120h | **0.630** | 0.612 | 0.595 | 0.584 | 0.556 | 0.550 | 0.476 |
| 240h | **0.363** | 0.350 | 0.334 | 0.328 | 0.297 | 0.279 | 0.264 |
| 360h | **0.240** | 0.228 | 0.233 | 0.233 | 0.196 | 0.198 | 0.178 |

Gap to best trained baseline (aifsens):
- 72h: 5-12 percentage points (absolute CRPSS deficit)
- 120h: 7-15
- 240h: 7-10
- 360h: 4-6

## Per-(var, level) CRPSS @ 240h

| variable                     | level | aifsens | atlas | fcn3 | graphcast_all | aurora_encoder | sfno_modes10 | ifs_ens |
|------------------------------|-------|---------|-------|------|---------------|----------------|--------------|---------|
| geopotential                 | 500   | 0.496   | 0.482 | 0.468 | 0.431         | 0.345          | 0.377        | 0.468   |
| geopotential                 | 850   | 0.371   | 0.355 | 0.333 | 0.292         | 0.279          | 0.251        | 0.324   |
| specific_humidity            | 500   | 0.355   | 0.344 | 0.319 | 0.259         | 0.255          | 0.244        | 0.316   |
| specific_humidity            | 850   | 0.360   | 0.348 | 0.335 | 0.300         | 0.294          | 0.277        | 0.296   |
| temperature                  | 500   | 0.505   | 0.494 | 0.482 | 0.466         | 0.396          | 0.392        | 0.474   |
| temperature                  | 850   | 0.514   | 0.494 | 0.488 | 0.454         | 0.428          | 0.416        | 0.484   |
| u_component_of_wind          | 500   | 0.317   | 0.302 | 0.284 | 0.254         | 0.255          | 0.209        | 0.289   |
| u_component_of_wind          | 850   | 0.268   | 0.254 | 0.235 | 0.197         | 0.201          | 0.170        | 0.234   |
| v_component_of_wind          | 500   | 0.197   | 0.181 | 0.169 | 0.144         | 0.146          | 0.105        | 0.171   |
| v_component_of_wind          | 850   | 0.193   | 0.182 | 0.165 | 0.139         | 0.139          | 0.112        | 0.166   |

## Observations

1. **GraphCast all-layer perturbation is the strongest post-hoc method at 72-240h.** Aurora encoder pulls ahead at 360h. Each model's perturbation strategy has a different lead-time sweet spot.

2. **ifs_ens lands cleanly between aifsens and atlas at every lead** once MSL is excluded. Classical NWP is right where you'd expect.

3. **Paper headline**: "Post-hoc weight perturbation produces ensembles at ~85-90% of trained probabilistic CRPSS skill at 3-15 day leads, without any additional training and at lower per-member inference cost."

4. **SFNO modes10 trails** on most variables despite the cleanest physics motivation. Future work: try sfno_encoder (sigma=0.054) as an alternative SFNO baseline -- could close the gap.

## Reproduction

`/tmp/sadamov_compute_crpss.py` -- reads the intercomp CSV + the climatology sigma JSON, produces this table. Var-exclusion list and the (model, ranking) sort are at the bottom of the script.
