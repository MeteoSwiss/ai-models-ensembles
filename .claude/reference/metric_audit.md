---
name: metric-audit
description: "Canonical definitions and optimal values for every probabilistic / multivariate / structural metric the paper uses. Splits cleanly into ABLATION (4 inits, lead 240 h, Tab. 1/3) and BASELINE (112 inits, lead 240 h with 360 h supplements, headline Tab. 2 + Tab. 4 + ES/VS supplement). Includes the spatial-mean-vs-pointwise SSR dichotomy that is the central perturbation-baseline limitation."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why this memo exists

I (Claude) repeatedly drifted on (a) which metric is optimal at what value, (b) which grid (ablation 4-init vs baseline 112-init) feeds which table cell, and (c) which CSV / eval module each number comes from. The user caught it multiple times -- most importantly on SSR (optimal = 1 for BOTH per-pixel and spatial-mean, not 0). This file is the single source of truth. Update it when a metric is added or a CSV path moves.

## Two grids, never confuse them

| Grid name | n_inits | Init dates | n_members | Lead times scored | Used by |
|---|---|---|---|---|---|
| **Ablation grid** | 4 | 2023-05-15, 2023-08-15, 2024-02-15, 2024-11-15 (mid-season) | 10 | 240 h headline; some modules also 24/48/120 h | Tab. 1 ablation phase grid, Tab. 3 calibration-basis table, Phase 6 fresh-vs-frozen, Phase 3-bis mode-skip |
| **Baseline grid** | 112 | weekly 2024-01..2024-12 (production protocol) | 10 | 240 h headline; some leads 24/72/120/360 h | Tab. 2 headline 7-way, Tab. 4 inference cost, ES + VS supplement, spatial-mean SSR bias analysis |

**Tab. 3 = ABLATION grid (4 inits, lead 240 h).** Not the production grid. Within-phase argmax / argmin is over those 4 inits only.

**Fresh-vs-frozen and mode-skip experiments = ABLATION grid (deliberately).** They reuse the 4 inits so they can be compared 1:1 with the ablation phases without re-running everything at 112 inits.

## Metrics: definition, optimal value, where the number comes from

### SSR -- per-pixel (Spread-Skill Ratio, pointwise)

```
SSR_pixel = mean_pixel( sigma_ens(x) / RMSE_to_truth(x) )
```

- **Optimal value: 1** (not 0). SSR < 1 = under-dispersive (over-confident); SSR > 1 = over-dispersive.
- Direction of "better": minimise `|SSR - 1|`.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/probabilistic/ssr_summary*ensprob.csv`
- Baseline CSV: `$STORE/baselines/<baseline>/eval/probabilistic/ssr_summary*ensprob.csv` (exists for all 6 baselines: aurora_encoder, graphcast_all, sfno_modes10, aifsens, fcn3, atlas)
- Eval module: SwissClim `probabilistic` -> `ssr` (per-pixel branch)
- Tab. 3 column: "SSR" (single number per (phase, model), averaged across 7 paper vars, lead 240 h)

### SSR-spatial (Spread-Skill Ratio, spatial-mean)

```
SSR_spatial = stddev_across_members( spatial_mean(member_field) )
              / RMSE_to_truth( spatial_mean(ensemble_mean) )
```

i.e. first cos(lat)-weighted spatial-mean each member field, then compute spread vs RMSE on those scalars per init, then aggregate across inits.

- **Optimal value: 1.** Same as per-pixel. The user explicitly corrected me on this: both SSR variants are calibrated at 1, not 0.
- Direction of "better": minimise `|SSR_spatial - 1|`.
- Baseline CSV: `$STORE/baselines/<baseline>/spatial_mean_ssr/*.csv`
- Tool: `tools/spatial_mean_ssr.py` (standalone, not part of the SwissClim eval module)
- Not in Tab. 3 main bundle; reported as a separate paper limitation paragraph + figure.

### CRPS (Continuous Ranked Probability Score, per variable, per pixel)

```
CRPS(F, y) = E_F|X - y| - (1/2) E_F|X - X'|     # per pixel
-> cos(lat)-weighted spatial mean per variable
```

- **Optimal value: 0**. Lower is better.
- **Mixed units across variables** (geopotential ~10^4, temperature ~K, humidity ~10^-4). NEVER report a naive cross-variable mean -- it is ~95% geopotential. Always either (a) per-variable CRPS columns, or (b) CRPSS for cross-variable aggregation.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/probabilistic/crps_summary_combined*ensprob.csv`
- Baseline CSV: `$STORE/baselines/<baseline>/eval/probabilistic/crps_summary_combined*ensprob.csv`
- Eval module: SwissClim `probabilistic` -> `crps`

### CRPSS (CRPS Skill Score) -- the cross-variable aggregate

```
CRPSS = 1 - CRPS / CRPS_clim
CRPS_clim = fair CRPS of the WB2 probabilistic climatology (1990-2019 years as members) vs eval truth, lead-resolved
```

- **Optimal value: 1**. 0 = climatology. <0 = worse than climatology.
- Direction: higher is better. Tab. 3 argmax bolding.
- Per-variable CRPSS = mean over (500, 850) for 3D vars, single value for 2D vars. Cross-variable CRPSS = mean across the 7 paper vars (equal weight per variable).
- Denominator (current, since 2026-06-12): exact WB2 probabilistic climatology, NOT the old analytical Gaussian `sigma_clim/sqrt(pi)` (removed). Precomputed denominators in `tools/data/crps_clim_eval_{1990_2019,ablation_1990_2019}.json`; recompute via `tools/compute_climatology_crps_vs_eval.py --grid {production,ablation}`.
- Tab. 3 column: "CRPSS" (replaces the old CRPS column, commit 901a039)
- See [[crpss-migration]] (`.claude/logbook/2026-06-12_wb2_crpss_climatology_migration.md`) for the migration history and why the Gaussian denominator was retired.

### Energy Score (multivariate over (var, level))

```
ES(F, y) = E_F||X - y|| - (1/2) E_F||X - X'||
norm runs over the (var, level) axis at each pixel; cos(lat) spatial mean
```

- **Optimal value: 0**. Lower is better. Strictly proper.
- Normalisation: ens + truth divided per (var, level) layer by the per-init truth std before computing the norm (otherwise geopotential dominates -- same scale problem as raw CRPS).
- Baseline output: `$STORE/baselines/<baseline>/energy_variogram_score/score.csv`
- Tool: `tools/energy_variogram_score.py`
- Used in Tab. 3 footnote / supplementary cell, **baseline grid (112 inits) only, lead 240 h** (not in the ablation bundle).
- See [[energy-variogram-score]] for the math and baseline list.

### Variogram Score VS_0.5

```
VS_p(F, y) = sum_{(i,j) in S} [ |y_i - y_j|^p - (1/M) sum_m |X_m,i - X_m,j|^p ]^2
```

p = 0.5 (Scheuerer & Hamill 2015 default), N = 5000 random pixel pairs per variable, mean across the 12 (var, level) layers.

- **Optimal value: 0**. Lower is better.
- Same CSV / tool / grid as Energy Score above.

### SSIM (Structural Similarity Index)

- **Optimal value: 1**. Higher is better.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/deterministic/ssim*.csv`
- Computed on ensemble mean vs truth per variable, then averaged across the 7 paper vars.
- Tab. 3 column: "SSIM".

### LSD (Log Spectral Distance)

```
LSD(f, g) = sqrt( mean_k ( 10 log10 P_f(k) / P_g(k) )^2 )
```

- **Optimal value: 0**. Lower is better.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/energy_spectra/lsd*.csv`
- Bug to watch: see [[swissclim-lsd-pooled-bug]] -- LSD per band silently uses ensemble-mean instead of pooled. Affects 7-way intercomp plots, but the Tab. 3 cell still reflects what the tool reports.
- Tab. 3 column: "LSD".

### FSS (Fractions Skill Score)

- **Optimal value: 1**. Higher is better.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/fss/*.csv`
- Tab. 3 column: "FSS" (averaged across thresholds and scales reported in the table).

### W1 (Wasserstein-1)

- **Optimal value: 0**. Lower is better.
- Ablation CSV: `$STORE/ablation/phase{N}/<model>/eval/<run_tag>/wd_kde/*.csv`
- Note: SwissClim `wd_kde` intercomparison module is broken for n_models > 7 (fixed-length colors list); affects plots only, not the table number.

## Tab. 3 cell composition (ablation grid, lead 240 h, 4 inits)

For each (phase, model) cell in `figures/calibration_basis_table.tex`:

| Column | Metric | Optimal | Aggregation |
|---|---|---|---|
| CRPSS | CRPS skill score | max (= 1) | mean over 7 paper vars (per-var = mean over 500/850 for 3D) |
| SSR | per-pixel SSR | nearest 1 | mean over 7 paper vars |
| SSIM | structural similarity | max (= 1) | mean over 7 paper vars |
| LSD | log spectral distance | min (= 0) | mean over 7 paper vars |
| FSS | fractions skill | max (= 1) | mean over thresholds + scales |
| W1 | Wasserstein-1 | min (= 0) | mean over 7 paper vars |

Bolds: argmax within each phase block when "higher is better", argmin when "lower is better". Caption explicitly states the bold direction per column.

## The spatial-mean vs per-pixel SSR dichotomy (paper limitation)

This is the central perturbation-baseline finding and the reason Phase 6 (fresh-per-step weights) and the IC-perturbation hybrid (Phase 5) exist as proposed remedies.

**Mechanism.** Weight perturbation applied once at member init produces a coherent whole-field shift in each member's forecast -- mass/energy of the perturbed weight propagates to the same large-scale pattern across the entire domain. Per-pixel spread looks fine (every pixel sees some local variability across members), but the spatial mean of each member is biased by a near-constant offset, so the across-member spread of spatial means inflates well past the across-member RMSE of the spatial-mean ensemble mean.

**Affected baselines (lead 240 h, 112 inits, baseline grid):**

| Baseline | per-pixel SSR | spatial-mean SSR | Interpretation |
|---|---|---|---|
| graphcast_all | 0.940 | **3.38** | Near-perfect pointwise; ~3x over-dispersive globally. Heavily biased members under spatial averaging. |
| aurora_encoder | 0.823 | **1.65** | Mildly under-dispersive pointwise; ~1.65x over-dispersive globally. Heavily biased members under spatial averaging. |
| sfno_modes10 | 0.851 | 0.70 | Under-dispersive on BOTH axes. Different mechanism -- does NOT exhibit the perturbed-baseline coherent-shift bias. |
| aifsens (trained prob) | 0.842 | 0.65 | Trained probabilistic; under-dispersive on both, no spatial bias. |
| fcn3 (trained prob) | 0.896 | 0.79 | Best joint calibration; mild under-dispersion both axes. |
| atlas (trained prob) | 0.861 | 1.49 | Slightly under pointwise; over-dispersive globally. |

**Wording for the paper.** The bias finding applies specifically to **graphcast_all and aurora_encoder** -- they show heavily biased members under spatial averaging. sfno_modes10's spectral-mode mode-cut design (l < 10 only) deliberately avoids the global-constant l=0 mode in the K=1 ablation variant ([[phase3bis-sfno-mode-skip]]) and shows no such bias. Do NOT lump sfno_modes10 in with the other two when describing the spatial-mean limitation.

**Data location.**
- Per-pixel SSR: `$STORE/baselines/<baseline>/eval/probabilistic/ssr_summary*ensprob.csv` (all 6)
- Spatial-mean SSR: `$STORE/baselines/<baseline>/spatial_mean_ssr/*.csv`
- See [[spatial-mean-vs-pointwise-ssr]] for the original finding.

## Grid alignment (aurora is 720 lat, others 721)

All eval configs and the ESVS tool point at the **same** WB2 truth zarrs (`weatherbench2_2022_2023.zarr` + `weatherbench2_2024_2025.zarr`), 721 lat from +90 to -90. No per-model truth subset.

- aurora forecasts: `(720, 1440)`, +90.0 down to -89.75 (drops the -90 pole row).
- graphcast / sfno / aifsens / fcn3 / atlas: `(721, 1440)`, full +90 to -90.

SwissClim eval modules use xarray algebra (`forecast - truth`, etc.). xarray binary ops default to `join='inner'`, which silently drops the -90 row from truth when scoring aurora -- score is computed on the 720-row intersection aurora actually predicts. **No SwissClim metric was ever broken for aurora.**

The standalone `tools/energy_variogram_score.py` originally dropped to numpy via `.values` BEFORE alignment, so it lost the auto-join and crashed on aurora (job 2455433, ValueError on broadcast `(10,12,720,1440) (1,12,721,1440)`). Fixed via explicit `ta = ta.sel(latitude=fa.latitude, method='nearest')` before `.values`. Aurora's lat values are an exact subset of ERA5's (both 0.25 deg starting at 90), so `method='nearest'` resolves to exact matches -- mathematically identical to xarray's inner-join.

**Practical consequence.** For any cos(lat)-weighted metric (CRPS / SSR / SSIM / LSD spatial means), the dropped row at -90 has weight `cos(-90 deg) = 0`. Aurora's scores are mathematically identical to what they would be if aurora predicted -90 too. For un-cos-weighted metrics (FSS over equal-area patches, raw VS pair sums) the spatial domain is 720/721 = 99.86% of the others -- far below noise.

Going forward: anywhere a tool bypasses xarray algebra and drops to numpy, it must align grids first. The `method='nearest'` pattern is the safe replication of xarray's inner-join when forecast lat is an exact subset of truth lat.

## Variable list (7 paper vars)

```
2m_temperature, mean_sea_level_pressure, geopotential, temperature,
u_component_of_wind, v_component_of_wind, specific_humidity
```

3D vars (gpot, t, u, v, q) at levels 500 + 850 hPa. 2D vars (2m_t, MSL) single field. Total of 12 (var, level) layers for multivariate metrics, 7 variables for per-variable aggregation.

## Common pitfalls (caught at least once this project)

1. **Naive cross-variable CRPS mean = geopotential CRPS** -- always use CRPSS for cross-variable aggregation.
2. **SSR optimal = 1, NOT 0.** Both per-pixel and spatial-mean variants. Minimise `|SSR - 1|`, do not minimise SSR.
3. **Tab. 3 is the ablation grid (4 inits, lead 240 h), not the production grid.** Headline 7-way numbers are a separate table on the 112-init grid.
4. **Fresh-vs-frozen and mode-skip experiments stay on the 4-init ablation grid by design** -- they exist to be comparable to the ablation phase blocks, not to the production baseline.
5. **WB2 climatology zarr requires `chunks={"time": 100, "level": 1, ...}`** -- otherwise `.sel(level=500)` reads all 37 levels per chunk and OOMs at 800 GB. See [[crpss-migration]] for the four OOM history.
6. **MSL + u_wind require `consolidated=True`** on the WB2 truth zarrs -- otherwise these vars are silently absent from data_vars. See [[wb2-truth-zarr-broken-msl-uwind]].
7. **Spatial-mean SSR bias finding applies to graphcast_all + aurora_encoder only**, NOT sfno_modes10 -- different mechanism (mode-cut excludes l=0).

## Related

- [[crpss-migration]] -- why Tab. 3 moved from CRPS to CRPSS.
- [[headline-7way-crpss]] -- the production-grid 7-way CRPSS table.
- [[spatial-mean-vs-pointwise-ssr]] -- the bias finding in detail.
- [[energy-variogram-score]] -- ES + VS supplement to Tab. 3.
- [[calibration-winners]] -- the production-baseline pick logic.
- [[phase3bis-sfno-mode-skip]] -- mode-skip K=1 variant testing the l=0 coherent-mode hypothesis.
- [[phase6-fresh-per-step-weight]] -- the fresh-per-step random-walk remedy for the spatial-mean bias.
