---
name: energy-variogram-score
description: "2026-06-02 afternoon: added two strictly proper multivariate scoring rules (Energy Score + Variogram Score) at lead 240 h to complement the per-variable Tab. 3 bundle. ES tests joint inter-variable calibration at each location; VS tests spatial correlation structure. Tool at tools/energy_variogram_score.py, 6 baselines fired as jobs 2455433-38."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why a multivariate score

The Tab. 3 bundle (CRPSS / SSR / SSIM / LSD / FSS / W$_1$) is **per-variable** by construction. CRPSS is strictly proper but univariate; everything else is single-aspect (calibration, structure, scale, sharpness, or distributional). None of them score the JOINT distribution across (variable, level) at the same location, or across pixels for one variable.

A model can have:
- per-pixel CRPSS at optimum,
- SSR exactly 1,
- ideal LSD spectrum,
- correct FSS at all thresholds,
- AND still be wrong on the (u, v) wind-vector joint distribution, violate geostrophic balance, or have ensemble members that span the right per-variable range in unphysical combinations.

The existing SwissClim `multivariate` eval module covers this **qualitatively** via bivariate KDE plots. ES and VS give the same thing as scalars.

## Definitions

**Energy Score** (Gneiting & Raftery 2007, strictly proper multivariate generalisation of CRPS):

```
ES(F, y) = E_F ||X - y|| - (1/2) E_F ||X - X'||
```

with the norm `||·||` over the (variable, level) axis at each pixel; then cos(lat)-weighted spatial mean of the per-pixel score. Lower is better.

**Variogram Score** of order p (Scheuerer & Hamill 2015, sensitive to spatial dependence):

```
VS_p(F, y) = sum_{(i,j) in S} [ |y_i - y_j|^p - (1/M) sum_m |X_m,i - X_m,j|^p ]^2
```

Per variable, over N=5000 random pixel pairs, p=0.5 (the Scheuerer&Hamill default); then mean across the 12 (var, level) layers. Lower is better.

## Critical normalisation

Both scores work in a multivariate space where vars are on vastly different physical scales (geopotential ~10^4 m^2/s^2, temperature ~K, humidity ~10^-4 kg/kg). Without normalisation, geopotential would dominate the multivariate norm exactly as it dominated raw CRPS in Tab. 3 before the CRPSS migration ([[crpss-migration]]).

**The tool divides ens + truth per layer by the per-init truth field std** before computing ES or VS. This makes the multivariate norm scale-equivariant and gives equal weight to each (var, level) layer. The exact normalisation choice doesn't affect the relative ranking across baselines, only the absolute scale of the scores.

## Variables + grid

Same 7 paper-headline vars as Tab. 3: `2m_temperature, mean_sea_level_pressure, geopotential, temperature, u_component_of_wind, v_component_of_wind, specific_humidity`. 3D vars at levels 500 + 850 hPa. 5 3D vars x 2 levels + 2 2D vars = 12 layers per pixel for the multivariate ES; same 12 layers averaged for VS.

Lead time: 240 h only. Single-lead snapshot is the right scope for a supplementary table; the Tab. 3 bundle covers the multi-lead picture.

## Baselines + scope

6 baselines fired 2026-06-02 ~13:00 as jobs 2455433-38:

| Job | Baseline | Family |
|---|---|---|
| 2455433 | aurora_encoder | perturbed |
| 2455434 | graphcast_all | perturbed |
| 2455435 | sfno_modes10 | perturbed |
| 2455436 | aifsens | trained prob |
| 2455437 | fcn3 | trained prob |
| 2455438 | atlas | trained prob |

Skipped:
- `aifs_perturbed` -- pre-download still in flight ([[aifs-predownload-cache]]). Add once it lands.
- `ifs_ens` -- 51-member format mismatches the 10-member layout the tool expects. Different code path; deferred.

Each baseline reads 112 forecast.zarr files at lead 240 h, computes ES + VS per init, then means across inits. ~20-30 min per baseline.

## Output

`$STORE/baselines/<baseline>/energy_variogram_score/score.csv` with one row per (score, lead). Aggregated headline goes to a supplementary Tab. 3 footnote once results land.

## Implementation

`tools/energy_variogram_score.py`. CLI mirrors `tools/spatial_mean_ssr.py`. Wrapper at `/capstor/store/cscs/mch/s83/sadamov/wrappers/energy_variogram_7way.sh`. Submit script: `/iopsstor/scratch/cscs/sadamov/esvs_7way_submit.sh`.

## Grid alignment fix (2026-06-02 afternoon)

Initial 6-baseline batch had **aurora_encoder fail** in 74 s (job 2455433) because aurora outputs on `(720, 1440)` (drops the -90 pole row) while truth is `(721, 1440)`. The tool dropped to numpy via `.values` before alignment, so it lost xarray's `join='inner'` auto-alignment and broadcast crashed.

Fix in `tools/energy_variogram_score.py:_stack_var_level`: explicit `ta = ta.sel(latitude=fa.latitude, method='nearest')` before `.values`. Aurora's lat values are an exact subset of ERA5's (both 0.25 deg starting at 90), so `method='nearest'` resolves to exact matches -- mathematically identical to xarray's inner-join.

aurora_encoder resubmitted as 2455738; ~50 min wall expected. Other 5 baselines were unaffected (all on 721 lat).

See [[metric-audit]] "Grid alignment" section for the full story.

## Related

- [[crpss-migration]] -- the same scale-mixing problem ES faces (variable units span 7 orders of magnitude), solved here via per-layer truth-std normalisation instead of via climatology.
- [[calibration-winners]] -- the production-baseline picks; ES + VS provide an independent strictly-proper-multivariate sanity check.
- [[spatial-mean-vs-pointwise-ssr]] -- spatial-mean SSR also targets a multivariate-coherence gap, but only in the single dimension of global mean.
