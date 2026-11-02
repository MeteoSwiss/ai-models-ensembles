# 2026-06-15 Model-soups member-vs-mean RMSE decomposition (+ per-member lead-decay finding)

One-line: perturbed members are individually worse than the unperturbed control on deterministic RMSE,
but the ensemble mean is better - soup-beats-ingredients - and the per-member penalty decays to near zero
by 240 h.

## What was added to the paper
- Sec. results-ablation, after `tab:uplift`: a new paragraph + `Tab.~\ref{tab:member-vs-mean}` decomposing
  the deterministic gain into single-member vs ensemble-mean RMSE relative to the unperturbed control
  (`mag_0`). Plus a follow-on paragraph on the lead dependence of the per-member penalty.
- Table + first paragraph were committed concurrently by a parallel/nightly actor in paper `f74966b`
  ("...add model-soups RMSE decomposition table"). Only the lead-decay paragraph was committed by the
  attended session itself (paper repo).

## Numbers (verified)
240 h, 7 paper variables (3D averaged over 500+850 hPa first), ablation grid (4 mid-season inits), per-var
relative change vs the control's single-forecast RMSE then averaged over variables:

| production pick | per-member RMSE | ensemble-mean RMSE |
|---|---|---|
| aurora_encoder  (phase2b mag_0.025_layer_encoder) | +6.7%  | -19.9% |
| graphcast_all   (phase1  mag_0.01_layer_all)      | +13.2% | -15.8% |
| sfno_modes10    (phase3  mag_0.25_modes10)         | +3.2%  | -22.4% |

Cross-model ordering: the more the perturbation is confined to the coarse-scale operator, the LESS it
degrades the member (sfno coarse modes < aurora encoder < graphcast broadband all-weights). "Less harmful",
not beneficial.

## Per-member lead dependence (sfno_modes10, % RMSE vs control; aurora_encoder same shape)
```
variable                  24h    72h   120h   168h   240h
2m_temperature           +4.9  +11.3  +13.3   +8.0   +0.4
mean_sea_level_pressure +16.4  +21.7  +18.2  +11.3   +4.9
geopotential            +20.0  +24.9  +19.2  +12.2   +4.8
temperature              +9.2  +16.0  +14.1   +7.5   +0.5
u_component_of_wind      +8.4  +14.5  +11.9   +8.4   +4.3
v_component_of_wind      +7.7  +13.2  +12.1   +6.9   +3.6
specific_humidity        +6.0   +9.2   +8.8   +6.4   +5.3
```
All cells positive -> the perturbation never reduces per-member error (user hypothesis rejected). Penalty
peaks at 72-120 h, decays sharply toward 240 h (coarse scales drift, then saturation pulls every member
toward climatology near day 10). Practical read: at long leads the ensemble spread costs almost no
member-level accuracy.

## Validation + reproducer
- Ad-hoc cos-lat-weighted RMSE reproduces the pipeline deterministic `temporal_metrics_combined.csv` to 3
  decimals (perturbed ens-mean 2.514 vs CSV 2.515; control 3.019 vs 3.021, aurora 2m_t 240h) -> the
  per-member numbers from the same code path are trustworthy.
- `tools/compute_member_vs_mean_rmse.py` (NEW, main repo): host venv (`.venv`), reads forecast.zarr under
  `$STORE/ablation/<phase>/<model>/<init>/<cell>/` and WB2 truth
  `/capstor/store/cscs/swissai/weatherbench/weatherbench2_{2022_2023,2024_2025}.zarr`. Forecast lat grid is
  720 (90..-89.75), WB2 is 721 (90..-90) - align via `.sel(latitude=f.latitude, longitude=f.longitude)`.
  The per-lead/per-variable breakdown above was a throwaway script (not saved); regenerate by looping the
  tool over LEADS if needed.

## Honesty note
Did NOT write "physical perturbation reduces per-member error" anywhere - the data contradicts it. The
paper paragraph frames the lead dependence as "the penalty nearly vanishes", which is what the numbers show.
