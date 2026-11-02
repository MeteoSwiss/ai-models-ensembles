# IFS-ENS can fairly join Fig. tier1b (spatial-mean SSR) as a reference line - NaN gaps are clean and random

Date: 2026-10-07 (attended). Question: Fig. tier1b (`tools/plot_7way_spatial_mean_ssr.py`) never had IFS-ENS
(no `baselines/ifs_ens/spatial_mean_ssr/spatial_ssr.csv`; 2026-06-13 only fixed the caption count). Is adding it fair?

## Data (reusable, do not redo)
`$STORE/analysis/ifs_ens_spatial_ssr_audit/`: per-init spread^2/err2 + NaN audit for all 8 baselines on exactly the
tier1b slabs (112 inits x 12 fields x leads 24/72/120/240), `signed/` adds ens_mean/truth_mean for ifs_ens, aifsens,
fcn3 (the main CSVs from atlas onward also carry them). Scripts + `analysis_output.txt` alongside. Re-aggregation
reproduces all 7 published spatial_ssr.csv to <=1e-4 rel (atlas 1.2e-3).

## Findings
- IFS-ENS gaps are whole slabs only: 1105/5376 (20.6%) all-NaN, 0 partial, 0 member-mixed, no zero fill, ERA5 truth
  complete. Both 3D levels always drop together. Every init misses 3-21 of its 48 slabs (none complete).
- Available inits per (field, lead): 83-97/112; MSL 69-83 (includes the patched 2024-07-02..08 14-init block).
- No pattern: missing share 19-22% per season, init hour, year and lead.
- Subset effect: re-aggregating the 7 plotted baselines on IFS-ENS's surviving inits gives 10.7% of cells outside
  the random-subset 5-95% band (10% expected) -> missingness is random-like. Plot-level 240 h 7-var means move
  -2.1..+4.7%. Outliers are MSL only (graphcast_all MSL +25% @120 h, atlas -11% @240 h), via the July block.
- IFS-ENS (10 members, every fifth) plot-level 7-var mean SSR: 0.30/0.36/0.43/0.60 at 24/72/120/240 h
  (50 members 0.59 @240 h). AIFS-ENS on the same subset 0.40/0.45/0.50/0.615.
- Verification asymmetry, not NaN, drives the short-lead gap: IFS-ENS global-mean bias vs ERA5 is 63% of its err2 at
  24 h (AIFS-ENS 8%). Bias-removed SSR at 24 h: IFS-ENS 0.49 = AIFS-ENS 0.49. SPW bias shares are as large or
  larger (aurora_encoder 61-86%, graphcast_all 41-76%), so this is not IFS-ENS-specific at long lead.

## Decision
Fair to add, as a dotted grey classical-reference line (10 stratified members, per-variable skipna, same rule as
the CRPSS/rank-histogram IFS-ENS rows). Keep it out of the "TP band" (band stays AIFS-ENS/FCN3/Atlas, 0.62-1.42).
Caption must state the reduced n (69-97 of 112) and that its short-lead deficit reflects the ERA5-verification
asymmetry already listed in the limitations. Contrast: multivariate ES/VS/SIGK stay excluded
(`2026-06-21_ifs_ens_multivariate_score_infeasible.md`) because they need joint completeness; this score does not.

## Implemented same day (unpushed)
- `baselines/ifs_ens/spatial_mean_ssr/spatial_ssr.csv` written from the audit terms (members 0,5,...,45 = the
  `IFS_ENS_MEMBERS` of `scripts/evaluate_baselines.sh`). Cross-check: `tools/spatial_mean_ssr.py --members 0 5 ... 45`
  on `ifs_ens.zarr` (job 3605216, 3h23) matched (same n_inits, max SSR diff 0.2%, 7-var means identical) and is now the canonical `spatial_ssr.csv`; audit version kept as `spatial_ssr_from_audit.csv`. Figure not regenerated (sub-pixel change).
- `tools/plot_7way_spatial_mean_ssr.py` gains `REFERENCE = {"ifs_ens": ...}`; figure regenerated (other 7 baselines'
  240 h values unchanged), legend diffed vs committed PDF (+IFS-ENS only), synced, Fig. 4 caption extended
  (captions are word-count exempt).

## Why IFS-ENS q spatial-mean SSR is so low (0.14-0.21 level-mean)
Global-mean humidity offset vs ERA5, not dispersion. q850: +4.1% moist already at lead 0 (9 sampled inits, 3.2-4.8%),
+4.6% @24 h, +5.4% @240 h, same sign in 100% of inits and every season; bias = 98-99% of err2; spread of the global
mean 0.02-0.05 g/kg vs offset 0.29-0.34 g/kg -> raw SSR 0.06-0.16, bias-removed 0.82-1.00. q500: +1.2% at lead 0,
drifts dry (-1.7% @24 h, -5.1% @240 h), bias share 87-89%, bias-removed 0.66-0.70. AIFS-ENS has the same kind of
q850 offset (+2.0%, 98% share, raw q850 SSR 0.055 @24 h) but q500 is near-unbiased, so IFS-ENS is lowest in the panel.
