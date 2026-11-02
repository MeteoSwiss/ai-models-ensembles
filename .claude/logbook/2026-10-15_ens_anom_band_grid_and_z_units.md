Z is now plotted as geopotential (m²/s²) in all physical-unit figures; added the 4-latitude-band KDE grid.

- User correction (2026-10-15): Z must stay geopotential in m²/s², never converted to geopotential height.
  `PHYS["geopotential"]` in `tools/plot_ens_anom_gaussianity.py` is now (1.0, "m²/s²"); rule in
  `.claude/rules/plot-units.md`. All physical-unit figures in `figures/ens_anom/` re-rendered (jobs 3666328,
  3666329); the standardized-z figures were unaffected. The "gpm" numbers quoted in
  `2026-10-14_ens_anom_gaussianity.md` are geopotential height; multiply by 9.80665 for m²/s².
- `band_grid_values_240h.{png,pdf}` (`plot_point_ensemble.py --bands`): equal-area bands 30-90N, 0-30N, 0-30S,
  30-90S from the cos-lat 2048-point sample (28 inits, ~512 points per band, KDE subsample 50,000 per
  dataset). Extratropical bands: all models on climatology. Tropical bands: Aurora SPW cold T2m / low Z500,
  GraphCast SPW warm T2m / high Z500, both too peaked in Q850 near 12.5 g/kg, consistent with the
  single-point results.

Cleanup (user call, 2026-10-15): kept only `figures/ens_anom/{member_values,member_anomaly_qq,point_grid,
point_grid_opposite,band_grid}_values/_240h.{png,pdf}`. Removed the standardized-z figures (qq_*, hist_*,
shape_vs_lead, w1_vs_lead) with the wd_kde module run and moment stats, the 14-panel hist_full/hist_phys/
qq_phys, and the per-point point_<name>_* figures with their code (all recoverable from git history before
the cleanup commit). `plot_ens_anom_gaussianity.py` now only makes the two large figures,
`plot_point_ensemble.py` only the three grids, `submit_ens_anom_plots.sh` runs both; `compute_point_ensemble.py`
defaults to all 8 points. Derived data on $STORE (`analysis/ens_anom_gaussianity/`, incl. the old wd_kde/
module outputs and shape_stats.csv) left in place.
History: the 14 exploratory commits (5237e28..4018667, ~46 MB of figure re-renders) were squashed into ONE
pushed commit `af00c82` on main (pushed 2026-10-15). The full exploratory history, incl. the removed
figures/code, lives only on the LOCAL branch `ens-anom-exploration` (not pushed); commit hashes quoted in
`2026-10-14_ens_anom_gaussianity.md` refer to that branch.
