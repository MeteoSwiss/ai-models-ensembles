---
date: 2026-06-13
summary: "Paper-plot colour audit vs swissclim helper (all aligned except MSLP maps kept on RdYlBu_r by choice); milton case-study scripts moved scratch -> tools/milton/."
---

# Paper colour-scheme audit + Milton script relocation (2026-06-13)

## Colour audit vs swissclim `get_colormap_for_variable`
Source of truth: `SwissClim_Evaluations/.../helpers.py` -- `get_colormap_for_variable`
(pressure/geopotential -> viridis, temperature -> magma, wind speed -> YlOrRd,
wind components/fluxes -> RdBu_r, precip/moisture -> Blues) + `SPATIAL_METRIC_SPECS`
(MAE/RMSE -> Reds vmin0, Bias -> RdBu_r). swissclim's own `intercomparison/modules/maps.py`
enforces both, so "aligned" for a field map = whatever the helper returns.

Result across all `main.tex` figures: everything that carries a variable/error colormap
is aligned. Line plots (CRPSS/SSR/temporalCRPS/track) use model/series palettes -- out of
scope. spectrogram_delta + bivariate_Tq are swissclim-generated (helper by construction).

Two findings:
- milton_F3_cascading_* and milton_F4_storm_relative plot MSLP with `cmap="RdYlBu_r"`;
  swissclim opinion for pressure = viridis. **Kept RdYlBu_r by user choice** -- diverging
  is the conventional synoptic MSLP coloring; not regenerated.
- milton_F6 member-spread maps used `magma` for 10m wind speed; swissclim wind_speed = YlOrRd.
  Changed `CMAP_SPD` magma -> YlOrRd in `tools/plot_milton_member_spread_maps.py:37`
  (spread panel stays viridis). Regenerated F6 (+ user added F7 72h / F8 96h variants).

## Milton scripts: scratch -> repo
Moved all case-study scripts out of `/iopsstor/scratch/cscs/sadamov/milton_case_study/`
into version-controlled `tools/milton/` (13 .py + env.sh + 2 submit_*.sh). The F6 script
`tools/plot_milton_member_spread_maps.py` was already in tools/ (also committed now).

Scratch dir keeps the DATA only (pair_list*.txt, milton_*.csv, *.nc, tracks/, __pycache__)
-- it remains the input home; scripts still read it by absolute path.

Link fixes so nothing dangles after the move:
- `gen_missing_f3.py` / `restitch_all.py`: hardcoded scratch `sys.path.insert` -> `Path(__file__).resolve().parent`
  (they import co-located `figures_milton` / `track_one_init`).
- `submit_milton_tracker.sh` / `_phase5.sh`: `source .../env.sh` and the `track_one_init.py`
  call repointed scratch -> `tools/milton/`. `LIST=.../pair_list*.txt` left on scratch (data).
- `analyze_milton_stats.py` `CASE` left on scratch (data dir it reads/writes CSVs from).

Verified: `py_compile` clean on all moved .py; no moved file references a deleted scratch script.
