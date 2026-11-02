# 2026-06-16 attended session: paper-review ATTENDED follow-ups (F4 thickness regen, phase6 reproducibility, F2 grid, abstract, FGN)

Cleared the 5 ATTENDED items the nightly e2e review (`2026-06-16_paper_e2e_review.md`) deferred. Code in
main repo, prose in paper repo. Paper compiles clean (main.pdf ~18.9 MB, no undefined cites/refs).

## Item 3 (phase6) - the zarrs were NEVER deleted (the headline finding)
A scout earlier this session wrongly concluded the SFNO phase6 refresh-N=20 forecasts were deleted because
`phase6_family_v2/sfno/` holds only `intercomparison/`. WRONG TREE. The forecasts live in the phase3 tree:
`$STORE/ablation/phase3/sfno/<init>/mag_0.35_modes10_refresh20_0.35/forecast.zarr` (4 inits, ~26 GB each, all
intact). The sweep reused the phase3 run-tag convention; canonical path documented in
`phase6d_higher_anchor.md:156-158`. Convention now also recorded in `reference/project-conventions.md`
("Phase 6 forecast location") so this is not re-discovered the hard way. Decided NOT to physically move the
zarrs into a phase6 dir (breaks eval/intercomp paths + the regen command for no gain; NEVER-move rule).
- The only missing artefact was the derived CSV `$STORE/diagnostics/sfno_phase6_ssr/refresh20_0.35.csv`.
  Regenerated from the intact zarrs (no GPU) via `tools/spatial_mean_ssr.py` over the 4 inits +
  WB2 truth. VERIFIED the paper claim: frozen_0.25 = 0.83, refresh20_0.35 = 0.935 (7-var mean at 240h) =~ 0.94.
  (fresh_0.60 = 0.439 =~ 0.44 also checks.) Plotter's printed summary is 6-var (0.82->0.92); paper quotes the
  7-var mean. So 100% reproducible now.
- Extended `tools/plot_sfno_phase6_ssr.py` with a third "refresh" curve category (greens, `^-`, parses
  `refresh<N>_<sigma>` label). Regenerated `figures/phase6_sfno_spatial_mean_ssr.{pdf,png}`.
- Caption (main.tex ~799) rewritten to the ACTUAL figure: 6 variable panels, spatial-mean SSR (log), frozen
  sweep / fresh-per-step / refresh-N=20. Dropped the never-plotted "Top: per-pixel SSR" claim; kept the
  0.83->0.94 and fresh 0.44 numbers (now both shown as curves).

## Item 1 (F4) - regen as 500-850 thickness warm-core composite + ERA5 control; the "20-30%" claim was FALSE
Rewrote `f4_storm_relative_composite` in `tools/milton/figures_milton.py`. Was plotting MSL while the
caption/body claimed "500 hPa geopotential anomaly / warm-core thickness suppressed 20-30%". The ERA5 window
`/iopsstor/scratch/cscs/sadamov/milton_case_study/era5_milton_window.nc` provides ONLY `z_thick_500m850`
(500-850 thickness in m^2/s^2, no raw z500), so the only field with an ERA5 control is the thickness -> figure
is now the 500-850 hPa thickness anomaly (warm core). Per member: recentre on MSL min, interp to +-10 deg rel
grid; added a planar-detrend of the environment (radius>5 deg) to remove the strong N-S background gradient;
ensemble mean; peak = mean over inner 3 deg disc. ERA5 = analysis thickness composited the same way (the
reference). 8th panel ifs_ens dropped (no per-init forecast.zarr; lives in the IFS source zarr).
- VERIFIED numbers (gradient-corrected): ERA5 peak 45.9 m; aurora 26 m (+44%), graphcast 41 (+10%),
  atlas 43 (+5%), sfno 47 (-3%), aifs_perturbed 48 (-4%), aifsens 49 (-6%), fcn3 55 (-20%). So the warm core
  is NOT uniformly suppressed - it is model-dependent (only Aurora strongly damped; trained-prob + SFNO/AIFS
  match or exceed ERA5). Body (main.tex ~1093) + caption (~1162) reworded to this verified result; dropped
  "20-30%" and "grid-ceiling". (Ordering is robust to the reference choice; planar detrend only shifts levels.)

## Item 2 (F2) - grid 2x3 -> 2x4
`plt.subplots(2,3)` truncated 8 baselines to 6 via `zip(axes.flat, BASELINES)`. Changed to `(2,4)`
figsize (20,9); atlas + ifs_ens now shown. Caption (main.tex ~1134) "six of eight" -> "all seven ML
baselines plus IFS-ENS".

## Items 4 + 5 (abstract, FGN cite)
(paper-process detail kept in the local notes)

## Files
- main repo: `tools/milton/figures_milton.py` (F2 grid, F4 rewrite + `_storm_relative_thickness` helper),
  `tools/plot_sfno_phase6_ssr.py` (refresh category). New data: `$STORE/diagnostics/sfno_phase6_ssr/refresh20_0.35.csv`.
- paper repo: `main.tex` (abstract, FGN, phase6 caption, F4 body+caption, F2 caption) + regenerated
  `figures/{milton_F2,milton_F4,phase6_sfno_spatial_mean_ssr}`.
