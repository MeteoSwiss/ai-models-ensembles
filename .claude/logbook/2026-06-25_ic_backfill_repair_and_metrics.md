# 2026-06-25 IC-augmentation backfill: root-caused the 84/112 gap, repaired IC zarr, regenerated 112/112, refreshed _ic metrics

One-line: the four `_ic` runs were 84/112 because the IFS-ENS perturbed-IC zarr had 100% NaN at 150 + 600 hPa on the 28 early-2023 inits; filled those levels (log-p interp), backfilled the 108 missing rollouts to 112/112, recomputed the paper-cited `_ic` metrics on the full grid.

## Root cause (the "84-init anomaly")
- `_ic` = aurora_encoder_ic, graphcast_all_ic, sfno_modes10_ic, aifs_perturbed_ic. All were 84/112: the 28 inits Jan 2-8 + Apr 2-8 2023 (00/12Z) were NaN past lead 0.
- NOT the SIGSEGV/NaN-merge the old `submit_ic_backfill.sh` comment guessed. Actual cause: `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr` (224 init_time slots = 112 prod inits x their T-6h companions, CF long var names, levels [1000,925,850,700,600,500,400,300,250,200,150,100,50]) had 100% NaN at levels 150 and 600 on init_time indices 0:56 (the Jan + Apr 2023 clusters). Those two PL levels are never archived by MARS for type=pf step=0 (`download_ic_perturbed.py` reindex-fills them NaN and defers to a post-download log-p pass); that pass never ran on the early-2023 slots. NaN at 2 of 13 input levels -> every backbone's first AR step ingests NaN -> whole rollout NaN.
- Source levels (100/200/500/700) ARE finite on the bad inits, so log-p interp produces finite values. Forecasts are scored only at 500/850, so a fill at 150/600 has negligible downstream effect.

## Provenance caveat (worth knowing, negligible impact)
- The 84 good inits' 150/600 do NOT match simple log-p LINEAR interp: z600 residual ~22 m, near-spatial-constant (rmse approx bias) -> the good dates used a marginally finer/hydrostatic interp, NOT real MARS (150/600 are never MARS-archived for type=pf step=0). The 28 repaired inits use the documented log-p linear fill. Gap = <=22 m geopotential / <=0.8 K at 2 of 13 INPUT levels, scored at 500/850 -> negligible for any paper metric.
- `ifs_ens.zarr` in the same folder was NOT touched (0 files modified; only `ifs_analysis_perturbed_ic.zarr` written).

## What was done
- `tools/fill_ic_perturbed_levels.py` (NEW): CF-long-name log-p fill of 150/600 (the shipped `repo-download-ifs/fill_interp_levels.py` keys on GRIB short names and KeyErrors on the renamed live store). Idempotent, NaN-only writes, per-(init,level) region write into the per-init-sharded v3 store. Canary on idx0 proved archived levels + good inits untouched (z600 NaN->40253 = exact log-p prediction). Ran via `tools/submit_ic_fill.sh 0:56` (CPU sbatch, debug partition; ~30 min; user authorised the shared-store write).
- Backfill: `scripts/submit_ic_backfill.sh` (canary 1 init x4 -> all finite at 6/24/240h, then SKIP_INIT_TAGS=20230102_0000 for the other 27 inits x4 = 108 jobs). 1 SFNO GH200 SIGSEGV (member -11), resubmitted, landed finite. Result: 112/112 GOOD all 4 `_ic`. NaN shells archived to `forecast.zarr.nan_pre_backfill_2026-06-24` (never deleted). Added `SKIP_INIT_TAGS` env + corrected the root-cause comment.

## Bug fixed: per_pixel_ssr.py grid mismatch
- Aurora's native grid drops the -90 pole row (720 lat vs WB2 truth 721) -> `mse = ens_mean - truth` failed to broadcast. Added truth->forecast grid alignment (sel nearest when lat/lon sizes differ). graphcast/sfno/aifs are 721 (unaffected). Validated `per_pixel_ssr.py` vs SwissClim `ssr_line` to 1e-5 on fresh 112 data (graphcast_ic 0.7766=0.7766, aifs_ic 0.7894=0.7894).

## Metrics on the full 112 (paper §results-spatial)
- Per-pixel SSR @24h MSL (validated): wt-only -> _ic: aurora 0.629->0.783, graphcast 0.577->0.777, sfno 0.680->0.765, aifs 0.653->0.789. Paper: AIFS "0.65 to 0.85" -> "0.65 to 0.79"; gain "+0.11 to +0.29" -> "+0.09 to +0.20". DONE (L793-794).
- Spatial-mean SSR @240h (variable-mean over 7 vars, levels avg = the paper's aggregation; graphcast wt-only reproduced 3.204 = paper 3.20 exactly): graphcast_all_ic = 1.903. Paper "3.20 -> 1.86" -> "3.20 -> 1.90". DONE (L800-801).
- CRPSS "+/-0.02 at every lead" (L791): PENDING clean check. First pass (May-vintage wt-only CRPS vs fresh _ic) showed a 6h drop of -0.06..-0.10, but that is cross-vintage; same-vintage per-pixel SSR shows _ic is NOT over-dispersed at 6h (0.43->0.76, <1), so IC should help 6h CRPS not hurt it -> the drop is a vintage artifact. Re-running the 4 weight-only probabilistic modules at current vintage (their eval_main_config.yaml had STALE pre-migration swissai/a122 paths -> regenerated to mch/s83). Clean same-vintage crpss check pending (scratchpad crpss_compare.py; uses crps_clim_eval_1990_2019.json + per-baseline crps_line, the assemble_production_metrics_table.py recipe).

## Tooling added (uncommitted at time of writing)
- NEW: tools/fill_ic_perturbed_levels.py, tools/submit_ic_fill.sh, tools/submit_ic_ssr.sh, tools/submit_perpixel_ssr.sh (VARS/LEADS env-overridable; MSL-only is ~14x faster), scripts/submit_ic_backfill.sh.
- MODIFIED: scripts/evaluate_baselines.sh (optional 2nd arg = module filter, e.g. `evaluate_baselines.sh aifs_perturbed_ic probabilistic`), tools/per_pixel_ssr.py (grid align), ai-models-ensembles-paper/main.tex (per-pixel + spatial-mean numbers).

## Remaining
- Clean CRPSS lands -> finalize L791 (likely "at leads >=24h" if 6h is real, or keep "every lead" if vintage-confirmed flat). Compile-check (tectonic). Commit main + paper (no push).
