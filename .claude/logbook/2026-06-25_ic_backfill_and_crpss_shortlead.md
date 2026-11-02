# 2026-06-25 IC zarr repair + 28x4 _ic backfill to 112/112, and a real short-lead CRPSS cost of IC augmentation

One-liner: Root-caused + fixed the _ic "84/112" gap (IC zarr NaN at 150/600 hPa on the 28 early-2023
inits), backfilled all four _ic baselines to 112/112, recomputed the paper-cited _ic metrics on the full
grid, and corrected three main.tex numbers - including a genuine (not vintage) short-lead CRPSS cost the
"unchanged within +/-0.02 at every lead" claim was hiding.

## Root cause of the _ic gap (was mis-attributed to SIGSEGV/NaN-merge)
- The IFS-ENS perturbed-IC zarr `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr`
  (224 init_time slots = 112 prod inits x their T-6h companion; CF long var names) was 100% NaN at
  pressure levels 600 and 150 hPa on exactly the first 56 slots (init_time idx 0:56 = the Jan-2023 +
  Apr-2023 week clusters). Those two levels are NEVER MARS-archived for type=pf step=0 (the downloader
  `reindex`-fills them NaN and defers to a log-p interpolation pass that never ran on those slots).
- NaN at 2 of 13 INPUT levels -> every backbone's first AR step ingests NaN -> whole rollout NaN past
  lead 0. That, not the SIGSEGV/AIFS-cancel guesses in the old script comment, is why those 28 inits x 4
  models were NaN shells.
- The other 84 inits' 600/150 are finite but do NOT match simple log-p linear interp (z600 residual
  ~22 m, near-spatial-constant) -> a finer/hydrostatic interp, still not real MARS. `ifs_ens.zarr` was
  only ever READ (round-1 reviewer work), never written.

## Fix
- `tools/fill_ic_perturbed_levels.py` (new, CF-long-name pendant of the build's short-name
  fill_interp_levels.py): log-p linear fill f(150)=0.415 f(100)+0.585 f(200), f(600)=0.458 f(500)+0.542
  f(700); idempotent, NaN-only region writes into the per-init-sharded v3 store. Canary on idx0 proved it
  fills the NaN target levels and leaves archived levels + good inits byte-untouched. Ran 0:56 via
  `tools/submit_ic_fill.sh` (debug partition, ~30 min). User explicitly authorised the shared-store write.
- Backfill `scripts/submit_ic_backfill.sh` (per-init isolation; archives NaN shells to
  forecast.zarr.nan_pre_backfill_2026-06-24, never deletes): canary on 2023-01-02 00Z (4 models, member
  output finite at 6/24/240 h) then full 27x4=108 (SKIP_INIT_TAGS guards the canary init). 1 SFNO GH200
  SIGSEGV (member 8/9 code -11) resubmitted -> landed finite. Result: 112/112 GOOD all four _ic.

## Metric recompute on 112 + main.tex corrections (sec:results-spatial)
- Per-pixel SSR @24h MSL (Fortin; `tools/per_pixel_ssr.py`, validated vs SwissClim ssr_line to 1e-5):
  aurora 0.629->0.783, graphcast 0.577->0.777, sfno 0.680->0.765, aifs 0.653->0.789. Paper: AIFS
  "0.65 -> 0.85" corrected to "0.65 -> 0.79"; gain "+0.11 to +0.29" -> "+0.09 to +0.20" (the old 84-init
  numbers were a stale vintage). BUG fixed in per_pixel_ssr.py: Aurora's native grid drops the -90 pole
  row (720 vs 721 lat) -> truth now aligned to the forecast grid before differencing.
- Spatial-mean SSR @240h (variable-mean over 7 vars): graphcast_all 3.204 (reproduces the paper's "3.20"
  EXACTLY) -> graphcast_all_ic 1.903. Paper "3.20 -> 1.86" corrected to "3.20 -> 1.90".
- CRPSS: re-ran the 4 weight-only probabilistic at current vintage (they reproduced the May values
  EXACTLY -> CRPS is vintage-stable). Same-vintage variable-mean CRPSS(_ic) - CRPSS(weight-only):
  6h = -0.06 to -0.10 (aurora -0.104, gc -0.074, sfno -0.060, aifs -0.070); 24h <= -0.022; >=72h within
  +/-0.02. So "unchanged within +/-0.02 at every lead" is FALSE at 6 h - a REAL spread-skill tradeoff
  (IC adds analysis spread to a near-deterministic 6 h forecast; not yet skill-bearing). Paper corrected
  to "within +/-0.02 of weight-only at 24 h and beyond, at a modest short-lead cost (0.06-0.10 reduction
  at 6 h ...)". User chose the honest-tradeoff-with-reason framing. Per-pixel SSR @6h corroborates the
  spread is added but stays < 1 (0.43->0.76), i.e. better calibrated yet a CRPS cost - the sharpness of
  the 6 h forecast is what drops.

## Gotchas logged
- Weight-only eval_main_config.yaml for aurora/graphcast/sfno carried the PRE-migration
  `swissai/a122/sadamov` $STORE path (configs dated 05-27/28, before the 05-30 move to mch/s83) -> 1-min
  FileNotFoundError. Fix: `evaluate_baselines.sh gen-configs <model>` regenerates with the correct path.
  aifs_perturbed (config 06-06) was already correct.
- Spatial-mean SSR on the 112 zarrs MUST be a CPU sbatch; per-pixel SSR restricted to MSL via
  VARS="mean_sea_level_pressure 2m_temperature" runs ~14x fewer slab reads (minutes vs hours).
- `evaluate_baselines.sh <model> probabilistic` now accepts a module filter (2nd positional arg).

## Tooling added/changed (committed, NOT pushed)
- new: tools/fill_ic_perturbed_levels.py, tools/submit_ic_fill.sh, tools/submit_perpixel_ssr.sh,
  tools/per_pixel_ssr.py, tools/submit_ic_ssr.sh, scripts/submit_ic_backfill.sh
- mod: scripts/evaluate_baselines.sh (module filter), per_pixel_ssr.py grid-align fix
- paper: main.tex 3 numbers.
