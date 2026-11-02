# 2026-08-27 Review response: items 2 + 5 delivered, item 4 unblocked, AIFS cost sink fixed

One-line: the fixed-climatology rescore changes no selection (item 2), the spread-error diagram shows
the post-hoc ensembles ARE state-dependent (item 5), and two silent data-plumbing bugs were found -
AIFS reading a purged cache (~2 h/init instead of ~8 min) and IFS-ENS being skipped entirely.

## Item 2 - ANSWERED (no paper claim moves)
Fixed scale = `tools/data/channel_scale_1990_2019.json`: mean over 1990-2019 (100 weekly samples,
00 UTC) of the cos-lat-weighted SPATIAL std of ERA5, per (var, level). Deliberately the spatial std,
NOT a per-pixel temporal sigma - it replaces the per-init truth spatial std like for like, whereas a
temporal sigma would reweight the channels against each other and move the scores for a second,
unrelated reason.
- Winner rows, published -> fixed: ES +4.0..+6.0%, VS +5.7..+8.2%, SIGK +2.8..+33% (and SFNO 240 h
  -0.69 -> +1.89, a 2.6 absolute shift on a scale whose across-config spread is ~250).
- 0 of 24 winner/rival comparisons change direction, at either lead.
- THE PAPER STILL PRINTS THE TRUTH-STD NUMBERS. Swapping tab:calibration to the fixed-scale column is
  a user decision, not made.

## Item 5 - ANSWERED, `figures/spread_error.pdf`
Weighted spread histogram + matching ensemble-mean squared error, collapsed to spread deciles.
| baseline | lead | RMSE low decile | high decile | ratio |
| aurora_encoder | 240 h | 1.447 | 5.774 | 4.0x |
| sfno_modes10 | 240 h | 0.824 | 6.001 | 7.3x |
| aifs_perturbed | 240 h | 0.732 | 5.240 | 7.2x |
| aifsens (TP) | 240 h | 0.475 | 5.275 | 11.1x |
Monotone for every baseline: large predicted spread really does go with large error. At 240 h the
post-hoc curves sit on the 1:1 line, indistinguishable from TP. At 24 h they rise above it (mean
RMSE/spread 2.2-3.5 vs 1.1-1.2 TP) - the known short-lead underdispersion, now shown to be a
MAGNITUDE deficit rather than a failure to discriminate.
NOTE: ifs_ens is absent from the figure - `spread_error_binned.py` has the same per-init-dirs
assumption that hid it from the CRPS pass (see below). Rerun for ifs_ens alone if it is wanted.

## Item 4 - was silently unanswerable
`compute_per_init_crps.py` looked for `$STORE/baselines/ifs_ens/<init>/forecast.zarr`. That directory
holds eval configs only: the forecasts are ONE consolidated zarr at
`/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr` (init_time x 50 members x lead x lat x lon). The
tool skipped it without a word, so every "common sample" column came back NaN with n=0. Reader now
handles the consolidated layout + the stratified 10-member subsample [0,5,...,45]. Probe: 6 inits in
94 s, and the `n_members=0` rows ARE the valid-case mask at (variable, init, lead) granularity, so no
separate mask job is needed. Full pass = job 3197062 (~2.5 h), then `tools/crpss_common_sample.py`.

## AIFS cost sink (the big one)
The campaign mounted `/iopsstor/scratch/cscs/sadamov/e2s_cache`, which the scratch purge had cut to
898 entries, so every AIFS init re-queued against the CDS API: ~2 h/init, of which ~36 min was the
sequential CDS warm-up. The persistent cache documented in `.claude/logbook/aifs_predownload_cache.md`
and `.claude/handoff/LATEST.md` - `$STORE/e2s_cache_backup`, 5376 GRIBs, 43 GB, README_DO_NOT_DELETE -
was never mounted. `submit_all_inference.sh` now picks it whenever `dsrc == cds`. Verified warm: first
init hit Round 3/3 in ~7 min with 140 CDS lines at ~45 ms spacing (cache hits; the log line prints
BEFORE the cache check, so real queue waits show minutes between lines). ARCO cannot substitute - it
lacks AIFS's land-surface fields. Cost: 224 AIFS inits drop from ~245 node-h to ~34.

## Other fixes this session
- Week-helper scripts are read from INSIDE the container; only `$STORE` is mounted, so an overridden
  `LOG_DIR` on iopsstor made 15 jobs die in 15 s. Now bind-mounted when off `$STORE`.
- ARCO prefetch (`_prefetch_ic_data`) had no retry of its own despite existing to avoid blosc races:
  3 attempts, 30/60 s backoff. GraphCast is worst hit (82 variables x 2 times per init).
- Round retries 1 -> 2 (`E2S_ROUND_RETRIES`) for the GH200/UCX startup SIGSEGV.
- MY REGRESSION: `os.environ` used in `_run_members_parallel` above its later function-scoped
  `import os` -> UnboundLocalError killed 6 jobs in a 4-min window. Verify import/use order in the
  actual module before leaving an edit that 250 queued jobs will execute.
- Rivals extended 56 -> 112 inits (user call): 4 seasonally-spaced weeks was defensible but the full
  grid makes the paired winner-vs-rival comparison match the winners' own grid and lifts the block
  bootstrap from 4 weekly units to 8.
- `tools/data/crps_clim_eval_1990_2019.json` regenerated (now also emits the per-init companion):
  values IDENTICAL to the committed ones, which independently re-validates the CRPSS denominator.

## Tab. C1 SSR column was stale by one eval vintage (found 08-27, FIXED)
Symptom: HEAD's own `assemble_production_metrics_table.py`, run today against today's data, gives
aurora per-pixel SSR 0.86 where the committed table says 0.76.
Cause: the table's SSR column was generated 2026-06-19 (`e56b71e`). On 06-25 the FOUR weight-only
baselines' probabilistic modules were re-run because their `eval_main_config.yaml` still held
pre-migration `swissai/a122` paths (logbook `2026-06-25_ic_backfill_repair_and_metrics.md`), which
rewrote every `ssr_line_*_by_lead_ensprob.csv`. The table was never regenerated afterwards. `ssr()` is
byte-identical across that span, so the 06-25 values are the corrected ones. The TP rows did not move
(they were not re-run), which is the fingerprint that confirms the diagnosis.
Corrected per-pixel SSR (published -> current):
  aurora 0.76/0.82 -> 0.86/0.94 | graphcast 0.76/0.94 -> 0.86/1.08
  sfno   0.73/0.85 -> 0.83/0.97 | aifs     0.85/0.96 -> 0.98/1.11   (120 h / 240 h)
THIS CHANGES A CLAIM, not just digits: the 240 h post-hoc range moves 0.82--0.96 (all < 1,
"underdispersion") to 0.94--1.11, which STRADDLES 1 - graphcast_all and aifs_perturbed are now mildly
OVERdispersed pointwise. Updated main.tex L693 (the underdispersion sentence) and L1036 ("pointwise
fields are already calibrated ... 0.82-0.96" -> 0.94--1.11), plus the Tab. C1 SSR cells.
Audited the regenerated table cell by cell: ONLY SSR (9 cells) and ES/VS/SIGK (14 each) differ from the
published version; CRPSS, SSIM, LSD, FSS and W1 are untouched.
LESSON: a re-eval that rewrites the per-baseline eval CSVs does not refresh the LaTeX tables. After any
`evaluate_baselines.sh` re-run, regenerate the tables and diff them against the committed versions.
