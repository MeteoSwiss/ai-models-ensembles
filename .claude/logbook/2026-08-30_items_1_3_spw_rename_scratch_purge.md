# 2026-08-30 Items 1+3 landed, SPW rename, and a second scratch-purge casualty

One-line: answered review items 1 and 3 on the full 112-init grid, renamed the method to SPW
(stochastically perturbed weights) and rewrote the positioning for items 7-9, then discovered the
iopsstor purge had destroyed both the Milton venv and every pre-existing Milton track.

## Review items: all nine now answered
- Item 1 (IC vs weight attribution): `ic_decomp_4bb` job 3222494, all four backbones at 112/112/112,
  720 rows -> `tools/data/ic_decomp_4bb.csv`. Two findings:
  * The sources combine in VARIANCE, not in spread. Quadrature ratio
    sigma^2_both/(sigma^2_wt + sigma^2_ic) is 0.99-1.00 at 6 h and falls to 0.59-0.69 at 240 h.
    Implied correlation rho within +-0.015 of 0 at 6 h, -0.32 to -0.41 at 240 h = saturation, not
    anti-correlation.
  * Roles reverse: weight-only supplies 8-17% of combined variance at 6 h, 86-97% at 240 h;
    IC-only 84-92% at 6 h, 57-76% at 240 h. This is what justifies weight-only for the headline and
    weight+IC for Milton.
- Item 3 (runner-up validation): `per_init_rivals2` job 3230743 (7:30), all 6 rivals at 112.
  At 240 h no rival displaces its winner (max delta +0.0011). The sfno_enc_s054 near-tie is resolved
  by the SSR tie-break the selection rule already specifies. Table has 0 pending rows.
- Items 7-9 (positioning): title is now "Stochastically Perturbed Weights: Ensembles from
  Deterministic Machine-Learning Weather Models". Three-question frame (Q1/Q2/Q3) in the intro,
  Discussion 5.1 retitled to lead with architecture-specificity, Conclusions restructured as three
  answers. `post-hoc` kept ONLY as the literature-taxonomy term (5 places); everything else is SPW.

## Method naming: why SPW and not "retrofitting uncertainty"
A co-author suggested "retrofit uncertainty". Rejected after literature check: (a) you retrofit a
capability, not a phenomenon, and "adding uncertainty" reads as degrading, which Tab. uplift's
per-member RMSE penalty would hand a referee; (b) "retrofitting" is taken in ML (Faruqui 2015).
SPW deliberately echoes SPPT/SKEB so the NWP audience reads it as the ML-era continuation of
stochastic physics. Note SPP (Stochastically Perturbed Parametrizations) went operational at ECMWF
Nov 2024, so SPW sits one letter from it - accepted deliberately by the user.

(paper-process detail kept in the local notes)

## SECOND scratch-purge casualty (the big lesson)
The iopsstor purge signature is: regular files stripped, directory tree LEFT STANDING. It killed
- `venvs/milton_case`: pyvenv.cfg gone (so sys.prefix reported the base interpreter and site-packages
  was never added), and every package lost its top-level files. numpy imported with no attributes.
  All 56 Milton tracker tasks died in 8 s. Venv DELETED 2026-08-30 per user; Milton now uses the
  capstor project venv + capstor miniforge3 (DetectNodes/StitchNodes/udunits). `tools/milton/env.sh`
  rewritten, 10 scripts repointed.
- `milton_case_study/tracks`: ALL 12 pre-existing arms had 14 init dirs and ZERO files.
  I initially reported "tracks survived" because I counted DIRECTORIES with `ls | wc -l` and never
  looked inside - the same shallow-count error as the truncated-glob trap. Re-tracked via jobs
  3236670 (6 weight-only) + 3236673 (4 weight+IC). Vendored figure PDFs in the paper repo were
  never at risk.

LESSON: to check an output arm is intact, count FILES INSIDE the leaf dirs, never the dirs.

## Durability pass
All analysis outputs were living only on scratch. Copied and byte-verified with `stat` (du -sh on
capstor lags and read 67K for 13 MB - do not trust it):
- `tools/data/ic_decomp_4bb.csv` (in repo, feeds tab:ic-decomp)
- `$STORE/analysis/`: per_init_crps_production_v2.csv, spread_error_{all,binned,ifsens}.csv,
  per_init_crps_ifsens.csv, ic_decomp_12ch.csv, table_metrics_ic_decomp/ (ES/VS/SIGK, 7 h job)
Both table generators now default to durable paths, no scratch dependency.
STILL on scratch: e2s_cache 155G (backed up at $STORE/e2s_cache_backup 44G, only re-download time
at risk), review_logs, milton_case_study (move to capstor once trackers finish).

## New tooling
- `tools/make_ic_decomp_table.py`, `tools/make_rival_validation_table.py` - generate the two appendix
  tables from CSV so numbers cannot drift. The rival one writes literal `(pending)` for any arm short
  of 112 rather than scoring a different sample.
- `tools/submit_esvs_sigk_ic_decomp.sh` - ES/VS/SIGK for all 12 decomposition arms, fixed
  climatological scale, leads 24 + 240. Job 3213194 done, all arms n=112.
- `tools/milton/submit_milton_tracker_ic_only.sh` - completes the weight-only/IC-only/weight+IC
  triptych for the case study.
- `ic_weight_decomposition.py`: added `spread_rms` (sqrt of mean variance - the aggregation that makes
  variance additivity testable; mean-of-std is biased UPWARD by Minkowski, confirmed with a synthetic
  control reading 1.0365 vs 1.0041 under exact additivity), a scorability pre-flight replacing the
  presence-only guard, and a hard assert if a cell scores fewer inits than were pre-validated.

## The guard hole that mattered
`--require-inits 112` counted DIRECTORIES. aifs_ic_only/20230704_0000 had a forecast.zarr written
during the 08-27 capstor incident holding 7 of 9 variables and no consolidated metadata, so the guard
passed while every aifs `ic` cell silently scored 111. Archived to
`baselines/aifs_ic_only_truncated_2026-08-28/` (never deleted), re-ran that init, and the guard now
opens every store before scoring.

## ES/VS/SIGK on the decomposition arms (new, not yet in the paper)
At 240 h SIGK, weight+IC is BEST for every backbone and IC-only is dramatically worst:
aifs -20.2 / aurora -18.4 / graphcast -11.8 / sfno -5.8 (weight+IC) against +53.0 / +7.6 / +46.0 /
+66.3 (IC-only). So the two sources are sub-additive in variance but SUPER-additive in temporal path
realism. graphcast_all moves from worst (+20.8 weight-only) to -11.8 - and GraphCast is exactly the
backbone that cannot use the refresh lever. Strong candidate for the paper; not yet written in.

## Paper state at end of day
(paper-process detail kept in the local notes)
NOTHING COMMITTED in either repo - both sit at their pushed heads (main daea583, paper 724db94).
