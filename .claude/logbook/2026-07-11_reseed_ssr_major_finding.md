# 2026-07-11 - MAJOR: the seeding bug was FAKING the refresh SSR "correction" of the over-dispersed backbones

One-line summary: the full 112-init reseed grids (corrected fresh-hook seeding, 2026-07-10 fix) show that
"refresh pulls the over-dispersed Aurora/AIFS spatial-mean SSR down toward 1" was largely a BUG ARTIFACT.
With correct, independent noise the over-dispersed backbones barely improve (Aurora) or do not improve /
get worse (AIFS); only the under-dispersed SFNO genuinely benefits. This changes the fig:phase6 + sec 4.3
narrative and needs a paper decision.

## The numbers (spatial-mean SSR @240h, 7-var x 2-level mean, 112 inits, reseed vs original-buggy)

| model | FROZEN (production, unchanged) | refresh BUGGY (paper) | refresh FIXED (reseed) | fixed vs buggy |
|-------|-------------------------------:|----------------------:|-----------------------:|---------------:|
| Aurora (over-disp) | 1.56 | 1.11 | 1.434 | +0.322 |
| AIFS   (over-disp) | 2.54 | 1.68 | 2.504 | +0.822 |
| SFNO   (under-disp)| 0.66 | 0.76 | ~0.92 (spot; full pending) | +0.10 |

Per-lead (reseed - buggy), confirming it is not a 240h fluke:
- Aurora: 24h -0.227, 72h -0.085, 120h -0.020, 240h +0.322 (mixed; small net pull-down at long lead only).
- AIFS:   24h +1.073, 72h +0.955, 120h +0.967, 240h +0.822 (reseed HIGHER at EVERY lead; at short leads
  2.85-3.27, i.e. ABOVE the frozen 2.54 - refresh makes AIFS over-dispersion WORSE, not better).

## Interpretation
The old seeding reused noise across members (11 distinct draws for 20 member/epoch slots) and across
tensors within a step. That SUPPRESSED ensemble spread -> artificially LOW SSR -> made refresh look like it
"pulled the over-dispersed backbones down toward 1". With independent noise (the fix), the spread is
restored and:
- Aurora frozen 1.56 -> refresh 1.43 @240h (a modest pull-down, not the dramatic 1.56->1.11 claimed).
- AIFS frozen 2.54 -> refresh 2.50 @240h, and WORSE at short leads: refresh does NOT correct AIFS
  over-dispersion.
- SFNO (under-dispersed) frozen 0.66 -> refresh 0.92: refresh genuinely lifts the under-dispersed backbone
  toward 1 (spot-check; full-grid value lands when the 3 resubmitted sfno inits finish + sfno SSR runs).

So the true story: **refresh reliably improves spatial-mean SSR only for the UNDER-dispersed backbone
(SFNO). For the OVER-dispersed backbones the apparent "correction toward 1" was the seeding bug depressing
spread; corrected, refresh barely helps Aurora and does not help (arguably harms) AIFS.**

## Paper impact (USER DECISION - do NOT rewrite unilaterally)
fig:phase6 + sec 4.3 currently claim refresh "moves the spatial-mean SSR toward 1 wherever the perturbation
lives ... pulls the over-dispersed Aurora (1.56->1.11) and AIFS (2.54->1.68) down toward 1, while for the
mildly under-dispersed SFNO (0.66->0.76) the same lever lifts it up". After the fix:
- The numbers change: Aurora 1.11->1.43, AIFS 1.68->2.50 (and SFNO 0.76->~0.92).
- The claim is FALSIFIED for the over-dispersed backbones. The "refresh-every-20 as the recommended
  spatial-mean fix for the weight-space backbones" framing does not hold for Aurora/AIFS.
- Options for the user: (a) reframe Phase-6 as "refresh helps the under-dispersed regime (SFNO), not the
  over-dispersed one - a scale/coherence argument about why"; (b) drop/soften the over-dispersed
  correction claim; (c) reposition the whole refresh excursion. This is a genuine scientific correction the
  seeding fix surfaced - the buggy numbers should not go into the paper.

Frozen headline (8-way, Milton, uplift, per-variable) is UNAFFECTED - the frozen arm always seeded
correctly (verified: R7 8/8-block sign test). Only the Phase-6 refresh excursion is affected.

## CRPS costs (refresh vs FROZEN production baseline, @240h per variable) - all eval DONE ~12:30

SFNO full-grid SSR landed: frozen 0.66 -> refresh 0.833 (fixed) vs 0.764 (buggy) -> refresh lifts the
under-dispersed SFNO toward 1, and the fixed value is closer to 1 than the buggy. SFNO story HOLDS.

CRPS % change refresh-vs-frozen @240h (FIXED reseed | vs BUGGY where available):
- Aurora (vs aurora_encoder): 2t +9.0%, t +5.1%, z +3.4%, msl +1.9%, v +1.6%, q +1.4%, u +1.2%. So the
  paper's "1-5% cost, Aurora 2t the +10% outlier" HOLDS for Aurora (fixed 2t +9.0%, was +9.5% buggy).
- AIFS (vs aifs_perturbed): ALL NEUTRAL-to-NEGATIVE - 2t -0.6%, msl -2.2%, z -1.6%, t -0.1%, u -0.6%,
  v -0.6%, q -0.4%. The paper's "AIFS z500 +9% CRPS cost" was a BUG ARTIFACT (buggy z +6.1%); with correct
  seeding AIFS refresh is CRPS-neutral (slightly better). So AIFS refresh neither improves SSR nor costs CRPS.
- SFNO (vs sfno_modes10): all ~-1% (2t -1.2%, z -1.3%, v -1.2%, ...): CRPS-neutral, consistent with the
  ablation "CRPSS unchanged 0.143 vs 0.145". (sfno_p6c had no CRPS computed, so no buggy column.)

## COMPLETE corrected picture (fixed seeding, 112 inits)
| model | SSR frozen->refresh | CRPS cost (refresh vs frozen @240h) | verdict |
|-------|---------------------|-------------------------------------|---------|
| SFNO (under-disp)  | 0.66 -> 0.83 (toward 1)          | ~-1% (neutral)          | refresh HELPS, cost-free |
| Aurora (over-disp) | 1.56 -> 1.43 (small toward 1)    | +1-5%, 2t +9%           | marginal (small gain, real cost) |
| AIFS (over-disp)   | 2.54 -> 2.50 (no help; worse <=120h) | ~0/-2% (neutral)    | refresh does NOTHING useful |
Two paper claims need correction: (1) fig:phase6/sec-4.3 SSR "pulls over-dispersed down toward 1" (Aurora
1.56->1.11, AIFS 2.54->1.68) -> corrected 1.43 / 2.50, claim FALSIFIED for the over-dispersed backbones;
(2) sec-4.3 "AIFS z500 +9% CRPS cost" -> bug artifact, corrected AIFS refresh is CRPS-neutral. Aurora's
1-5%/2t-+9% CRPS cost and SFNO's toward-1 lift both HOLD. USER DECISION on the reframe.

## PHASE-6 REFRAME APPLIED (2026-07-11, attended, user decisions) - uncommitted, held for review
User chose: (1) "cut/reposition the excursion", (2) "I do it now", (3) keep the cautious empirical mechanism
wording, (4) MOVE fig:phase6 to an appendix. Applied to main.tex:
- Results core paragraph rewritten: "Refresh does not generalise across the backbones" - SFNO 0.66->0.83
  (budget-rescaled sigma_N=0.35), Aurora 1.56->1.43, AIFS 2.54->2.50 (unchanged-to-worse at short leads);
  Aurora 1-5% CRPS (2t +9%); the falsified "recommended fix for the weight-space backbones" claim REMOVED.
- fig:phase6 caption rewritten (no "pulls overdispersed down"); figure MOVED to appendix D and REGENERATED
  from the reseed grids (tools/plot_refresh_ssr.py repointed at *_p6c_reseed; new PDF synced to paper repo).
- Abstract: dropped the false "refreshing ... can reduce this" clause.
- GraphCast exception: "opposite of its effect on the weight-space backbones" -> "on the underdispersed SFNO".
- Conclusions: "shrinks the gap wherever the weights localise" -> "only for the underdispersed SFNO
  (0.66->0.83, no CRPS cost); the same cadence ... barely moves the overdispersed Aurora and AIFS (1-5% CRPS
  for Aurora)".
- Milton "refresh-every-N SSR fix demonstrated end-to-end on it [SFNO]": LEFT (SFNO-scoped, still correct).
tectonic PASS (0 errors, 10.58 MiB); net word-negative (abstract clause + results figure removed). The
methods sec 3.2.2 refresh mechanism + appendix A variance-budget derivation LEFT (real method). Files:
main.tex, tools/plot_refresh_ssr.py, figures/refresh_frozen_vs_refresh_ssr.pdf (both repos). NOT committed.

## Eval state (in flight, 2026-07-11 ~06:50)
- Aurora/AIFS: inference 112/112, spatial-mean SSR DONE (numbers above), CRPS (probabilistic) RUNNING
  (jobs 2733344 aurora, 2733345 aifs; 112-init configs verified).
- SFNO: 109/112 -> 3 inits resubmitted (2733398/99/2733400, empty dirs, no forecast.zarr; not a failure,
  just didn't finish). SSR watcher b2witahjn + CRPS watcher b0yhblkq1 will auto-fire sfno SSR + CRPS at 112
  (stale 109-init sfno config was removed so CRPS regenerates fresh).
- CRPS still needs its numbers read once the 2 jobs finish (sec 4.3 refresh cost deltas + the AIFS z500 /
  Aurora 2t outliers - all recomputed from the FIXED runs).

## MISTAKE caught + fixed this session
Running `bash scripts/evaluate_baselines.sh` with NO ARGS defaults ACTION to "all" -> it submitted the FULL
eval matrix (138 jobs: ~8 modules x 17 models, re-evaluating existing baselines). Caught immediately;
cancelled 136 (kept only aurora/aifs reseed probabilistic); most were PENDING so minimal compute burned.
Lesson: never run evaluate_baselines.sh without an explicit model+module for a targeted eval.
Also fixed a latent `python3`-not-on-PATH date bug in evaluate_baselines.sh (lines 51/111 -> `date -d`),
same class as the submit_p6c_reseed.sh bug (both were producing the fromisoformat error / malformed dates).

## SIGK / ES-VS on the reseed grids (job 2734247, 112-init, done 2026-07-11 ~16:27) - CONFIRMS the reframe

Computed the multivariate/path scores for the 3 reseed refresh grids to check whether refresh helps on
SIGK where it does not on SSR/CRPS. It does NOT for the overdispersed backbones - SIGK tells the SAME
story as SSR, more sharply. SIGK@240h (7-var x 2-level, 128 pixels, seed 42; more-negative = better):

| model | frozen | refresh (reseed) | delta | verdict |
|-------|-------:|-----------------:|------:|---------|
| SFNO (underdisp)  | +1.0  | -13.21 | -14.2 | refresh helps hugely (worst post-hoc -> better than frozen aifs_perturbed -12.2) |
| Aurora (overdisp) | -9.9  | +4.85  | +14.7 | refresh HURTS badly |
| AIFS (overdisp)   | -12.2 | -10.17 | +2.0  | refresh slightly worse |

ES@240 frozen (aurora 1.117, aifs 1.054, sfno 1.156) vs reseed (aurora 1.136, aifs 1.048, sfno 1.151):
Aurora ES worse (+0.019), AIFS/SFNO ~flat. VS@240 reseed: aurora 0.092, aifs 0.086, sfno 0.096.

Implication: no hidden path-score benefit for Aurora/AIFS; SIGK exposes the refresh damage more clearly
than SSR. This independently VALIDATES the AIES-review M1 fix (appendix C SIGK clause rescoped from "the
weight-space backbones" to "the underdispersed SFNO"): the review flagged it as an unmeasured assertion;
it is now measured and correct. No paper change required - the reframe already tells this on SSR+CRPS.
Ephemeral compute script: /iopsstor/scratch/cscs/sadamov/tmp/submit_reseed_mvar.sh; CSVs in
scratch/table_metrics/sigk_*_p6c_reseed_L{120,240}.csv.

(paper-process detail kept in the local notes)
