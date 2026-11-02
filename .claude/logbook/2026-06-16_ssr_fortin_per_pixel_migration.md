# 2026-06-16: per-pixel SSR migrated WB2 unbiased-MSE -> Fortin (M+1)/M everywhere

One-line: switched SwissClim per-pixel SSR from WB2 `UnbiasedSpreadSkillRatio` (debias the MSE by
s^2/M, with a robust >=0 filter) to a `FortinSpreadSkillRatio` (inflate ddof=1 variance by (M+1)/M,
plain ensemble-mean MSE denominator), so BOTH SSR streams (per-pixel + spatial-mean) now use the same
Fortin finite-M correction. Recomputed every per-pixel SSR number in the paper from forecast.zarr.

## Why
User asked: "are all SSR Fortin compliant?" -> no, per-pixel used the WB2 debiased-MSE form (a different
but also finite-M-unbiased correction), spatial-mean (tools/spatial_mean_ssr.py) used explicit (M+1)/M.
Decision: converge on Fortin everywhere. Fortin is the canonical/citeable form and is bounded (the WB2
robust >=0 filter biases SSR DOWNWARD - verified: reliable ensemble reads 0.867 under WB2-robust vs 1.000
under Fortin). Per-pixel is the STANDARD metric (WB2/ECMWF); spatial-mean stays the purpose-built
coherent-shift diagnostic confined to Sec 4.3.

## Code (SwissClim research branch, swiss-ai/SwissClim_Evaluations)
- `src/swissclim_evaluations/metrics/probabilistic/calc.py`: added `FortinInflatedEnsembleVariance`
  (var*(M+1)/M, distinct unique_name so it can't be conflated with plain EnsembleVariance),
  `PlainEnsembleMeanSquaredError` (squeezes size-1 ensemble dim off targets like the Robust class, returns
  (ensmean-y)^2 with NO debias), `FortinSpreadSkillRatio` (subclasses UnbiasedSpreadSkillRatio, uses those
  two stats, _values_from_mean = sqrt(var/mse)). `wbx.py:103` "SSR" -> FortinSpreadSkillRatio.
- Commits: `ab028db` (first version, BROKEN - subclassed plain SpreadSkillRatio whose SquaredError+EnsembleMean
  wrapper dies on size-1 ensemble targets), `9bc17e5` (fix). BOTH PUSHED to research (user authorised: only
  they use research). RobustUnbiasedSpreadSkillRatio left defined-but-unused (don't delete; reversible).
- swissclim is an editable install in the host venv, so reruns pick up the edit automatically.

## Magnitude
Fortin per-pixel SSR is systematically ~9-10% higher than WB2 (more for SSR near 1, less for very
under-dispersed low-SSR rows). Data-dependent, NOT a constant rescale (the WB2 robust filter is data-dependent),
so it CANNOT be scalar-rescaled like the 2026-06-13 spatial-mean Fortin fix - required full recompute from zarr.

## Eval reruns (all prob-only, modules: probabilistic only; outputs to SCRATCH, non-destructive)
Scratch tree: `/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/ssr_fortin_regen/` (+ pilot in
`ssr_fortin_pilot/`). Generator `gen_fanout.py` in the pilot dir. CRPS/PIT come out identical (only SSR
code changed); nothing in $STORE touched; no forecast.zarr touched.
- 17 ablation calibration-table tags (4-init each): jobs 2543398-2543415, ALL COMPLETED (~1h each).
  Outputs: `ssr_fortin_regen/<model_id>__<phase>__<tag>/probabilistic/ssr_summary_averaged_*.csv`.
- 8 Phase 5 baselines (112-init, 360h): aifs_perturbed 2543432 + aifs_perturbed_ic 2543433 (running);
  6 backbones aurora_encoder/graphcast_all/sfno_modes10 {weight-only, _ic}. The 3 WEIGHT-ONLY first
  attempts (2544223/5/7) FAILED in 27s: stale `swissai/a122` zarr paths in their
  eval_probabilistic_only_config.yaml (migrated to mch/s83). Resubmitted with fixed paths:
  aurora_encoder 2544283, graphcast_all 2544284, sfno_modes10 2544285. _ic variants 2544224/6/8 fine.
  STILL IN FLIGHT at umtrac time: 2543432, 2543433, 2544224, 2544226, 2544228, 2544283, 2544284, 2544285.

## Aggregation
Calibration-table SSR recomputed with the var7/level-mean convention (calibration_crpss.py: 2D vars direct,
3D vars level-mean over 500/850 first, then mean of 7 paper vars). NOTE: this gives ~0.05 different WB2
numbers than the historical 2026-05-31 table column (which used an undocumented slightly-different agg) -
the WHOLE SSR column was therefore replaced with the consistent agg under Fortin, not number-matched to old.

## Paper edits DONE (ai-models-ensembles-paper/main.tex, NOT yet committed at umtrac time)
- Methodology: SSR^pix prose (1575-1579) + summary-table formula (1614) -> Fortin (M+1)/M, plain MSE
  denominator (was: subtract s^2/M).
- figures/calibration_basis_table.tex (MAIN repo): all 17 SSR cells -> Fortin values + recomputed bold
  (argmin|x-1| within model per lead) + provenance comment.
- Winner-selection prose: aurora P2b (1.05,1.06)->(1.14,1.16); GraphCast P1 "SSR~1 at 72-120h (0.95,1.13)";
  SFNO P3 "SSR=1.10 at 240h"->1.22.
- Per-pixel claim 784: "well-calibrated for all three" -> "stays close to 1 ... mild under->mild over".

## Phase 5 prose DONE (2026-06-16, paper commit d94e030, all 8 baseline jobs landed)
All 8 Fortin baselines COMPLETED (2544284 graphcast_all weight-only was last, 20:44). Recomputed from
the scratch ssr_line_<var>_by_lead csvs (host venv pandas, read directly; did NOT use plot_ssr_phase5_aifs.py).
KEY FINDING: under Fortin the per-pixel SSR sits ~10% higher, so the long-lead AIFS story CHANGED - it is
now mildly OVER-dispersed at 240h, not "~1.0", and IC perturbation DOES mildly inflate the long lead
(+0.09 MSL @240h). No single variable/aggregate preserves the old "~1.0 at 240h, without inflating" story
(the old 0.49/0.67-0.86-0.94-1.00 numbers most closely matched the under-dispersed fields: 10m winds /
specific_humidity_850 / Z-gradients, NOT MSL). User chose MSL anchor + honest rewording (AskUserQuestion).
- 4-backbone MSL SSR@24h weight-only -> ic (gain): aurora 0.63->0.80 (+0.17), graphcast 0.58->0.87 (+0.29),
  sfno 0.68->0.79 (+0.11), aifs 0.65->0.85 (+0.20). So footnote "+0.10..+0.25" -> "+0.11..+0.29" (range = min/max).
- AIFS MSL weight-only SSR(lead): 0.65/0.95/1.07/1.12 @24/72/120/240h. AIFS MSL ic: 0.85/1.15/1.21/1.21.
- Edits: main.tex:1031 footnote (+0.11..+0.29); main.tex:1444-1461 body (MSL anchor, "well-calibrated to
  mildly over-dispersed at long leads (SSR=1.12 @240h)", "lifts SSR at every lead, most at short lead",
  "hybrid trades small long-lead over-dispersion for large early-lead gain"). Recompile clean (18.0MB, exit 0,
  no undefined refs). The ssr_phase5_aifs fig is NOT in the paper (prose only) - not regenerated.
- COMMITTED paper d94e030; staged ONLY my 2 hunks via git apply --cached, leaving a
  concurrent-session main.tex Pasquini2026balances Milton-intro citation + paired paper_refs.bib entry
  UNSTAGED/untouched. NOT pushed (awaiting user).

## NOT mine in the working tree (concurrent/parallel session, do NOT commit under SSR work)
Main repo also shows modified: figures/headline_crpss_vs_lead_8way.pdf, milton_F1/F5/F9 pdf+png,
tier1b pdf, tools/milton/*.py, tools/plot_7way_spatial_mean_ssr.py, plot_headline_crpss_vs_lead*.py,
plot_ssr_phase5_aifs.py; untracked tools/model_colors.py. Left untouched.
