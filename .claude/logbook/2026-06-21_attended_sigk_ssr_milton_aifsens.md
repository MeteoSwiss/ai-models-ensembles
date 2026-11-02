# 2026-06-21 attended follow-up (Opus 4.8): sigk probe + 3 paper fixes (SSR^pix, Milton, AIFS-ENS)

One-line: cleared the 3 user-chosen items from the 2026-06-21 nightly review plus the carried-over sigk
refresh probe. Paper committed `57b4f73`; main repo clean (figures committed by a concurrent
session `0d8083f`). tectonic clean (main.pdf 12.36 MB, 0 undefined). Both LOCAL, NOT pushed.

## Context
After "how did the nightly go" -> reported the 2026-06-21 nightly (exit=0, sentinel written, paper 01406c3
+ main bf52871). User asked to be prompted on all outstanding tasks, then chose: report sigk CSVs, fix
SSR^pix, draft the 2 flagged paper-claim items. Then "yes to all" on applying everything.

## 1. SSR^pix formula (FIXED, main.tex:1693)
> CORRECTION (added at handover, from the 2026-06-22 local notes): this section is WRONG and the edit was
> reverted. The SwissClim SSR^pix path is `wbx.py`, not `calc.py`: the Aggregator reduces over time only
> (`wbx.py:94`, `reduce_dims=[time_dim]`) and the spatial mean is taken afterwards in
> `_save_probabilistic_summaries` (`wbx.py:282-286`). So SSR^pix = `< sqrt(<var>_t / <mse>_t) >_x` (spatial
> mean OUTSIDE the sqrt) and the original appendix formula was correct. Do not "fix" SSR^pix again.

Nightly flag B: the appendix-B SSR^pix row was written as a SPATIAL MEAN OF per-pixel SSRs
(<sqrt(mean_i var / mean_i mse)>_x, sqrt inside, spatial mean outside). VERIFIED the implementation:
`FortinSpreadSkillRatio` in SwissClim `src/swissclim_evaluations/metrics/probabilistic/calc.py:139-146`
returns `sqrt(EnsembleVariance / PlainEnsembleMeanSquaredError)` where both statistics are mean-aggregated
(over space+init) by the WBX aggregator BEFORE the single sqrt -> sqrt(mean_var / mean_mse) = sqrt of ratio
of means. mean(sqrt) != sqrt(mean) by Jensen. Changed L1693 to
`sqrt( <mean_i (M+1)/M s^2>_x / <mean_i (Xbar-y)^2>_x )`, consistent with the SSR^sp row just below.
(spatial_mean_ssr.py:206-208 is the SSR^sp path, also sqrt-of-means; same conclusion.)

## 2. Milton 976 hPa floor (FIXED, main.tex:1032; data-VERIFIED)
Nightly flag C: text said per-baseline deepest min-MSL "spans 942 to 967 hPa" then singled out IFS-ENS
(942) + graphcast_all_ic (948) as the ones crossing the 976 hPa ERA5 floor - but if all are in [942,967]
then ALL cross 976. VERIFIED vs `/iopsstor/scratch/cscs/sadamov/milton_case_study/milton_master_tracks.csv`
(groupby baseline, min psl_hpa): all 12 variants' deepest member < 976 hPa, range 942.4 (ifs_ens) to 967.1
(aifs_perturbed). Reworded to "every baseline's deepest member crosses the 976 hPa floor ... the most
intense being IFS-ENS (942 hPa) and graphcast_all_ic (948 hPa)" - preserves every number.

## 3. AIFS-ENS training cost (FIXED, main.tex:960; source-VERIFIED)
Nightly flag A: "as reported by the source papers, AIFS-ENS is fine-tuned from the deterministic AIFS with
of order 20K additional GPU-hours". Read the full arXiv:2412.15832 PDF (Lang et al., AIFS-CRPS):
- No GPU-hour figure is reported. Table 3: "trained for ~7 days on 128 H100 64GB GPUs" (N320) and "~4 days
  on 64 H100" (O96). 7d x 24 x 128 = 21,504 ~= 21.5K -> "of order 20K" is a derived estimate, not reported.
- NOT fine-tuned from the deterministic AIFS. Sec 2.3: "We train AIFS-CRPS in four stages. The first
  training stage follows Lam et al. [2023]. The model learns to forecast one 6h step (rollout=1)." The only
  "fine-tuning" is the internal 4th stage (full rollout on operational IFS analysis), not warm-start from
  the deterministic checkpoint.
Reframed to "AIFS-ENS's probabilistic training is of order 20K GPU-hours (estimated from the
source-reported ~7 days on 128 H100, not given directly)", matching the safe FCN3/Atlas clauses.
OPEN QUESTION for the user: if their first-hand ECMWF knowledge is that operational AIFS-ENS IS warm-started
from the deterministic checkpoint (preprint describes from-scratch), revert to "fine-tuned" wording.

## 4. sigk_refresh probe (job 2563873, finished Jun 18 21:47) - READ + reported; probe only, NOT in paper
CSVs in `/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/sigk_refresh/`. Signature kernel score,
lower = better.
- AIFS (112-init production): aifs_perturbed (frozen weight-only) -12.24 (= production -12.2); aifs_p6c
  (fresh-per-step) +6.11 (WORSE by ~18); aifs_perturbed_ic = NaN aggregate (running mean poisoned).
- SFNO (3 ablation inits): frozen025 (sigma0.25) +0.67; frozen060 (sigma0.60) -12.02; fresh060 (sigma0.60)
  +9.18; refresh20_035 (sigma0.35) -16.21 (BEST).
- FINDING: fresh-per-step weight noise HURTS sigk for both models. Clean paired contrast (SFNO sigma0.60):
  frozen -12.02 vs fresh +9.18 = +21 worse - continuous re-draw makes temporal paths incoherent, penalised
  by the path-signature kernel. Intermittent refresh-every-20 is sigk-optimal, beating even larger frozen.
  Sweet spot between frozen (over-coherent; why graphcast_all ranks last in the paper) and fresh-every-step
  (over-noisy). Caveats: SFNO is 3 inits not 112; refresh20 uses sigma0.35 vs frozen 0.60 (cadence/magnitude
  confounded); only frozen060-vs-fresh060 is a clean pair. GraphCast refresh = no data on disk.
- aifs_perturbed_ic NaN diagnosis: parsed all 112 per-init SIGK from the .log; 28 are NaN (25%, likely
  missing/NaN IC members for those inits), so plain mean was poisoned. FILLED the CSV cell via nanmean =
  -25.057772 over the 84 finite inits, and set n_inits=84 (was 112) to keep coverage honest. NOT directly
  comparable to frozen's -12.24 (112 inits). For a fair IC comparison: restrict frozen to the same 84 inits
  or chase the 28 NaN inits.

## Commits / state
- paper `57b4f73` "Review fixes: SSR^pix reduction order, Milton 976 hPa floor, AIFS-ENS cost attribution"
- main repo CLEAN; the 28 regenerated figures + ES/VS/SIGK table expansion were committed by a CONCURRENT
  session in `0d8083f` "Figure fixes + ES/VS/SIGK at 120/240h for Tables 2 & 4" (not by this session).
- tectonic main.pdf 12.36 MB, 65 pages, 0 undefined refs, no em/en/double-dashes in edited prose.
- NOT pushed (standing default: user reviews + decides pushes).
