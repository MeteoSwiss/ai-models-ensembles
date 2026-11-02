---
name: phase6d-higher-anchor
description: "2026-06-02 afternoon: SFNO refresh-N sweep at LIFTED variance-budget anchors (above production sigma_frozen=0.25 which is itself underdispersed). 16 inferences, 4 inits x 4 (sigma, N) cases. Combined with Phase 6c, gives 4 points per refresh-N to fit SSR(sigma) curves on both pointwise and spatial-mean axes."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why a higher anchor

Phase 6c anchors at production `sigma_frozen = 0.25` and uses `sigma_N = 0.25 * sqrt(40 / N)` for refresh-N variants:

- N=20: sigma = 0.35
- N=10: sigma = 0.50
- N=1 (Phase 6 fresh): sigma = 1.58
- N=40 (frozen): sigma = 0.25

Problem: sigma=0.25 frozen is itself underdispersed -- production sfno_modes10 has per-pixel SSR=0.85, spatial-mean SSR=0.70 (see [[metric-audit]] for the canonical numbers). Anchoring at 0.25 inherits that under-dispersion across all refresh intervals.

To bring SSR toward 1 on both axes (the user's calibration target -- both per-pixel AND spatial-mean SSR are optimal at 1, NOT 0; see the user-correction noted in [[metric-audit]]), the anchor needs to be lifted. Phase 6d picks two lifted anchors (factor 1.6x and 2.0x) and applies the same `sigma_N = sigma_p' * sqrt(T/N)` rule.

## The four cases

Anchored at the same T=40 SFNO-step rollout (lead 240 h):

| Anchor | N | sqrt(T/N) | sigma_N |
|---|---|---|---|
| Option A (sigma_p'=0.40) | 20 | 1.41 | 0.57 |
| Option A | 10 | 2.00 | 0.80 |
| Option B (sigma_p'=0.50) | 20 | 1.41 | 0.71 |
| Option B | 10 | 2.00 | 1.00 |

User authorised both options ("both") so 16 inferences (4 inits x 4 cases) submitted as 2455708-23.

## Combined sweep (6c + 6d) for SSR(sigma) curves

| Refresh-N | sigmas available (post 6c + 6d) | Anchors |
|---|---|---|
| N=20 (every 120 h) | 0.35, 0.57, 0.71 | 0.25, 0.40, 0.50 |
| N=10 (every 60 h) | 0.50, 0.80, 1.00 | 0.25, 0.40, 0.50 |
| N=1 (fresh-per-step, Phase 6) | 0.60, 1.58, 1.93 | various |
| N=40 (frozen-per-member, Phase 3/6) | 0.25, 0.60, 1.93 | various |

3 sigma points per refresh-N for N=20 and N=10 lets us fit SSR(sigma) per axis and read off the per-pixel and spatial-mean SSR=1 crossings without extrapolating blindly. Combined with the Phase 6 fresh/frozen endpoints, the SSR(sigma, N) surface gets enough coverage for a 2D contour plot of "where does SFNO hit SSR=1 on both axes simultaneously".

## Status

- Inference submitted 2026-06-02 afternoon as 2455708-23 via `/iopsstor/scratch/cscs/sadamov/submit_phase6d_sfno_higher_anchor.sh`
- 15 of 16 COMPLETED clean (~9 min each)
- 1 FAILED: `mag_0.71` 20230815 (job 2455722, 7:41 wall) -- the cursed SFNO init that hits state-accumulation SIGSEGV. Resubmitted as 2455790.
- Eval batch (phase6c + phase6d, 6 case tags) submitted 2026-06-02 as 2455778-83 via `/iopsstor/scratch/cscs/sadamov/eval_phase6cd_sfno.sh`. Job 2455781 (mag_0.71 refresh20) launched with 3 inits because 20230815 was still in flight; re-eval after 2455790 lands.

## Results (per-pixel, 2026-06-02 evening)

Eval batch 2455778-83 + 2455790 resubmit all COMPLETED. Per-pixel SSR + CRPSS at lead 240 h (4-init ablation grid, 7 paper vars):

| Family | sigma | N | CRPSS | SSR_pix |
|---|---|---|---|---|
| production frozen (Phase 3) | 0.25 | 40 | **0.310** | 1.098 |
| Phase 3-bis K=1 skip l=0 | 0.25 | 40 | 0.309 | 1.099 |
| Phase 6c (anchor 0.25) | 0.35 | 20 | 0.308 | 1.167 |
| Phase 6c (anchor 0.25) | 0.50 | 10 | 0.298 | 1.197 |
| Phase 6d A (anchor 0.40) | 0.57 | 20 | 0.243 | 1.287 |
| Phase 6d A (anchor 0.40) | 0.80 | 10 | 0.131 | 1.265 |
| Phase 6d B (anchor 0.50) | 0.71 | 20 | 0.155 | 1.480 |
| Phase 6d B (anchor 0.50) | 1.00 | 10 | -0.067 | 1.488 |
| Phase 6 fresh | 0.60 | 1 | 0.212 | 0.900 |
| Phase 6 fresh | 1.93 | 1 | -1.956 | 2.016 |

**Per-pixel verdict:**

1. **Phase 6c preserves CRPSS within ~0.012 of production** (0.308-0.298 vs 0.310). Variance-budget refresh-N is a free metric-equivalent transform along the temporal-correlation axis IF you stay at production's anchor sigma_p=0.25.
2. **Phase 6d (lifted anchors) costs CRPSS aggressively** -- each +0.10 sigma_p' step costs ~0.06 CRPSS at N=20 and ~0.10 at N=10. By sigma_p'=0.50 N=10 we're below climatology (CRPSS=-0.067).
3. **Pure fresh sigma=1.93 (variance-budget equivalent of frozen 0.60) is catastrophic** (CRPSS=-1.96). The variance-budget rule is **only approximately linear up to sigma ~ 0.5**; lifting the anchor past that breaks the assumption.
4. **All Phase 6c/6d points have SSR_pix > 1** on the 4-init grid. Extrapolation of the refresh-N series to SSR_pix=1 gives **sigma ~ 0.16** for both N=10 and N=20, consistent with the variance budget. But production sigma=0.25 is already CRPSS-optimal at SSR=1.10 -- so the "SSR=1 crossing" is not the right paper target; what matters is whether spatial-mean SSR moves toward 1 at constant CRPSS.

The spatial-mean half of the picture is pending: standalone `tools/spatial_mean_ssr.py` batch 2459717 (12 SFNO configs + skip1). Until that lands we cannot pick a Phase 6 winner for the paper.

CSV: `/iopsstor/scratch/cscs/sadamov/sfno_full_ablation_240h.csv`.

## Joint per-pixel + spatial-mean SSR (2026-06-03 morning, batch 2459717 DONE)

| Family | sigma | N | CRPSS | SSR_pix | SSR_spatial |
|---|---|---|---|---|---|
| Production frozen (prod) | 0.25 | 40 | **0.310** | 1.098 | 0.875 |
| Phase 3-bis K=1 skip l=0 | 0.25 | 40 | 0.309 | 1.099 | 0.930 |
| **Phase 6c** (anchor 0.25) | 0.35 | 20 | 0.308 | 1.167 | **0.986** |
| **Phase 6c** (anchor 0.25) | 0.50 | 10 | 0.298 | 1.197 | 1.080 |
| Phase 6d A (anchor 0.40) | 0.57 | 20 | 0.243 | 1.287 | 1.396 |
| Phase 6d A (anchor 0.40) | 0.80 | 10 | 0.131 | 1.265 | 1.012 |
| Phase 6d B (anchor 0.50) | 0.71 | 20 | 0.155 | 1.480 | 1.657 |
| Phase 6d B (anchor 0.50) | 1.00 | 10 | -0.067 | 1.488 | 0.867 |
| Phase 6 frozen (Phase 6) | 0.60 | 40 | 0.211 | 1.352 | 1.011 |
| Phase 6 fresh (Phase 6) | 0.60 | 1 | 0.212 | **0.900** | 0.462 |
| Phase 6 fresh (Phase 6) | 1.93 | 1 | -1.96 | 2.016 | 0.608 |

**Headline finding: Phase 6c sigma=0.35 N=20 is the joint winner for SFNO calibration.** Spatial-mean SSR jumps from production 0.875 -> 0.986 (1.4% from optimal) at CRPSS cost of 0.002 (0.6%). The variance-budget refresh-N rule successfully repackages the noise budget in time without harming per-pixel CRPSS, and recovers nearly all of the spatial-mean coherence gap. The variance-budget hypothesis from [[phase6-fresh-per-step-weight]] is fully confirmed.

**Phase 6d (lifted anchors) is uniformly worse than Phase 6c on the joint criterion** -- either CRPSS collapses (sigma_p'=0.50 N=10 hits CRPSS=-0.067) or spatial-mean SSR overshoots (sigma_p'=0.40 N=20 has SSR_sp=1.396). Phase 6d should be dropped from the paper.

**Phase 6 fresh sigma=0.60 over-corrects spatial-mean** (0.462 vs target 1.0) -- pure-fresh decorrelation kills global-mean coherence. Refresh-N is the sweet spot.

**Phase 6 frozen sigma=0.60** hits SSR_sp=1.011 (slightly better than Phase 6c on this axis alone) but pays 0.099 CRPSS -- not a free win.

Full Tab. 3-style multi-metric CSV with all 7 columns + spatial-mean SSR: `/iopsstor/scratch/cscs/sadamov/full_table3_sfno_240h.csv`. Joint SSR-only CSV: `/iopsstor/scratch/cscs/sadamov/phase6_joint_ssr_240h.csv`.

## Phase 6e (stacked refresh20 + skip l=0): in flight

`sigma=0.35, mode_cut=10, refresh_every=20, mode_skip=1`. Combines Phase 6c best config with Phase 3-bis K=1 mode skip in the runtime SFNO_FRESH hook (env var SFNO_FRESH_MODE_SKIP=1, plumbed via commit `6b1eadc`). 4 inits jobs 2461050-53, eval 2461054 (dependency=afterok). Expected result: SSR_spatial near 1.00 with CRPSS within 0.005 of production if the two refinements are independent; near Phase 6c's 0.986 if redundant. Both refinements move spatial-mean SSR in the same direction (K=1 alone: 0.875 -> 0.930; Phase 6c alone: 0.875 -> 0.986), so we expect additive stacking around 0.999-1.05 -- close to (but not necessarily through) the SSR_sp=1 mark.

## Phase-6-family cross-config intercomp v1 (2026-06-03 morning, COMPLETED)

11-config SwissClim intercomparison (jobs 2461058 + 2461059) at `$STORE/ablation/phase6_family/sfno/intercomparison/`. Completed in 17-19 min each. All modules (probabilistic, deterministic, energy_spectra, ssim, fss, wd_kde, multivariate, ets, maps) ran. Per-config plots + pooled CSVs.

**Key cross-config CSV**: `energy_spectra/lsd_metrics_banded_lead_time_combined.csv` gives LSD per (model, lead, variable, band) for the 4 standard spectral bands (planetary, synoptic, upper_mesoscale, lower_mesoscale). This is what reveals the band-specific story that per-pixel metrics hide.

## LSD per spectral band at lead 240 h (4 inits, 7 paper vars)

| Config | planetary | synoptic | upper_mesoscale | lower_mesoscale |
|---|---|---|---|---|
| Production 0.25 frozen | 0.133 | 0.059 | 0.047 | 0.272 |
| Phase 3-bis K=1 skip l=0 | **0.091** | 0.059 | 0.048 | 0.290 |
| Phase 6c 0.35 N=20 | 0.101 | 0.069 | 0.051 | 0.292 |
| Phase 6c 0.50 N=10 | 0.111 | 0.073 | 0.060 | 0.315 |
| Phase 6 frozen 0.60 | 0.113 | 0.067 | 0.068 | 0.340 |
| Phase 6 fresh 0.60 | **0.080** | 0.104 | 0.087 | 0.322 |
| Phase 6d A 0.57 N=20 | 0.117 | 0.094 | 0.090 | 0.336 |
| Phase 6d A 0.80 N=10 | 0.171 | 0.154 | 0.144 | 0.430 |
| Phase 6d B 0.71 N=20 | 0.065 | 0.154 | 0.156 | 0.377 |
| Phase 6d B 1.00 N=10 | 0.174 | 0.121 | 0.177 | 0.579 |
| Phase 6 fresh 1.93 | 1.638 | 1.238 | 0.295 | 0.744 |

**Spectral signatures interpreted:**

1. **Phase 3-bis K=1 cuts planetary LSD by 32%** (0.133 -> 0.091) while leaving synoptic / mesoscale identical to production. This is the textbook qualitative signature of "stopped perturbing l=0 (global constant mode)" -- only the planetary band should change, and only the planetary band did. The K=1 hypothesis is now confirmed in spectral space, not just in spatial-mean SSR.

2. **Phase 6c 0.35 N=20 trades planetary for synoptic**: planetary 0.133 -> 0.101 (improvement) but synoptic 0.059 -> 0.069 (modest degradation). The N=20 refresh interval corresponds to a ~120 h ≈ 5-day cycle = mid-synoptic timescale, so the noise injection has a synoptic spectral footprint -- as designed by the variance-budget rule, not a bug.

3. **Phase 6 fresh sigma=0.60 has the best planetary band (0.080) but the worst synoptic (0.104) and upper-mesoscale (0.087) among the non-catastrophic configs.** This is the spectral explanation for spatial-mean SSR collapse (0.462): fresh-per-step injects high-frequency synoptic noise at the 6 h step, destroying the global-mean coherence. Pure fresh is the wrong design point for SFNO.

4. **Phase 6d all damage multiple bands.** Lifted-anchor variance-budget breaks past sigma_p ~ 0.40, with synoptic + upper_mesoscale + lower_mesoscale all degrading 2-3x vs production at sigma_p'=0.50.

5. **Lower-mesoscale band is the SFNO modes10 architecture floor** (0.27-0.58 across all configs). No Phase 6 variant moves it. Improving lower mesoscale would require a different perturbation strategy (higher mode cut, or finer noise-injection point).

## Publication figure

[`figures/sfno_phase6_lsd_bands.pdf`](../../../../pyprojects/ai-models-ensembles/figures/sfno_phase6_lsd_bands.pdf) (+ `.png`). 4-panel horizontal bar chart: one panel per spectral band, configs sorted by overall ranking, production-frozen reference shown as dashed line. Script: `/iopsstor/scratch/cscs/sadamov/plot_sfno_phase6_lsd_bands.py`.

## Phase-6-family intercomp v2: dependency-queued

12-config intercomp adds `mag_0.35_modes10_refresh20_0.35_skip1` (Phase 6e stacked). Jobs 2461574 (non-prob) + 2461575 (prob), both `dependency=afterok:2461054` so they auto-fire when the Phase 6e eval completes (~5 h after this update). Output: `$STORE/ablation/phase6_family_v2/sfno/intercomparison/`. Once landed, re-running `plot_sfno_phase6_lsd_bands.py` with the v2 path will produce a figure including the stacked row.

## Output paths

`$STORE/ablation/phase3/sfno/<init>/mag_<sigma>_modes10_refresh<N>_<sigma>/forecast.zarr`

Eval CSVs under `$STORE/ablation/phase3/sfno/eval/mag_<sigma>_modes10_refresh<N>_<sigma>/`.

## Related

- [[phase6-fresh-per-step-weight]] -- the original Phase 6 fresh-vs-frozen motivation (random-walk sqrt(T) suppression of spatial-mean spread).
- [[sigma-sweep-design]] -- variance-budget anchor logic + SSR-target-aware sigma brackets.
- [[metric-audit]] -- SSR optimal=1 (user correction), spatial-mean vs per-pixel dichotomy.
- [[spatial-mean-vs-pointwise-ssr]] -- the original finding this Phase 6 line is trying to fix.
