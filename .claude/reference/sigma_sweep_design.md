---
name: sigma-sweep-design
description: How we pick sigma values across ablation phases. Variance-budget anchoring + SSR-target-aware bracket design + log-sigma interpolation for sigma selection. Used by Phase 2b and Phase 6.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Sigma sweep design (variance budget + SSR-target bracket)

Single unifying principle for sigma selection across Phases 2, 3, 3b, 6: **hold a Gaussian variance budget constant against a known-calibrated reference, then refine in a bracket that targets SSR=1**.

## 1. Variance-budget anchor

For Gaussian per-parameter noise on a parameter group of size N at per-parameter std sigma_group, the total injected variance is ~ N·sigma_group^2. To compare configurations fairly we hold that total constant against a full-weight reference at sigma_full = 0.01:

```
N_total · sigma_full^2  =  N_group · sigma_group^2
sigma_group  =  sigma_full · sqrt(N_total / N_group)        (spatial budget)
```

Anchored at **sigma_full = 0.01** (picked because Phase 1 showed graphcast all-layers calibrated best near 0.01; also a "moderate" value across the board). This is the sqrt(N) anchor used in Phase 2.

For Phase 6 (fresh-per-step weight perturbation) the analogous **temporal** budget holds cumulative pointwise variance over T rollout steps constant:

```
T · sigma_persistent^2  =  T · sigma_fresh^2 / T          (random-walk decorrelation, persistent -> fresh)
sigma_fresh  =  sigma_persistent · sqrt(T)                (temporal budget)
```

At T=60 (10-day rollout, 4h step) this gives `sigma_fresh ≈ 7.7 · sigma_persistent`.

These two rules cover every sigma value we've used or plan to use. Phase 1 is the exception -- it sweeps sigma_full itself on a log-grid {0.001, 0.003, 0.01, 0.03, 0.1} to identify the calibration window (factor sqrt(10) between adjacent).

## 2. SSR-target bracket (used by Phase 2b, planned for Phase 6)

### Reverse-engineered Phase 2b bracket design (verified 2026-05-31)

The actual bracket-picking method used in Phase 2b (`submit_ablation.sh:172-181`, original analysis script at `/tmp/sadamov_optimal_sigma.py` no longer exists):

1. Phase 2 anchor produces one observed (sigma, SSR) point per (model, group).
2. Bracket direction is chosen from `SSR_anchor` vs 1:
   - `SSR_anchor > 1` (over-dispersive): place "below" multiplier ~0.5-0.65x of anchor, place "above" multiplier ~1.4-1.7x (mostly to confirm slope).
   - `SSR_anchor < 1` (under-dispersive): SKIP the "below" multiplier (it would push deeper under-dispersion); place both points above, at ~1.4x and ~1.85x (SFNO-decoder pattern).
3. The bracket was designed to **span SSR = 1 in the swept range** when possible. Verified per-group SSR @ lead 240h:

| model | group | anchor (sigma, SSR) | below (sigma, SSR) | above (sigma, SSR) | bracket spans SSR=1? |
|---|---|---|---|---|---|
| aurora | encoder | (0.044, 1.22) | (0.025, 1.06) | (0.060, 1.32) | barely |
| aurora | decoder | (0.062, 1.13) | (0.040, 1.04) | (0.085, 1.19) | barely |
| graphcast | g2m | (0.027, 1.51) | (0.014, 1.31) | (0.045, 1.69) | **NO** -- bracket too tight, SSR stayed >1 across all 3 |
| graphcast | m2g | (0.030, 1.15) | (0.018, 0.87) | (0.050, 1.42) | yes |
| sfno | encoder | (0.054, 1.14) | (0.035, 1.02) | (0.080, 1.21) | barely |
| sfno | decoder | (0.054, 0.90) | (0.075, 1.05) | (0.100, 1.16) | yes (both above the anchor) |

GraphCast g2m is the lesson: the chosen "below" sigma (0.014, 0.52x of anchor) was not low enough to cross SSR=1 because the per-group slope is shallower than expected. We ate the over-dispersion and picked the bracket winner on argmin-CRPS instead.

### Per-group SSR-vs-log(sigma) slope (verified 2026-05-31)

Computed from the three-point bracket per group:

```
slope = (SSR_above - SSR_below) / (log(sigma_above) - log(sigma_below))
```

| model | group | slope |
|---|---|---|
| aurora | encoder | 0.30 |
| aurora | decoder | 0.20 |
| graphcast | g2m | 0.32 |
| graphcast | m2g | 0.54 |
| sfno | encoder | 0.23 |
| sfno | decoder | 0.38 |

Mean per-group slope ~ **0.3** in SSR per unit log-sigma. This is **much shallower than the Phase 1 all-weights slope** (Phase 1 σ swept 0.001 -> 0.1 = 4.6 log units, with SSR ranging ~0.05 -> ~2.5 = factor ~50x → slope ~0.5-0.8 depending on model). Per-group perturbation is structurally less spread-effective than all-weights perturbation -- the unperturbed weights "absorb" some of the noise during propagation.

**Implication for bracket multipliers**: with slope ~0.3, to cross SSR=1 you need to move log-sigma by `|SSR_anchor - 1| / 0.3`. For an anchor at SSR=1.5 (GraphCast g2m), that's log-multiplier ~-1.67 → sigma multiplier `exp(-1.67) ≈ 0.19x`. The actual "below" choice of 0.52x was 3x too high -- explains why the bracket failed to cross SSR=1.



After the variance-budget anchor, we refine in a 3-sigma bracket around it: **{below, anchor, above}**. The bracket is **deliberately asymmetric** and **shifted toward the predicted SSR=1 crossing**, not symmetric around the anchor.

Empirical Phase 2b bracket multipliers (anchor = 1.0):

| Model | Group | below | anchor | above |
|---|---|---|---|---|
| aurora | encoder | 0.57x | 1.0 | 1.36x |
| aurora | decoder | 0.65x | 1.0 | 1.37x |
| graphcast | g2m | 0.52x | 1.0 | 1.67x |
| graphcast | m2g | 0.60x | 1.0 | 1.67x |
| sfno | encoder | 0.65x | 1.0 | 1.48x |
| sfno | decoder | **1.39x** | **1.85x** | (no below-anchor point -- both above) |

So the typical bracket is `{0.5-0.65x, 1x, 1.4-1.7x}`, NOT the symmetric `{0.5x, 1x, 1.5x}` I'd been describing in conversation. SFNO decoder is the special case where Phase 2 SSR at the anchor was so underdispersive (much less than 1) that both refinement points sit ABOVE the anchor to reach SSR=1.

**Lesson for future sweeps**: design the bracket *after* observing the anchor's SSR. Don't pre-commit to a symmetric bracket if the data tells you SSR=1 is far from the anchor.

## 3. Log-sigma interpolation for picking the next sigma

To pick the bracket bounds (the "below" and "above" multipliers), we interpolate the SSR-vs-log(sigma) curve from the immediately-adjacent already-swept points. SSR is approximately linear in log(sigma) within the calibration window, so a 2-point fit suffices:

```
log(sigma_target)  =  log(sigma_a) + (log(sigma_b) - log(sigma_a)) · (SSR_target - SSR_a) / (SSR_b - SSR_a)
```

For sigma selection in Phase 2b we used `SSR_target = 1.0` to estimate the calibrated sigma, then placed the bracket points ~0.5-0.65x and ~1.4-1.7x of the *anchor* to span SSR=1 with margin on both sides (catching curvature in the SSR curve and giving 2 cells on the "lower-CRPS-but-overdispersive" side that often wins on argmin-CRPS).

**Caveat documented in [[calibration-winners]]**: at 4 inits the SSR curve is noisy enough that linear interpolation can mislead. For final winner selection we used **argmin-CRPS**, not interp-to-SSR=1. But for **sigma-bracket design** (i.e. where to place the next sweep cells), interp is still informative -- we just don't *trust the interpolated SSR=1 sigma as the final answer*, we use it to bracket.

## 4. Phase 6 refined plan (fresh-per-step weight perturbation)

See [[phase6-fresh-per-step-weight]] for the full Phase 6 plan. The refined sigma-selection process:

**Step 1: Variance-budget anchor**
For each of aurora_encoder, sfno_modes10, aifs_<winner>, compute the temporal-budget anchor:

```
sigma_anchor = sigma_persistent · sqrt(60)  ≈  sigma_persistent · 7.7
```

| Model | sigma_persistent | sigma_anchor (T=60) | risk |
|---|---|---|---|
| aurora_encoder | 0.025 | 0.19 | safe |
| sfno_modes10 | 0.25 | 1.93 | the watchpoint (spectral conv weights may overflow at sigma~2) |
| aifs_<winner> | TBD (await Phase 1+2 eval) | ~7.7 · winner | safe |

GraphCast already has fresh-noise data from Phase 3 (the n_coarse=42 activation hook is fresh-per-step by construction), so Phase 6a uses that as the smoking-gun comparison vs the frozen pilot.

**Step 2: Diagnostic probe (Phase 6a-prime, 30 forecasts)**
Before committing to a 3-sigma sweep, run ONE cell per model at `sigma_anchor` to observe the SSR:
- aurora_encoder at 0.19, 4 inits × 10 = 40 forecasts
- sfno_modes10 at 1.93 (or 1.0 if unstable), 40 forecasts
- aifs_<winner> at 7.7·winner, 40 forecasts

Total: ~120 forecasts. Read SSR_anchor at lead 240 h.

**Step 3: Bracket design (SSR-aware)**
Then place the bracket using SSR_anchor:
- If SSR_anchor ≈ 1 (within ~0.1): bracket symmetrically, `{0.6x, 1x, 1.5x}`.
- If SSR_anchor > 1 (overdispersive): bracket below, `{0.4x, 0.7x, 1x}`.
- If SSR_anchor < 1 (underdispersive): bracket above, `{1x, 1.5x, 2.5x}` (the SFNO-decoder pattern).

**Step 4: 3-sigma sweep (Phase 6b proper, ~360 forecasts)**
The bracket × 4 inits × 10 members = 120 forecasts per model × 3 models = 360 forecasts.

**Step 5: Argmin-CRPS within the bracket (final pick per model)**
Same decision rule as [[calibration-winners]] Stage 2: argmin-CRPS at lead 240 h, with multi-metric inspection (CRPS + SSR + LSD + FSS) as a tiebreaker.

## 5. Cross-phase comparability

Three things keep the Phase 6 sweep apples-to-apples vs Phases 1/2/3:

- Each model's anchor uses its own `sigma_persistent · sqrt(T)`, so we're not picking arbitrary absolute sigmas.
- The bracket multipliers come from the same SSR-target-aware rule we used in Phase 2b.
- Same 4 inits, same 10 members as Phases 1/2/3, so SSR diagnostics and CRPS compare cleanly against existing production baselines.

This way Phase 6 sits as a new column in the same Tab. 1 row structure rather than a separate study.

## Related
- [[paper-phase-structure]] -- the 4-phase narrative.
- [[calibration-winners]] -- Stage 1 + Stage 2 winner selection rule, "why argmin CRPS not interp-to-SSR=1" discussion.
- [[phase6-fresh-per-step-weight]] -- Phase 6 plan in detail.
- [[spatial-mean-vs-pointwise-ssr]] -- the spatial-mean coherence problem that Phase 6 is designed to fix.
