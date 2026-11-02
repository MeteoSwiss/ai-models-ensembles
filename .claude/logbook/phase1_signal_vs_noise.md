---
name: phase1-signal-vs-noise
description: "Diagnostic procedure for distinguishing signal degradation from noise injection in Phase 1 weight-perturbation ensembles, linking to Phase 3 motivation"
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Phase 1 diagnosis: signal degradation vs noise injection

## Observation (2026-05-18)

With increasing sigma in Phase 1:
- Extreme wind components become *less* frequent (tails of wind distribution
  shrink).
- Energy density across scales becomes *larger*.

## Interpretation

These are consistent and both expected: weight perturbation displaces the
operator from f* (optimum) to f* + epsilon(dW). Two simultaneous effects:

1. **Signal degradation**: perturbed operator is less precise at producing
   sharp/coherent features. Extremes (a tail behavior requiring full model
   capacity) get smoothed out toward operator-mean behavior.
2. **Noise injection**: perturbation propagates through ~100+ nonlinear
   layers, accumulating to ~sqrt(N_layers) * sigma output noise. Random
   spatial structure adds energy at scales where noise survives the
   operator's filtering -- typically near the grid scale.

So shrinking wind extremes + rising small-scale energy is the
**signal-degradation-plus-noise-injection** signature. Two sides of the
same coin.

## Diagnostic procedure

1. **Spectrum location of extra energy** (decisive test):
   - Near-Nyquist (50-100 km on 0.25 deg grid): almost certainly noise.
     Real atmosphere has steep ~k^-3 KE spectrum -- anything flatter is
     non-physical.
   - Mid-scales (500-2000 km): could be legitimate inter-member variability.
   - Large scales (>3000 km): synoptic-pattern spread, what we actually
     want.
   - Plot truth spectrum, deterministic spectrum, perturbed ensemble-mean
     spectrum on same axes.

2. **Per-member RMSE vs deterministic**:
   - RMSE up at each member -> signal degradation dominant.
   - RMSE same as deterministic -> useful spread without signal loss.

3. **Spread/skill ratio (SSR) at long lead times**:
   - Approaches 1.0 -> well-calibrated perturbation.
   - Far below 1 despite extra small-scale energy -> spread is in *wrong
     directions* (orthogonal to error modes), i.e. noise, not skill.

4. **CRPS vs single-value CRPS from deterministic**:
   - Perturbed CRPS worse than deterministic -> perturbation is
     net-negative; pure damage with no probabilistic benefit.

## Why this matters for [[phase3-physical-perturbation]]

If Phase 1 small-scale energy increase is dominantly noise, that strengthens
the case for Phase 3: targeting coarse-scale parameters (SFNO low-l, Aurora
bottleneck, GraphCast long-edges) should produce synoptic-pattern ensemble
spread *without* near-grid noise -- because we're not touching the operators
that produce fine detail.

**Testable prediction for the paper**: at fixed CRPS improvement, Phase 3
should produce less near-grid spectral energy than Phase 1 layer=all. If
this prediction holds, that's a clean result for the paper.
