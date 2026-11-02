---
name: phase5-uncertainty-decomposition
description: Phase 5 plan -- decompose forecast variance into epistemic (weight) vs aleatoric (IC) components by isolating noise sources across ensemble runs
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Phase 5: epistemic vs aleatoric uncertainty decomposition

## Goal

Decompose forecast variance into:
- **Epistemic** = uncertainty about model parameters W. Reducible. Captured by
  weight-perturbed ensemble at fixed IC.
- **Aleatoric** = proxy for chaotic atmospheric divergence under uncertain
  initial state. Captured by IC-perturbed ensemble at fixed weights.

Under noise-source independence:
```
Var_total approx Var_epistemic + Var_aleatoric
```

**Why:** most AI weather ensemble papers don't split these. Reporting the
decomposition vs lead time tells the reader whether model error or IC
uncertainty dominates forecast spread at each horizon, and supports the
claim that weight perturbation captures a genuine epistemic-type
uncertainty (not just a different flavour of IC noise).

**How to apply:** at the same 4 init times as Phase 1, run an
aleatoric-only ensemble (IC perturbation only) and (optionally) a both
ensemble. Combined with the existing Phase 1 epistemic-only (weight
perturb only), decompose the variance.

## Experimental design

For each of 3 deterministic models, at the same 4 init times, 10 members:
1. **Epistemic only**: weight perturb sigma_W=0.01, no IC perturb. ALREADY
   DONE in Phase 1 (after the 2026-05-18 re-run).
2. **Aleatoric only**: no weight perturb, IC perturb sigma_IC scaled to
   match ERA5 observation uncertainty (sigma_IC ~= 0.5% to match dust-of-
   ERA5-grid-uncertainty; convention to be set).
3. **Both** (optional): weight + IC perturb simultaneously. Lets us test
   the independence assumption (if independent, Var_total ~= Var_epi +
   Var_alea; if not, the residual reveals the interaction term).

## Cost

- Aleatoric only: 3 models x 4 inits x 10 members = 12 inference runs
  (~5-10 NH).
- + Both: another 12 runs (~5-10 NH).
- Eval: pooled with existing per-config eval (~3-5 NH for the analysis
  notebook on top).
- **Total: ~15-25 NH**.

## Implementation notes

- `perturb_initial_conditions` already exists at
  [ai_models_ensembles/e2s_perturbation.py:32]
  (ai_models_ensembles/e2s_perturbation.py#L32) -- multiplicative Gaussian
  on each IC variable, scaled by per-variable spatial std.
- Need to verify the CLI flag wiring: likely an `--ic-magnitude` option in
  `ai_models_ensembles.cli infer` -- check before submitting.
- Output dirs: `$STORE/uncertainty/aleatoric_only/<model>/<init>/...` and
  `$STORE/uncertainty/both/<model>/<init>/...` (kept separate from
  ablation/).

## Caveats to document in the paper

1. **sigma_IC choice is non-trivial.** ERA5 obs uncertainty is ~0.1-0.5 K
   for T, ~1-2 m/s for winds (Hersbach et al. 2020). Need a single number
   for the multiplicative noise convention. Document the choice.
2. **Decomposition is approximate for nonlinear dynamics.** Past ~5 days
   lead the two noise sources interact through chaotic divergence. The
   "both" experiment quantifies this.
3. **"Epistemic from weight perturbation"** is informal -- multiplicative
   Gaussian prior, not a Bayesian posterior. Fully principled epistemic
   uncertainty requires SWAG / Laplace approximation = [[phase4-fisher-info]]
   territory. Note this limitation.

## Status (2026-05-18)

PLANNED, deferred until Phase 1 re-run analysis is done. Can be slotted
in as a separate "uncertainty decomposition" section in the paper rather
than a numbered phase, since it analyses existing ensembles rather than
introducing a new perturbation strategy.
