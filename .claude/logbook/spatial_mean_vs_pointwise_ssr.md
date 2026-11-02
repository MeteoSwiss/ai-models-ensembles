---
name: spatial-mean-vs-pointwise-ssr
description: "Pointwise SSR / CRPS hide a structural difference between weight-perturbed and trained-probabilistic ensembles: spatial-mean spread is 6-9x larger for our perturbed baselines. Important paper-framing implication."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Finding (verified 2026-05-29 from baselines/intercomparison/ data)

For the 112-init baseline intercomparison, our 3 perturbed baselines (aurora_encoder, graphcast_all, sfno_modes10) and the 4 trained probabilistic / classical baselines (atlas, fcn3, aifsens, ifs_ens):

- **Pointwise SSR** (lead-averaged, var-averaged): all 7 models cluster between 0.73 and 0.87. Perturbed slightly underdispersed at short leads, converging toward 1.0 at long leads. CRPS rankings are competitive (paper headline "5-20% gap").
- **Spatial-mean spread** (std across members of area-weighted spatial mean at lead 360h, U10m): aifsens 0.020, ifs_ens 0.021, atlas 0.053, fcn3 0.063, sfno_modes10 0.052, **aurora_encoder 0.130 (6x aifsens)**, **graphcast_all 0.186 (9x aifsens)**.

The spaghetti plots are HONEST. The two metrics measure orthogonal components of the ensemble variance:

- **Pointwise SSR** = mean over grid of [std over members] -- captures per-cell variance.
- **Spatial-mean spread** = std over members of [mean over grid] -- captures the *coherent whole-field* component of inter-member differences (the DC / low-wavenumber mode).

## Mechanism

Weight perturbation modifies the deterministic forward map uniformly. Each perturbed member's forecast deviates from the unperturbed forecast by a **coherent whole-field shift** -- so the *spatial mean* of the deviation is itself non-zero per member. Different members → different coherent biases → wide spatial-mean spread.

Trained probabilistic models (FCN3 / Atlas / AIFSENS) inject stochasticity at intermediate features (diffusion noise, conditional layer-norm noise). These produce inter-member differences that are **spatially decorrelated** -- members vary in fine-scale detail but their spatial means agree.

## What CRPS actually does (correcting an earlier mis-statement)

CRPS is computed pointwise (per grid cell) and then averaged. It does NOT penalise spatially-correlated inter-member differences -- the integral at each cell is invariant to what happens at other cells. So pointwise CRPS cannot distinguish "10 random per-cell noises" from "10 coherent whole-field shifts" of the same per-cell magnitude.

Trained probabilistic models produce spatially-decorrelated noise NOT because the loss penalises correlated noise, but because of architecture + training mechanism (per-feature noise injection, training samples with no systematic per-instance biases).

## Implications for the paper

1. **Limitation to flag honestly**: "Pointwise CRPS is competitive, but spatial-mean spread is 6-9x larger because weight perturbation introduces coherent whole-field shifts that survive spatial averaging. Pointwise scores are invariant to this; users of area-averaged / regional products would see the difference."
2. **New diagnostic to add**: spatial-mean SSR per model per lead, in addition to pointwise SSR. ~30 lines of script on existing intercomp data.
3. **Reinforces Phase 5 future-work direction**: see [[phase5-uncertainty-decomposition]] / [[hybrid-ic-weight-plan]] -- adding real IFS-ENS perturbed IC contributes spatially-decorrelated aleatoric variance, lets us back off sigma_weight, fixes the coherent-shift issue at the cost of needing real IC data.

## Related: zero-tune hybrid IC plan

If we add IFS-ENS perturbed IC (Phase 5), the variance decomposition gives a near-free sigma adjustment per model: see [[hybrid-ic-weight-plan]].
