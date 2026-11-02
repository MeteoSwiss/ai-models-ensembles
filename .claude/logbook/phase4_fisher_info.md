---
name: phase4-fisher-info
description: Phase 4 ablation plan (NOT STARTED) — perturb weights weighted by inverse Fisher information; targets directions of low loss curvature for principled spread-without-skill-loss
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Phase 4: Fisher-information-targeted perturbation (NOT STARTED, deferred)

## Goal

Perturb weights along directions where the loss landscape is *flat* (low
curvature), measured by diagonal Fisher information:

    F_i = (1/N) * sum_n (dL_n/dw_i)^2

Hypothesis: low-Fisher weights are directions where the model is least
sensitive -> perturbing them produces ensemble spread without degrading
each member's skill. This is the SWAG / flat-minima ensemble approach
(Maddox et al. 2019, Wilson & Izmailov 2020) applied to weight
perturbation.

**Why:** [[phase1-signal-vs-noise]] shows uniform multiplicative noise
degrades signal (extremes shrink) while injecting small-scale noise.
Fisher-low targeting should produce diverse forecasts without that
degradation, because flat directions don't affect the loss.

**How to apply:** compute Fisher info once per model, save to disk, use
as a per-weight sigma multiplier in `_perturb_array`. The bottom-K%
Fisher info weights get all the perturbation budget.

## What's required from the user

1. Validation set: ~100-500 ERA5 ICs + truth at +6h (held-out month).
   Already have most of this from existing inference pipeline.

2. Loss function: latitude-weighted MSE on key vars (T, U, V, Z at
   500/850 hPa). GraphCast paper eq. (A4) gives the formula.

3. Gradient-enabled model loading -- the hard part. earth2studio wraps
   models for inference; gradient flow may need to bypass it.
   - SFNO, Aurora: PyTorch, straightforward (`loss.backward()` plus
     `param.grad**2` accumulator).
   - GraphCast: JAX, needs `jax.grad`. May need to import deepmind's
     graphcast package directly, skip earth2studio wrapper.

4. Compute budget (one-time):
   - Aurora: ~3-5 NH
   - SFNO: ~3-5 NH
   - GraphCast: ~5-10 NH (JAX backprop through 264 message-passing ops)
   - Total: ~15-20 NH for Fisher computation. Perturbation itself is free.

5. Storage: Fisher info matches weight shape -> ~5 GB Aurora + ~1 GB
   each for SFNO/GraphCast. Total ~7 GB on $STORE.

6. Code work: ~2-3 days.
   - New `ai_models_ensembles/fisher_info.py` (compute + save).
   - Per-model sbatch script (need `normal` partition, debug too short).
   - `_perturb_array` extension to accept per-tensor sigma multiplier
     from a Fisher-info file.
   - Per-layer ranking (perturb bottom-K% within each layer), not
     global ranking -- Fisher scales vary wildly per layer, naive
     global ranking is dominated by one or two layers.

## Pitfalls

- Backward through autoregressive rollouts is memory-heavy. Stick to
  1-step (+6h) backprop.
- earth2studio's inference wrapper may not expose gradient flow. Likely
  need to bypass it and load model classes directly.
- GraphCast JAX is the tallest barrier; bypassing earth2studio there is
  the main code investment.
- Loss choice matters more than commonly thought. Latitude-weighted RMSE
  on key vars is safe; using the model's actual training loss is ideal
  but each model has its own implementation to track down.

## Decision (2026-05-18)

**Defer indefinitely.** [[phase3-physical-perturbation]] already gives
a clean physically-motivated story for the current paper:
- Phase 1: uniform sigma (control)
- Phase 2: layer-group targeting (architectural prior)
- Phase 3: physical-scale targeting (physical prior)

Adding Phase 4 widens scope ~20 NH compute + 2-3 days code + new failure
modes (especially GraphCast/JAX) for incremental contribution. Better
saved for a follow-up paper specifically on "what does the loss
landscape say about which weights to perturb in AI weather models".

If revisited: lightest first step is **one model only** (SFNO, easiest
in PyTorch) as a proof of concept to check if per-tensor Fisher info
ranking aligns with anything interpretable.
