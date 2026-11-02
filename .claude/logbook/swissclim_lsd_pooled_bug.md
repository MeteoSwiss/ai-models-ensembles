---
name: swissclim-lsd-pooled-bug
description: "RETRACTED 2026-05-28. I (Claude) incorrectly claimed SwissClim's LSD per-band silently used spectrum-of-mean. Actually it correctly computes mean-of-per-member-spectra. Kept here as a postmortem to remember the misread."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

**Status: RETRACTED.** This file documents a misread, not a real bug. Kept for postmortem.

## What I claimed (wrong)

On 2026-05-28 I asserted SwissClim's `_plot_energy_spectra` was buggy because it hardcoded `reduce_ensemble=True` at L640 in `plots/energy_spectra.py`, regardless of the configured `ensemble.energy_spectra` mode. I argued this silently produced "spectrum of ensemble mean" and degraded LSD per band.

## What's actually true

Reading `calculate_energy_spectra` (L45-146) carefully, including the docstring:

> "If `average_dims` is provided we now (v2) compute the power spectrum FIRST for each member along those dimensions and *then* average the (latitude-weighted) power spectra. This implements the mean(spectrum(x)) instead of spectrum(mean(x))."

So the actual order inside `calculate_energy_spectra(da_var, average_dims=["ensemble"])`:
1. FFT per-member (apply_ufunc is vectorised over the ensemble dim)
2. `|FFT|^2` per-member -> power spectrum per-member
3. Latitude-weighted mean per-member
4. **THEN** mean across ensemble of those per-member power spectra (L142-145)

That is correct "pooled" behaviour: `mean(spectrum(member))`, not `spectrum(mean(field))`. The metric in the 7-way intercomp **is** valid.

## Lesson learned (feedback for me)

The L640 hardcoded `reduce_ensemble=True` only sets `average_dims=["ensemble"]` on the spectrum function. The semantic of "what average means" is decided INSIDE the spectrum function (post-FFT, on power spectra). My mistake was inferring semantics from the calling convention alone, not from the callee's implementation.

How to avoid this next time: when claiming a function-level bug, read the implementation, not just the call site. Especially for functions whose names suggest one ordering ("reduce_ensemble") but whose implementation does the other (post-FFT spectra averaging).

## Interpretation of the existing LSD per-band numbers

They reflect *real* pooled per-member-then-mean spectra. So the observation that "trained probabilistic baselines like fcn3 win LSD per band at lower mesoscale" is a genuine reflection of their training-in stochasticity producing better small-scale spectral content -- not a metric artifact. This is paper-worthy as it is.

## No rerun needed

I previously suggested re-running energy_spectra for all ablations and the baseline to "fix" this. That's unnecessary. Saved ~20-30 GPU-hours by catching my own error before submitting.
