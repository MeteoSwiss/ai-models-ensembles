---
name: noise-injection-taxonomy
description: "Cross-model table of where noise enters in every ensemble baseline used in this project (IFS-ENS, AIFS-ENS, FCN3, Atlas) plus the 3 post-hoc perturbation baselines."
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# Where noise enters in each ensemble baseline

For the 7-way intercomp (`ifs_ens`, `aifsens`, `fcn3`, `atlas` + project's own `sfno_modes10`, `aurora_encoder`, `graphcast_all`).

## Native ensemble systems (4)

| Model | Where | What | When | Loss |
|---|---|---|---|---|
| `ifs_ens` ([[model-ifs-ens]]) | (1) IC; (2) physical tendencies | (1) EDA + first 50 leading 48 h singular vectors of tangent linear, scaled to operational analysis spread; (2) SPP - 27 physics parameters x AR(1) random fields at 1000 km / 3000 km horizontal and 3 d / 30 d temporal decorrelation | (1) once at t=0 per member; (2) refreshed every model step | none (NWP, primitive-equation solver) |
| `aifsens` ([[model-aifs-ens]]) | Inside processor stack | Per-member Gaussian noise injected directly into processor activations (additive on sliding-window transformer hidden state). High-rank, unstructured | Fresh draw per member per 6 h step | afCRPS (alpha < 1) |
| `fcn3` ([[model-fcn3]]) | Extra conditioning input z_n fed alongside state | Samples from spherical diffusion processes with multiple horizontal length-scales and temporal correlation scales; fed in spectral (SHT) form -> band-controlled by construction | Fresh draw per member per 6 h step | composite spectral CRPS |
| `atlas` ([[model-atlas]]) | Inside latent DiT, every reverse-time step | Standard Gaussian driving the SDE that maps z_T ~ N(0, I) to z_0 = latent forecast. 60 reverse steps per member, each = denoising score + Brownian increment | Fresh draw per member PER REVERSE STEP (~60 draws per forecast hour) | stochastic-interpolant / score-matching in latent space |

## Project's post-hoc perturbation baselines (3)

| Model | Where | What | When | Loss |
|---|---|---|---|---|
| `sfno_modes10` | Parameter space, SFNO spectral conv filters W[..., :10] (lowest 10 spherical-harmonic modes) | Additive CN(0, sigma=0.25) on low-l weight slice; sigma chosen by argmin-CRPS sweep | Once at t=0, FROZEN across rollout (Phase 1-3); Phase 6 variant resamples per step | (none at train time; deterministic SFNO + post-hoc perturb) |
| `aurora_encoder` | Parameter space, Aurora encoder layers (encoder_layers.0/1/2, 33 tensors) | Additive N(0, sigma=0.025) on all encoder float32 weights | Once at t=0, frozen | none |
| `graphcast_all` | Parameter space, all 264 GraphCast params (g2m + m2m + m2g) | Additive N(0, sigma=0.01) on every NPZ float key | Once at t=0, frozen | none |

## Reading the table

- **IC noise**: only IFS-ENS does it natively. The other 3 native ensembles all start from a single deterministic analysis and generate spread purely from internal model stochasticity. That is why [[ifs-ens-ic-experiment]] cleanly isolates the "IC spread" question for Aurora/GraphCast/SFNO.
- **Per-step vs once-at-t=0**: all 4 native ensembles refresh noise within the rollout (in different places); the project's perturbation baselines freeze noise at t=0 (Phase 6 is the exception that re-samples per step).
- **Structured vs unstructured**: FCN3 (spectral spherical-diffusion prior) and FGN (low-rank layer-norm gating, see [[model-fgn]]) both impose strong structural priors on the noise. AIFS-ENS imposes none and relies on CRPS loss to shape it. Atlas imposes none, the structure comes from the diffusion solver schedule.
- **Rank vs wavenumber cuts**: project's `sfno_modes10` is a low-WAVENUMBER cut (physical-space spectral truncation). FGN is a low-RANK cut (parameter-space). Different axes. Both forbid grid-point-independent noise.

## Implications for paper framing

Per [[paper-story-refinement-2026-06-01]], the project's value proposition is "zero marginal training cost" - not "lower per-member inference cost". The noise-injection table makes the cost story clear:
- Native ensembles bake noise into training and (for FCN3/FGN/AIFS-ENS) pay the modelling cost during training.
- Project's perturbation baselines pay zero training cost; noise is applied to weights of an already-trained deterministic model.
- The honest CRPS gap ([[headline-7way-crpss]]) is the price of that zero training cost.

See also: [[spatial-mean-vs-pointwise-ssr]] for the coherence-of-perturbations finding, [[phase6-fresh-per-step-weight]] for the FGN-analogue post-hoc fix, [[reference-scoring-rule-jargon]] for the loss-function vocabulary.
