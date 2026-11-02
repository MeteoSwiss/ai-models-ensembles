---
name: model-fcn3
description: "NVIDIA FourCastNet 3 - spherical neural operator + per-step spherical-diffusion conditioning noise + spectral-CRPS loss. Baseline `fcn3` in this project."
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# FCN3 (FourCastNet 3, NVIDIA / UC Berkeley)

Bonev et al., "FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale", arXiv:2507.12144 (2025). In earth2studio as `fcn3`. Used as `fcn3` baseline in this project.

## Architecture
- Spherical neural operator architecture, heavy use of local and global spherical group convolutions.
- Global convolution filters parameterised in the SPECTRAL DOMAIN via the convolution theorem on the sphere + spherical harmonic transform (SHT). Same lineage as [[model-sfno]].
- Hidden Markov: forecast at step t conditioned on state at step t-1 plus an exogenous noise input z_n.

## Noise injection
- Extra CONDITIONING input z_n is fed alongside the atmospheric state at every 6 h step.
- z_n is sampled from a small NUMBER OF SPHERICAL DIFFUSION PROCESSES with different length-scales AND time-scales. Fed in spectral (SHT) form so the noise is band-controlled by construction.
- Fresh draw per member per step. Unlike AIFS-ENS unconstrained noise, FCN3 noise has a STRONG STRUCTURED PRIOR baked in: band-limited spherical diffusion fields, not raw Gaussian.
- This is conceptually closest to a Stochastic Pattern Generator from NWP (cf. IFS-ENS SPP AR(1) random fields), but driven by ML loss instead of physical tuning.

## Loss
- Composite probabilistic loss based on CRPS computed in the SPECTRAL DOMAIN (per spherical-harmonic mode), aggregated. Per [[reference-scoring-rule-jargon]], CRPS is strictly proper for univariate marginals; the spectral aggregation is what makes the multivariate version work.
- End-to-end ensemble training. Small training M; one-step generation per member at deploy.

## Training scale
- Up to 1024 GPUs. Simultaneous domain + batch + ensemble parallelism.

## Inference
- 15 day forecast in 60 s on a single H100. ~8x faster than GenCast, ~60x faster than IFS-ENS.

## How it sits in the project taxonomy
- See [[noise-injection-taxonomy]] for the cross-model where-does-noise-enter table.
- Closest cousins: AIFS-ENS (per-step noise + CRPS), FGN (per-step structured noise + fair-CRPS). FCN3 is the most "physics-prior-informed" of the three - the spherical-diffusion prior on noise is doing what AIFS-ENS leaves to the network and what FGN does via low-rank layer-norm gating.
- The SFNO backbone is shared with [[model-sfno]] in this project; the difference is FCN3 trains end-to-end as a probabilistic ensemble model, the project's `sfno_modes10` adds post-hoc spectral weight perturbation to a deterministic SFNO.
