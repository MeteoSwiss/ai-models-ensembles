---
name: model-fgn
description: "Google DeepMind FGN (WeatherNext 2 backbone) - architecture, functional noise injection, fair-CRPS training, and relation to user's Phase 6 fresh-per-step weight noise"
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# FGN (Functional Generative Network) - Google DeepMind, WeatherNext 2 backbone

Andrychowicz et al., "Skillful joint probabilistic weather forecasting from marginals", arXiv:2506.10772 (2026). Not in earth2studio. Production access only via Google Vertex AI + Earth Engine.

## Architecture
- GNN encoder/decoder between 0.25 deg lat/lon grid and a 6-times-refined icosahedral mesh. Graph-transformer processor on the mesh.
- 180 M params per seed, latent dim 768, 24 transformer layers, 6 heads.
- 6 h timestep, Markov order n=2 (conditions on t-2, t-1). 15 day horizon.

## Noise injection (the "functional" bit)
- Per-member 32-dim Gaussian noise vector z ~ N(0, I_32) is fed into every conditional layer-norm in the network (FiLM-style gating on per-channel scale/shift).
- Effective parameter perturbation theta(z) - theta* has rank <= ~32 across the whole network. Low-rank in parameter space, not in physical space.
- z is RESAMPLED EVERY 6 h STEP, not fixed across the rollout. So per-member trajectory is a random walk in function-space; spread grows as ~sqrt(T_steps).
- Layer-norm gates multiply whole feature maps by scalars, so perturbations are coherent rescalings of learned features rather than grid-point noise. This is what keeps members "physically plausible weathers" rather than "noisy versions of one weather".

## Uncertainty split (Section 2.2 of paper)
- Epistemic = 4 independently trained seeds (M_1..M_4), parameter uncertainty from finite training data + SGD stochasticity.
- Aleatoric = "learned functional perturbations" theta = theta* + Delta * epsilon with epsilon parametrised by z. Delta is learned jointly with theta*.
- Production WeatherNext 2: 4 seeds x 14 members = 56 ensemble members total.

## Loss
- Fair CRPS (alpha = 1, full bias correction), N=2 samples per location per gradient step. See [[reference-scoring-rule-jargon]] for fair/afCRPS definitions.
- Theoretical claim: minimising marginal fair-CRPS on the noise-injection-constrained network produces a calibrated JOINT distribution over trajectories. That is the paper's main contribution.

## Training schedule
- Stage 1: 400 k steps, 1 deg ERA5, dt=12 h.
- Stage 2: 100 k steps, 1 deg ERA5, dt=6 h.
- Stage 3: 32 k steps, 0.25 deg ERA5, dt=6 h.
- Stage 4: 18 k AR-finetune steps, 0.25 deg HRES-fc0, 1-8 rollout steps.
- ~3 wall-clock days per seed on ~490 TPUv5p+v6e-days.

## Inference
- ~1 minute per 15 day trajectory on one TPU v5p, members embarrassingly parallel.

## Relevance to this project
- FGN's aleatoric mechanism is what [[phase6-fresh-per-step-weight]] reaches for post-hoc: temporal random walk in parameter space, sqrt(T) spread growth, no spatial-mean coherence problem.
- Differences vs Phase 6: FGN learns Delta jointly with theta*; Phase 6 fixes sigma post-hoc. FGN routes noise via layer-norm gates (low rank, structured); Phase 6 uses dense additive noise on weights (high rank, unstructured). Structured-vs-unstructured probably explains part of the coherence gap in [[spatial-mean-vs-pointwise-ssr]].
- Cleanest one-liner contrast vs SFNO modes10: FGN constrains noise to low RANK in parameter space; SFNO modes10 constrains it to low WAVENUMBER in physical space. Different cuts along different axes; both forbid grid-point-independent noise.
