# GraphCast (DeepMind, Lam et al. 2023)

## Architecture
- **Encoder-Processor-Decoder GNN** with three bipartite/mesh graphs:
  - **g2m** (grid-to-mesh): input lat-lon nodes -> finest mesh nodes. 1 message-passing step.
  - **m2m** (mesh-to-mesh): multi-mesh processor, 16 GNN message-passing steps with shared topology, NOT shared weights (16 distinct param blocks).
  - **m2g** (mesh-to-grid): mesh nodes -> output lat-lon nodes. 1 message-passing step.
- **Icosahedral multi-mesh**: 7 refinement levels M0..M6. M6 = finest = 40,962 nodes, 327,660 directed edges (paper Fig 1 + Table E1). The "multi-mesh" used by the processor is the union of edges from ALL levels M0..M6 (so coarse edges coexist with fine), enabling long-range info flow in a single MP step.
- **Latent dim 512** for all node/edge MLPs, 1 hidden layer, LayerNorm + swish.
- **Parameter count**: 36.7M total (full 0.25 deg model).
- **Time stepping**: single 6h step, autoregressive. 10-day forecast = 40 rollout steps. Input is 2 consecutive states (t-6h, t) -> predicts residual to t+6h.

## Training data and resolution
- **ERA5** reanalysis, 1979-2017 train / 2018 val / 2019-2021+ test.
- **Resolution**: 0.25 deg lat-lon = **721 x 1440** grid (~1M nodes).
- **Variables**: 5 surface (2t, 10u, 10v, msl, total_precip_6h) + 6 atmospheric (z, q, t, u, v, w) on **37 pressure levels** = 5 + 6x37 = **227 input/output channels** per timestep (plus static + forcing channels).

## Training objective
Latitude-weighted MSE, summed over variables/levels/lead times:

```
L = (1/|D|) sum_{t in D} sum_{j in nodes} sum_{i in vars} s_i  w_j  (x_hat - x_true)^2
```

- `s_i` = per-variable weight; atmospheric vars weighted **linearly by pressure level** (heavier weight to higher-pressure / lower-altitude levels), normalised so sum over levels = 1.
- Per-variable scalar weights: surface vars 2t=1.0, msl=0.1, 10u=0.1, 10v=0.1, tp=0.1; atmospheric z/q/t/u/v/w each weight 1.0 distributed across levels.
- `w_j` = `cos(lat_j) / mean(cos(lat))` area weight.
- Target is per-variable-normalised residual (divided by stddev of 6h differences from training stats).
- **Curriculum**: phase 1 = 1 AR step (1000 epochs ERA5), phase 2 = ramp from 2 to **12 AR steps** (one extra step per 1000 examples), phase 3 = fine-tune 12 steps on most recent data. Total ~4 weeks on 32 TPU v4.
- Deterministic: single output per input (no probabilistic head).

## Operational variant we use
- **GraphCast_operational** = the one we perturb via earth2studio's `GraphCastOperational`.
- 0.25 deg, **13 pressure levels** (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa) -- the WeatherBench-13 set -- vs 37 levels for the science.org headline model.
- Pre-trained on ERA5 1979-2017, then fine-tuned on **HRES operational analyses 2016-2021**, so it accepts HRES ICs without the precipitation input.
- Architecture identical (M0..M6, 16 MP steps, latent 512). Param count smaller than 37-level model due to fewer channels.
- Verified from checkpoint (NPZ): **264 perturbable float arrays**, ordered g2m (0:36) | m2g (36:66) | m2m (66:264). m2m has 198 = 12 + 11x16 params (12 per MP layer x 16 layers + heads). See [[checkpoint-perturbation-audit]].

## Key results / claims
- "Significantly outperforms the most accurate operational deterministic systems on **90% of 1380 verification targets**" (vs ECMWF HRES, RMSE/ACC at 0.25 deg).
- 10-day global 0.25 deg forecast in **under 60 s on a single Cloud TPU v4** (paper abstract). A100 wall-time comparable order.
- Skill at severe weather: tropical cyclone tracks, atmospheric rivers, extreme heat.

## Strengths / limitations (for our perturbation work)
- **g2m has small param share but full input projection responsibility** -> perturbing g2m (36 of 264 arrays, ~14%) directly attacks the input embedding. Our Phase 2 g2m noise winner is consistent with this.
- **m2m dominates parameter budget** (198/264 = 75%) but is split across 16 sequential MP layers -- noise on one layer is averaged out by 15 downstream layers, so per-layer sigma needs to be larger for similar effect (or perturb several layers jointly).
- **Multi-mesh edges from coarse M0..M3 carry long-range info in 1 hop** -- this is why activation-perturbation on the first ~42 mesh nodes (M0+M1, 12+30 nodes) propagates globally in a single MP step. Basis for Phase 3 GraphCast coarse-scale perturbation.
- **Residual prediction + 12-step curriculum** means the trained model is heavily regularised against small-magnitude perturbations -- explains why GraphCast tolerates large sigma (0.01-0.1) without collapsing where Aurora collapses at sigma > 0.03.
- Encoder/decoder MLPs have NO weight sharing across nodes -> perturbation hits every node identically (multiplicative noise on the shared MLP weight), so spatial coherence of perturbation is high. Relates to spatial-mean coherence finding ([[spatial-mean-vs-pointwise-ssr]]).

## Citation
Lam, R. et al. "Learning skillful medium-range global weather forecasting." Science 382, 1416-1421 (2023). https://arxiv.org/abs/2212.12794 -- code https://github.com/google-deepmind/graphcast
