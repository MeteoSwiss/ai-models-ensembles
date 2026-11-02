---
name: model-weathermesh
description: WindBorne WeatherMesh-3 (paper) and WM-6 (blog only) - latent NATTEN encoder-processor-decoder + learned ensemble DA. Not in earth2studio.
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# WeatherMesh-3 / WeatherMesh-6 (WindBorne Systems)

WM-3 paper: arXiv:2503.22235 (2025). WM-6: blog https://windbornesystems.com/blog/introducing-wm-6 (June 2026), TechCrunch coverage 2026-06-01. No WM-6 paper. Closed weights. NOT in earth2studio.

## WM-3 backbone (still the WM-6 forecast engine)
- Encoder-processor-decoder in latent space.
- Encoder: conv + ResNet blocks, lifts 0.25 deg (720x1440) to a 2 deg latent grid, then 2 NATTEN blocks with window (5, 7, 7) over (depth, lat, lon), latent dim 1024.
- TWO processors, both 10 NATTEN blocks: a 6 h processor and a 1 h processor. "Latent rollout": stay in latent space, chain 6 h + 1 h processors until lead time, decode ONCE at end. This is how they get 14 days in 12 s on a single 4090.
- Decoder: mirror of encoder via deconv.
- Position encoding: rotary embeddings with custom circular padding for sphere wrap.
- Loss: MSE at target timestep dt, dt set by curriculum.
- Training: ERA5 1979-2022 (2020 held out), Distributed Shampoo, batch 1, fp16, 6x RTX 4090.
- "Matepoint": custom checkpointing offloading activations to CPU RAM, enables deep AR rollouts on consumer GPUs.
- Inputs: 8 surface + 5 atmospheric vars x 28 pressure levels. Outputs: 17 surface + 5 atmospheric.

## WM-6 additions (blog only, not in any paper)
- End-to-end learned data assimilation, RE-ARCHITECTED AS AN ENSEMBLE DA: emits N analyses in latent space at t=0 directly, then deterministic latent rollout per member. They call it "modelling the full ensemble in latent space". No equations published.
- 128 ensemble members (vs IFS-ENS 51).
- 0.25 deg global, 3 km regional (CONUS + Europe), hourly cycling, 72 h regional / 14 day global horizon.
- Observation operator now ingests 11 obs types: AMSR2, CrIS, IASI, GOES, MRMS radar, surface synops, regional high-res analyses, WindBorne balloon soundings (innovation-based encoding = obs-minus-background fed in directly).
- 24 new output variables (largest output catalogue of any AI weather model per blog).
- Loss for the ensemble component NOT disclosed. Plausibly CRPS-like proper score on latent ensemble + MSE on forecast head, but that is inference.

## Contrast with [[model-atlas]]
- Atlas: latent DiT + 60-step stochastic-interpolant sampler. Ensemble = different samples from generative latent distribution at every step. ~38 min per member.
- WM-6: deterministic latent rollout. Ensemble = different starting analyses from learned ensemble DA at t=0. Cheap per-member rollout.
- They share "latent space" but the stochastic mechanism is at opposite ends: Atlas = stochastic sampler at every step, WM-6 = stochastic DA at t=0 + deterministic rollout.
- Closer architectural cousin to WM-6 in the containerised set is actually [[model-aurora]] (deterministic encoder-processor-decoder, Swin not NATTEN).

## Citing in paper
If WM-6 needs a citation: cite WM-3 paper for the backbone and the blog for ensemble + obs claims. Flag that no training loss is published. Per [[feedback-verify-paper-claims]] do not hedge - just say "WM-6 details from product blog; not peer-reviewed".
