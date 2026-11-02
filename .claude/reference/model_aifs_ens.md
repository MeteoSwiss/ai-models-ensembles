---
name: model-aifs-ens
description: "ECMWF AIFS-ENS (AIFS-CRPS) - GNN enc/dec + sliding-window transformer with per-step Gaussian noise in processor, afCRPS loss. The ensemble variant of AIFS used as baseline in this project."
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# AIFS-ENS (ECMWF, operational ensemble)

Lang et al., "AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the Continuous Ranked Probability Score", arXiv:2412.15832 (2024). Operational at ECMWF since 2025. In earth2studio as `aifsens`.

Separate from [[model-aifs-deterministic]] - the architecture is the same family but the noise injection + loss makes it a fundamentally different probabilistic system.

## Architecture
- Encoder-processor-decoder. GNN encoder + GNN decoder between lat/lon grid and a hidden mesh.
- Processor: sliding-window transformer, 16 layers, embedding dim 1024, 8 heads, ~229 M params total.
- 6 h timestep.

## Noise injection
- Per-member Gaussian noise injected DIRECTLY INTO PROCESSOR ACTIVATIONS (additive on hidden state). Each member draws a fresh realisation.
- Fresh draw per member PER STEP. No rank constraint, no spectral constraint - the CRPS loss is what teaches the network to "shape" the raw Gaussian noise into structured uncertainty.
- Operational deploys 51 members, training uses small M (M=2 typical) with afCRPS bias correction.

## Loss
- Almost-fair CRPS (afCRPS) = interpolation between naive and full-fair CRPS with parameter alpha in (0, 1].
- Motivation: at start of training members are nearly identical, fair-CRPS gradient is dominated by spread term pushing members apart arbitrarily; afCRPS with alpha < 1 tames that without losing most of the fair correction.
- See [[reference-scoring-rule-jargon]] for the full proper/fair/afCRPS taxonomy.

## How it differs from [[model-fgn]]
- Same overall recipe: GNN enc/dec + transformer-on-mesh + CRPS-flavoured proper score + noise in network.
- Architecture: AIFS sliding-window transformer (16 layers, dim 1024) vs FGN graph transformer (24 layers, dim 768).
- Noise: AIFS = unconstrained Gaussian directly into activations, fresh per step; FGN = 32-dim global vector through conditional layer-norm gates, fresh per step. AIFS noise is HIGH-RANK and unstructured; FGN is LOW-RANK (rank ~32) and structured by layer-norm gating.
- Loss: AIFS uses afCRPS (alpha < 1), FGN uses fair CRPS (alpha = 1).
- Both end up calibrated; the noise-shaping vs noise-constraining is a training-engineering trade.

## In this project
- Used as `aifsens` baseline. See [[noise-injection-taxonomy]] for the cross-model table.
- IFS-ENS perturbed analyses are used as IC for Aurora/GraphCast/SFNO experiments ([[ifs-ens-ic-experiment]]), NOT for aifsens itself.
