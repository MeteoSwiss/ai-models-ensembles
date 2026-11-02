---
name: model-aifs-deterministic
description: "Architecture, training, and operational details for ECMWF AIFS v1 deterministic (Lang et al. 2024). The perturb target whose probabilistic sibling is AIFS-CRPS / AIFS-ENS."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# AIFS v1 deterministic (ECMWF, Lang et al. 2024)

## Architecture
- Encoder: graph-attention GNN mapping ERA5 N320 reduced Gaussian grid (~31 km, 542,080 points) onto a hidden mesh.
- Processor: pre-norm transformer with **shifted-window attention along latitude bands**, **16 layers**, operating on the **O96 octahedral reduced Gaussian grid (40,320 nodes, ~1 degree)**. Embedding dim, head count, window size are not stated in the v1 paper (the often-cited "1024" is from AIFS-CRPS; verify separately for v1 if needed).
- Decoder: graph-attention GNN N320 -> output.
- Eight learnable features added per node (input mesh, processor mesh) and per encoder/decoder edge.
- Anemoi framework (`anemoi-models`, `AnemoiModelInterface`). Our verified state-dict: **250 keys, ~255M params**, broken down as model.encoder (30 tensors / 19.9M), model.node_attributes (4 / 7M), model.processor (176 / 201M), model.decoder (32 / 27M), 8 pre/post-processor normaliser stats (SKIP for perturbation).

## Training data and resolution
- ERA5 pre-training **1979-2022** (v1 model card; arXiv v1 said 1979-2020); 13 pressure levels 50-1000 hPa; upper-air T, U, V, Q, Z plus surface (2t, 2d, 10u, 10v, msl, tp), soil, accumulated radiation.
- Native **N320 (~31 km / 0.25 deg)** input/output; **O96 processor mesh**.
- 6 h step: inputs at t-6h and t0, target at t+6h.
- Fine-tuning on operational IFS analyses **2016-2022**.

## Training objective (deterministic AIFS = "single-mse")
- **Area-weighted MSE** with per-variable scaling so prognostic variables contribute roughly equally; vertical velocity down-weighted; **loss weights decrease linearly with height**.
- Pre-train: 260,000 steps, cosine LR 0 -> 1e-4 -> 3e-7.
- **Rollout fine-tune** to **72 h (12 AR steps)**, rollout length grows every 1000 steps; LR 6e-7 on ERA5 then ~8e-7 -> 3e-7 on IFS analyses (~7,900 steps, 12 epochs). Quote: "the deterministic AIFS is trained to produce forecasts that minimise mean squared error (MSE) up to 72 h into the forecast."

## Operational variant we use
- `hf://ecmwf/aifs-single-1.1`, file **`aifs-single-mse-1.1.ckpt`** (loaded by [earth2studio/models/px/aifs.py](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/aifs.py)). "single-mse" = single-resolution (vs ensemble variant) trained with **deterministic MSE** (vs CRPS). 96 output variables, 721x1440 lat/lon I/O.

## Key results / claims
- vs IFS HRES: >12 h forecast advantage at longer lead times on Z500 ACC; "substantially lower errors than IFS after day 1"; better TC track errors.
- vs GraphCast: "Difference between AIFS and GraphCast is much smaller than between AIFS and IFS."
- Inference: **~2 min 30 s for a 10-day forecast on one A100** (incl. I/O). Training: ~1 week on **64 A100s (40 GB)**.

## Relationship to AIFS-CRPS (probabilistic baseline)
Same anemoi-models framework, same encoder/processor/decoder topology and N320/O96 meshes. AIFS-CRPS adds conditional-LayerNorm noise injection in the processor and swaps MSE for an almost-fair CRPS loss; AIFS-ENS (`aifs-ens`) = the trained-stochastic variant we use as a baseline. Deterministic AIFS = identical architecture without noise injection and without CRPS objective.

## Strengths / limitations (relevant to our perturbation work)
- Matched-architecture comparison: weight-perturbed AIFS-det vs AIFS-ENS isolates "trained stochasticity + CRPS objective" as the only knob, since backbone and resolution are identical.
- Skipping the 8 pre/post normaliser stats is correct (they are running mean/std, not learnable inference weights).
- Layer-group sweep for Phase 2: model.encoder / model.processor / model.decoder map cleanly to the GNN-enc / sliding-window-transformer / GNN-dec functional split.

## Reference
Lang, S. T. K. et al. (2024), "AIFS - ECMWF's data-driven forecasting system", [arXiv:2406.01465](https://arxiv.org/abs/2406.01465). Model card: [huggingface.co/ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0).
