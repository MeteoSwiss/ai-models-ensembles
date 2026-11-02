---
name: model-sfno
description: "Architecture, training, and operational details for NVIDIA SFNO (Bonev et al. 2023). Key facts for paper writing and perturbation interpretation."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# SFNO (NVIDIA, Bonev et al. 2023)

## Architecture
- Backbone: stack of **spherical neural-operator blocks**, each = SHT -> learned mode-space conv (complex weights) -> inverse SHT, followed by a pointwise MLP and skip connection. Replaces FNO's planar DFT with the **Spherical Harmonic Transform (SHT)**, removing equator/pole artefacts that FNOs incur on lat-lon grids (paper: "DFTs cause visual and spectral artifacts as well as pronounced dissipation when learning operators in spherical coordinates since they incorrectly assume a flat geometry").
- Encoder (pointwise lift to embed dim) -> 8 SFNO processor blocks -> decoder (pointwise project to channels), plus a residual/skip pathway.
- The 73-channel ERA5 recipe identified in `makani` is `sfno_linear_73chq_sc3_layers8_edim384_asgl2`: **8 layers, embed_dim 384, scale_factor 3** (3x downsampling between encoder and processor), with `dhconv` (real-valued spherical conv) operator.
- Spectral truncation: each processor block keeps modes up to `hard_thresholding_fraction` of the SHT Nyquist (default ~1.0); for 721x1440 with scale_factor 3, processor grid 240x480 -> `modes_lat ~= 240` (matches our SFNO diagnostic 2328355).
- Parameter count: the 73-channel operational checkpoint is **~573M float weights** (verified from our state-dict dump).

## Training data and resolution
- ERA5, **0.25 deg** equirectangular grid (south-pole excluding) = **721 x 1440**, **6 h** step.
- Train years roughly 1979-2015, val 2016, test 2018 (Pathak-FCN convention reused by makani).
- **73 channels** (per [earth2studio sfno.py](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/sfno.py)): 8 surface (`u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv`) + {`u, v, z, t, q`} x 13 pressure levels {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000} hPa = 8 + 5*13 = 73.

## Training objective
- **Channel-normalised, latitude/area-weighted relative spherical L2 (p=2)**. Per-channel: divide by per-channel std; per-grid-cell: weight by cos(lat). Spherical "relative L2" = `||pred - true||_geom / ||true||_geom` integrated against the area measure on S2. makani exposes it as `geometric l2` with `channel_weights: auto`.
- **Two-stage training**: (1) single-step pretraining minimising the 6 h forecast loss; (2) autoregressive fine-tuning starting from `multistep_count=2` and increasing -- gradients flow through unrolled rollouts to stabilise long horizons.
- Deterministic: one output per input, no stochastic head.

## Operational variant we use
- 73-channel, 0.25 deg, 6 h step.
- State-dict: **87 ckpt tensors = 79 float32 + 8 complex64** spectral conv filters at indices `0, 10, 20, 30, 40, 50, 60, 70` (one per processor block).
- Our layer groups: **encoder (83:86)**, **processor (0:80)**, **decoder (80:83)**, **residual (86:87)**.
- `.npy` files (`global_means.npy`, `global_stds.npy`) are normalisation stats, not perturbed.

## Key results / claims
- Headline (paper Tab. 1 / Fig. 5): SFNO matches or beats FCN (FourCastNet) on ACC/RMSE for z500, t850, u10 at 1-7 day lead, and is competitive with IFS HRES at short lead, losing skill beyond ~5 d.
- **Stable autoregressive rollouts for 1 year (1,460 6-h steps)** without blowup, with physically plausible spectra -- the central claim ("stable auto-regressive rollouts for a year of simulated time, while retaining physically plausible dynamics").
- Inference: order of seconds per 6 h step on a single A100, ~ minutes for a 10 day forecast; much cheaper than IFS HRES.

## Strengths / limitations (relevant to our perturbation work)
- Coarse spherical modes (small `l`) carry planetary-scale flow (Rossby waves, planetary thermal structure). Perturbing only `W[..., :L_cut]` of the 8 complex64 operators (our `--coarse-mode-cut`) directly targets that large-scale subspace -- this is exactly why Phase 3 `modes10` (l<10, lambda >= ~4000 km) is the physically natural axis.
- The **8 complex64 spectral conv tensors are the operator weights**: perturbing them edits the learned Green's-function-like spectral kernel of the neural operator. Perturbing the surrounding float32 MLP and encoder/decoder weights instead edits the pointwise channel-mixing, not the spatial operator.
- Limitation for us: deterministic head means no native spread -- weight perturbation is one of the few post-hoc ways to recover ensemble spread without re-training.

## Reference
Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K. and Anandkumar, A. "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere", ICML 2023. https://arxiv.org/abs/2306.03838. Code: https://github.com/NVIDIA/torch-harmonics, https://github.com/NVIDIA/modulus-makani.
