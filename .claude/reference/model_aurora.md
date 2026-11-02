# Aurora (Microsoft, Bodnar et al. 2025)

## Architecture
- Backbone: 3D Swin Transformer U-Net, 48 layers total, symmetric down/up, 3 stages. Hybrid with Perceiver encoder/decoder.
- Encoder: 3D Perceiver, projects heterogeneous inputs (surface + multi-level atmos) into a fixed latent stack of `latent_levels=4`.
- Processor: Swin Transformer U-Net with local-window self-attention.
- Decoder: 3D Perceiver decoder, reconstructs per-variable / per-level outputs.
- Default config from [microsoft/aurora `aurora.py`](https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py): `embed_dim=512`, `encoder_depths=(6,10,8)`, `decoder_depths=(8,10,6)`, `num_heads=16`, `window_size=(2,6,12)`, `patch_size=4`.
- Total params: ~1.3B (paper headline).
- Variables: surface `(2t, 10u, 10v, msl)`; atmos `(z, u, v, t, q)` x 13 levels `[1000,925,850,700,600,500,400,300,250,200,150,100,50] hPa`.
- 644 float32 tensors in the state dict (verified from our checkpoint dump, see [[checkpoint-perturbation-audit]]): backbone 0:594, decoder 594:611, encoder 611:644.

## Training data and resolution
- Pretraining: ~1M+ hours of Earth-system data, mixed sources -- ERA5, MERRA-2, GFS analyses + forecasts, IFS HRES, CAMS, CMIP6 climate sims. Primary atmos pretraining at 0.25 deg.
- Fine-tuning variants: 0.25 deg weather, 0.1 deg HRES-trained weather, CAMS air-quality (12h step + log transform), HRES-WAM ocean waves.
- Eval: 2022-2023 holdout for weather; per-task holdouts for waves/air quality/cyclones.

## Training objective
- Pretraining: weighted MAE (paper Eq. 1) summed over surface vars (weights `w^S_k`, factor alpha) and atmos vars per level (weights `w^A_{k,c}`, factor beta), normaliser gamma over variable counts. Target = next 6h state. No explicit latitude weighting in the published equation.
- Fine-tuning: two-phase -- (1) short-lead 1-step error, (2) rollout fine-tuning with multi-step unrolled loss. LoRA used on operational/HRES variants (paper Sec B.6-B.9); pretrained variant has no LoRA.
- Deterministic. No native ensemble.

## Operational variant we use
- earth2studio default: [`aurora.py`](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/aurora.py) loads `hf://microsoft/aurora@refs%2Fpr%2F1` -> `aurora-0.25-pretrained.ckpt`. Comment in code: "The pretrained version does not use LoRA."
- 0.25 deg (720x1440), 6h step, autoregressive rollout.
- This is the **pretrained** checkpoint (not the operationally fine-tuned variant). Matches our 644-tensor dump.

## Key results / claims
- Beats IFS HRES on >91% of 0.25 deg targets, and 92% of 0.1 deg targets with up to 24% RMSE reduction past 12h lead.
- Beats GraphCast on >91% of 0.25 deg targets.
- Air-quality variant ~50,000x faster than CAMS, ~1.1s per hour-of-lead on a single A100.
- Cyclone tracks: outperforms all official agencies on 100% of tracks tested.

## Strengths / limitations for our perturbation work
- Encoder is the Perceiver projection from 13 physical levels to 4 latent levels -- a low-dim bottleneck. Weight noise here perturbs the level-mixing kernels and propagates broadly. Matches our Phase 2b finding that encoder is the best target for Aurora ([[calibration-winners]]).
- Processor is the bulk of the params (backbone 594/644 tensors). Swin local windows mean noise injected here propagates within a window first, then mixes via stage transitions -- diffuses spatially with depth.
- Decoder is small (17 tensors), reconstructs per-variable outputs; perturbing it should produce variable-specific biases rather than dynamical perturbations.
- Architectural quirk: Perceiver level reduction (C -> 4) means the encoder mixes pressure levels nonlocally before any spatial attention, so encoder-weight noise creates vertically-correlated perturbations -- physically plausible direction for ensemble generation.

## Swin U-Net indexing (verified 2026-05-31 against source)

The 0.25 deg `Aurora` class instantiates `Swin3DTransformerBackbone` with `encoder_depths=(6, 10, 8)` and `decoder_depths=(8, 10, 6)` (see [`aurora/model/aurora.py:62`](https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py#L62)). That gives **3 encoder layers and 3 decoder layers**, indexed **top-down**:

- `backbone.encoder_layers.0` = shallowest, embed_dim 512, ~0.25 deg tokens. PatchMerging3D after.
- `backbone.encoder_layers.1` = middle, embed_dim 1024, ~2 deg tokens. PatchMerging3D after.
- `backbone.encoder_layers.2` = **deepest = U-net bottleneck**, embed_dim 2048, ~4 deg tokens. **No** downsample after it.

Per [`aurora/model/swin3d.py:820-833`](https://github.com/microsoft/aurora/blob/main/aurora/model/swin3d.py#L820-L833): `dim = int(embed_dim * 2**i_layer)`; `downsample=(PatchMerging3D if (i_layer < N - 1) else None)`. Same convention for `AuroraHighRes` (0.1 deg) and `AuroraSmall`.

**Implication for our Phase 3 hook**: `unet_bottom` = `net.backbone.encoder_layers.2.*` (96 tensors) is correctly the deepest 2048-ch / 4 deg block, NOT the second-from-top. Our `enc_12` and `enc_012` named groups in [`e2s_perturbation.py:160-166`](ai_models_ensembles/e2s_perturbation.py#L160-L166) include layers 1+2 and 0+1+2 respectively, broadening the perturbation toward finer scales (Phase 3b "spatial-reach threshold sweep"). All three Phase 3 hooks across Aurora/GraphCast/SFNO land in the architectural middle stack (backbone bottleneck / m2m processor / spectral conv blocks), not in the input encoder or output decoder.

## Source
Bodnar et al., "A Foundation Model for the Earth System", Nature 641, 2025; arXiv:2405.13063v3. <https://arxiv.org/abs/2405.13063> / <https://github.com/microsoft/aurora>
