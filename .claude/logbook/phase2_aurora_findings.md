---
name: phase2-aurora-findings
description: Phase 2 Aurora results (2026-05-19) — sqrt(N) scaling validated; encoder perturbation is best probabilistic but slightly overdispersive; backbone gives near-zero spread.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Phase 2 Aurora: results and decision

Phase 2 ran sigma_full=0.01 with sqrt(N_total/N_group) variance scaling on
three named groups (encoder, backbone, decoder). Per-group sigma:
encoder 0.044, backbone 0.010, decoder 0.062. All vs Phase 1
mag_0.01_layer_all reference.

## Key findings
- **sqrt(N) scaling works.** Spread magnitudes across the three group runs
  are comparable to Phase 1 layer=all at sigma=0.01, confirming the
  variance-budget rule.
- **Encoder = best probabilistic.** CRPS, SSR, PIT, spaghetti all favor
  encoder-only perturbation over backbone/decoder. Encoder gives the most
  forecast-skill-relevant spread.
- **Backbone alone = almost no spread.** Despite being 594/644 of the
  parameters, perturbing only the backbone with the scaling-required sigma
  produces a nearly-deterministic ensemble. Suggests backbone weights are
  collectively very robust to small perturbations -- spread is generated
  upstream/downstream.
- **Backbone wins on FSS + bivariate physics + spectrogram.** The three
  exceptions where backbone > encoder: FSS (categorical fields), bivariate
  histogram with physics constraints (Clausius-Clapeyron / geostrophic),
  energy spectra delta. These are tied to physical consistency, which the
  small-spread backbone runs preserve.

## Decision for Aurora
- ~~Pick encoder perturbation as the Aurora ensemble recipe~~ **REVISED
  2026-05-21**: the paper's Aurora recipe is the Phase 3 backbone-side
  *coarse-scale* mechanism (encoder_layers.2), NOT the Phase 2 encoder
  input-projection group. These target different architectural pieces and
  produce different behavior:
  - Phase 2 "encoder" = input projection (33 tensors, perturbs how IC is
    read in). Best of Phase 2 but not a physically-motivated coarse-scale
    target.
  - Phase 3 "unet_bottom" = backbone encoder_layers.2 (96 tensors,
    perturbs deepest Swin block at 4 deg tokens / ~5000 km attn window).
    Physically motivated but inherits backbone robustness.
- **Phase 2b (tune encoder sigma down) DROPPED**. Phase 2 encoder was
  overdispersive (SSR > 1 at sigma=0.044) but we're not shipping Phase 2
  encoder as the recipe -- Phase 3 backbone is the story. Tuning Phase 2
  sigma doesn't inform Phase 3 behavior.

**Why:** the encoder is the read-in layer mapping ERA5 channels to model
tokens. Perturbing it perturbs the model's *interpretation of the IC*,
which is the most physically defensible weight-perturbation target (close
in spirit to IC perturbation but operating in feature space). Backbone
weights, being deeply averaged across many tokens and layers, may
self-correct any small perturbation -- the lack of spread is consistent
with that picture.

**How to apply:** in any future Aurora-ensemble discussion, the default
is encoder-group perturbation; backbone-only is a useful comparison case
but not the production recipe. If the user mentions tuning amplitude,
the expected direction is *down* from sigma=0.044.

## Cross-model implication
- Worth checking the same pattern in SFNO + GraphCast Phase 2 once they
  run: is the "input-mapping" group (encoder for SFNO, g2m for GraphCast)
  also the best probabilistic? If yes, that's a paper-worthy
  cross-architecture finding.

## Knock-on effect: Phase 3 Aurora target narrowed
- Original Phase 3 Aurora plan was "bottleneck" = `encoder_layers.2` +
  `decoder_layers.0`. After this Phase 2 finding (encoder >> decoder for
  spread), Phase 3 Aurora target is now **`encoder_layers.2` only**
  -- the coarse-resolution *encoder* block, not the full U-Net
  bottleneck. See [[phase3-physical-perturbation]] for the updated plan.
- Logic: doubling down on the side that produces spread, instead of
  mixing in a dec_0 contribution that would dilute the signal.

## Status
- Phase 2 Aurora intercomp (job 2304464) done 2026-05-19.
- Output at $STORE/ablation/phase2/aurora/intercomparison_vs_phase1/.
- Phase 2 SFNO + GraphCast launching 2026-05-19 evening.
