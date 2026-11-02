---
name: phase3-physical-perturbation
description: "Phase 3 ablation plan — physically-motivated coarse-scale-only perturbations targeting upper-synoptic-and-above scales (lambda >= ~3000 km, per SwissClim band defs) aligned across SFNO / Aurora / GraphCast. Visual: figures/phase3_schematic.svg"
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Phase 3: physically-motivated coarse-scale-only perturbations

## Goal

Perturb only the parts of each model that learned representations of the
**planetary / large-synoptic scale** of the atmospheric flow (Rossby waves,
blocking patterns, jet positioning). Hypothesis: this gives a more
forecast-skill-relevant ensemble spread than scale-agnostic perturbation,
because that's where weather predictability decays most rapidly.

**Why:** weak point of weight perturbation in [[phase1-magnitude]] and
[[phase2-layer-location]] is they are scale-agnostic. Targeting the slow
modes mirrors what singular-vector and SKEB-style methods do in NWP, and is
the most physically defensible perturbation a paper can claim.

**How to apply:** define a single shared physical scale threshold
(wavelength >= ~3000-4000 km), then translate it into the appropriate
parameter subset in each model's checkpoint or forward pass.

## Threshold alignment (verified against actual dumps + SwissClim bands)

**SwissClim band conventions** (from
`SwissClim_Evaluations/src/swissclim_evaluations/plots/energy_spectra.py`):
- planetary       5,000–20,000 km
- synoptic        1,000–5,000 km
- upper meso      250–1,000 km
- lower meso      10–250 km

All three Phase 3 mechanisms target **upper synoptic and above**
(lambda >= ~3000 km). The earlier "planetary" label for 4000 km was wrong:
4000 km is upper-synoptic per SwissClim.

**SFNO**: spherical-harmonic degree `l <= L_cut`. **VERIFIED 2026-05-20.**
- Earth radius R = 6371 km. lambda = 2*pi*R / l.
  l=10 -> 4000 km (upper synoptic boundary).
  l=20 -> 2000 km (mid synoptic).
  l=5 -> 8000 km (planetary).
- `L_cut = 10` (upper synoptic -> planetary).
- Spectral conv tensor (`*.filter.filter.weight`) shape (384, 384, 240)
  complex64, one per block, 8 blocks. Earth2studio's `unsqueeze(0)` on
  load -> runtime shape (1, 384, 384, 240).
- Diagnostic job 2328355 confirmed operator_type=dhconv, modes_lat=240,
  forward_transform.lmax=240. The 240 axis is the spherical harmonic
  degree l directly (NOT a flattened (l,m) layout). dhconv ("Driscoll-
  Healy spherical conv") stores only modes_lat; convolution shares
  weights across m via SHT convolution theorem.
- Low-l |w| (mean 0.176) > high-l |w| (0.135) -- low-modes-first ordering
  verified empirically.
- L_cut=10 -> perturb W[..., :10] only -> wavelengths >= 4000 km.

**GraphCast**: icosahedral mesh refinement levels. Mechanism revised
2026-05-20 from edge-embedding perturbation (original plan) to
**coarse mesh-node activation perturbation**, after architecture review.
- Mesh node indices are preserved across refinement levels by deepmind's
  `merge_meshes`: level-0 vertices are nodes [0:12], level-0+1 = [0:42],
  level-0+1+2 = [0:162], etc. Edge counts vary per level but **edges are
  NOT tagged by level** in the merged mesh, so per-level edge selection
  would require haversine distance computation. Perturbing coarse-mesh
  NODE activations is simpler, hooks at a clean architectural boundary
  (between `_run_grid2mesh_gnn` and `_run_mesh_gnn`), and is physically
  equivalent: targets the planetary-scale latent representation.
- Default N_coarse = 42 (levels 0+1) -> ~3300 km separation, matches the
  original "λ ≥ 3300 km" framing.
- Mechanism: subclass `graphcast.GraphCast`, override `_run_grid2mesh_gnn`
  to multiply `latent_mesh_nodes[:42]` by `(1 + sigma * N(0,1))`. Uses
  `hk.next_rng_key()` for fresh RNG per forward step. Per-member
  randomness via `model.prng_key = jax.random.PRNGKey(seed + member_id)`.
- Monkey-patches `graphcast.GraphCast` before earth2studio's loader runs.
- **Sigma scaling caveat**: sqrt(N) tensor-count rule from Phase 2 breaks
  down here. Mesh has 40,962 total nodes (mesh "2to6") -> 42/40,962 = 0.1%
  fraction -> strict sqrt(N) sigma = 0.01 * sqrt(975) = 0.312, far larger
  than Phase 1/2 magnitudes. Activation perturbation at sigma=0.312 may
  be destructive. **Phase 3 GraphCast runs as a 4-sigma sweep:
  {0.01, 0.03, 0.10, 0.312}** to find the empirical sweet spot.

**Aurora**: coarse-resolution ENCODER block only (`encoder_layers.2`).
*Decision 2026-05-19 (post Phase 2)*: drop `decoder_layers.0` from the
Phase 3 target. Phase 2 showed encoder perturbation >> decoder perturbation
for spread, so doubling down on the coarse encoder is cleaner than mixing
in a dec_0 contribution that would dilute the signal.
- enc_0: 512 channels, 1 deg (~112 km) tokens
- enc_1: 1024 channels, 2 deg (~224 km) tokens
- enc_2: 2048 channels, 4 deg (~448 km) tokens [TARGETED]
- Channels double / spatial halves per downsample. Input grid 0.25 deg
  with 4x4 patch embed -> tokens at 1 deg at enc_0.
- Per-token spatial extent at enc_2: ~450 km (upper meso / lower
  synoptic boundary).
- Swin attention window at enc_2 (~6 lat x ~12 lon tokens per Aurora
  config) -> receptive field ~2700 x 5400 km per attention computation.
- Effective scale = synoptic to planetary (0.5-5 Mm), driven by the
  attention-window reach rather than the token resolution.
- New named group in `_MODEL_LAYER_GROUPS["aurora"]`: `"unet_bottom"`
  = indices of `net.backbone.encoder_layers.2.*` tensors only.

Threshold alignment now precise. Document the **upper synoptic to planetary**
framing in the paper, not "planetary" alone.

## Implementation plan

### SFNO — sub-axis slice perturbation (DONE 2026-05-20)
- Added `_is_spectral_conv_tensor` + `_perturb_array_low_modes` in
  `e2s_perturbation.py`. `_perturb_torch` branches on `coarse_mode_cut`:
  when set, skip non-spectral tensors, perturb only `W[..., :mode_cut]`
  for matching ones.
- CLI flag: `--coarse-mode-cut N`.
- Submitted as `phase3 sfno` in submit_ablation.sh: sigma=0.049261 from
  DOF-based scaling sqrt(572.5M/23.6M), N_partial = 8*384*384*10 complex
  elements = 11.8M params (23.6M DOF including complex x2).

### Aurora — new named group, existing API (DONE 2026-05-19)
- Added `"unet_bottom": (494, 590)` to `_MODEL_LAYER_GROUPS["aurora"]`
  (96 tensors, encoder_layers.2 only, indices verified from aurora_keys.log).
- sigma=0.025900 from sqrt(644/96).
- Run via `--layer unet_bottom`.

### GraphCast — runtime hook (DONE 2026-05-20)
- Cannot use `materialise_perturbed_package` pipeline (weight perturbation
  bypassed: weights are shared across all edges/nodes).
- New module `ai_models_ensembles/graphcast_coarse_perturbation.py`:
  `install(sigma, n_coarse_nodes)` monkey-patches `graphcast.GraphCast`
  with a subclass that overrides `_run_grid2mesh_gnn` to perturb
  `latent_mesh_nodes[:n_coarse_nodes]` by `(1 + sigma * jax.random.normal)`
  using `hk.next_rng_key()` for the seed.
- CLI: `--graph-coarse-sigma X --graph-coarse-nodes N` (default N=42).
- Sequential mode forced (graph_coarse_sigma>0 -> use_shared_model=False).
- Per-member RNG: `model.prng_key = jax.random.PRNGKey(seed + member_id)`
  set in `_model_for_member` after model load.

## Experimental design (consistent with [[phase2-layer-location]])

- Same 4 init times as Phase 1/2: 2024-02-15, 2023-05-15, 2023-08-15,
  2024-11-15.
- 10 members, sigma_full = 0.01 (consolidated value from Phase 1).
- Variance scaling: sigma_partial = 0.01 * sqrt(N_full / N_partial) so that
  total perturbation energy matches full-weight perturbation. N_full is the
  *count of perturbed parameters*; for SFNO the partial set is the low-l
  slice, for Aurora it's the bottleneck-layer tensors, for GraphCast it's
  conceptual (activation-level, no parameter count).
- Compare against: Phase 1 mag_0 (deterministic baseline), Phase 1
  layer=all sigma=0.01 (scale-agnostic perturbation), Phase 2 best
  layer-group per model.

## Cost estimate

- 3 models x 1 perturbation type x 4 inits x 10 members = 12 inference runs.
- ~20 NH inference + ~30 NH eval + ~3 NH intercompare = **~55 NH total**.
- Same scale as Phase 2 ([[phase2-layer-location]]).

## Visual

`figures/phase3_schematic.svg` (+ .pdf, .png, regen script
`figures/draw_phase3_schematic.py`) shows the three mechanisms side by
side: SFNO l-mode bar chart, Aurora U-Net with verified per-block dims,
GraphCast 6-globe icosahedral refinement series (levels 0-5).

## Status (2026-05-20)

- Aurora Phase 3: SUBMITTED 2026-05-19, eval running (job 2328038, ~17m
  elapsed of 5h walltime, currently the only RUNNING job).
- SFNO Phase 3: SUBMITTED 2026-05-19 (jobs 2328109-12), inferences
  completed; eval chained (job 2328118). 240-axis layout VERIFIED via
  diagnostic 2328355 (operator_type=dhconv, modes_lat=240, low-l-first).
- GraphCast Phase 3: SUBMITTED 2026-05-20 (jobs 2328399-2328414, 16 jobs:
  4 sigmas x 4 inits). Eval chained (job 2328415).
- Aurora dims verified from checkpoint (512 / 1024 / 2048 channels at
  the three encoder/decoder levels). Phase 3 Aurora target = enc_2 only
  (decoder dropped after Phase 2 finding).
- BLOCKED on: SFNO 240-dim layout, edge-level enumeration code in
  earth2studio graphcast wrapper.
- Will proceed only after Phase 2 complete and threshold alignment
  validated against actual checkpoints.

## Literature touchpoints

See [[references]] for full citations. Relevant entries:
- Buizza & Palmer 1995 (singular vectors, dominated by low wavenumbers).
- Shutts 2005 SKEB (perturbs *unresolved* small scales -- inverse spirit).
- Palmer et al. 2009 SPPT.
- Price et al. 2023 GenCast (diffusion adds noise at multiple scales by
  construction).
- Pathak et al. 2022 FourCastNet (FNO spectral bias).
