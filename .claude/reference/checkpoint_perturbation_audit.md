---
name: checkpoint-perturbation-audit
description: "Verified per-model audit of which weights get perturbed by ai_models_ensembles.e2s_perturbation, established 2026-05-18 from direct dumps. Includes the 2 bugs found and the corrected weight counts."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Checkpoint perturbation audit (verified 2026-05-18)

## Method

For each model, ran a SLURM debug job in the corresponding container that
loaded the actual cached checkpoint via the same path used by
`_perturb_torch` / `_perturb_npz` (sorted-keys-with-numpy / stored-files
respectively) and dumped every tensor name + shape + dtype. Cross-referenced
against the `_HANDLERS` extension dispatch and `_perturb_array` dtype gate.

Dumps: `$STORE/model_test_logs/{aurora,graphcast,sfno}_keys{,_direct}.log`
plus `graphcast_keys_{stored,sorted}.tsv` and `graphcast_keys_summary.txt`.

## Key findings table

| Model | Format | Tensor count | Currently perturbed (`--layer all`) | Should perturb | Bug |
|-------|--------|--------------|--------------------------------------|----------------|-----|
| Aurora | `.ckpt` torch | 644 (all float32, all `net.*`) | 644 | 644 | NONE |
| SFNO | `.tar` torch | 87 (79 float32 + 8 complex64) | 79 only | 87 | **complex-skip: 8 spectral conv filter weights never perturbed** |
| GraphCast | `.npz` numpy | 320 (267 float + 53 non-float) | 267 (incl. 3 wrong) | 264 | **hyperparam-perturb: 3 float scalars in `model_config:*` wrongly perturbed** |

## Per-model checkpoint contents

### Aurora -- CLEAN
- File: `aurora-0.25-pretrained.ckpt`
- Top-level prefixes in state_dict: only `net.*` (644 keys)
  - `net.backbone.*` 594 (Swin-3D U-net, encoder_layers + decoder_layers + time_mlp)
  - `net.decoder.*` 17 (atmos_head, surf_head, level_decoder)
  - `net.encoder.*` 33 (perceiver, level_embed, token_embeds, scale_embed)
- All float32. No complex tensors. No buffers in the state_dict.
- Stats are hardcoded as constants in microsoft-aurora library code, NOT
  in the checkpoint.
- Existing `_MODEL_LAYER_GROUPS["aurora"]` ranges (backbone 0:594, decoder
  594:611, encoder 611:644) are exactly correct.

### SFNO -- BUG: complex-skip
- File: `training_checkpoints/best_ckpt_mp0.tar`
- After `_extract_state_dict` unwraps the `model_state` key: 87 tensors.
- Architecture: encoder (1 layer) + 8 SFNO blocks + decoder (1 layer) +
  residual_transform.
- Per-block (10 tensors, indices N*10 .. N*10+9 for block N=0..7):
  - `blocks.N.filter.filter.weight` (384, 384, 240) **complex64**
  - `blocks.N.mlp.fwd.0.bias` (768,) float32
  - `blocks.N.mlp.fwd.0.weight` (768, 384, 1, 1) float32
  - `blocks.N.mlp.fwd.3.bias` (384,) float32
  - `blocks.N.mlp.fwd.3.weight` (384, 768, 1, 1) float32
  - `blocks.N.norm0.bias` (384,) float32
  - `blocks.N.norm0.weight` (384,) float32
  - `blocks.N.norm1.bias` (384,) float32
  - `blocks.N.norm1.weight` (384,) float32
  - `blocks.N.outer_skip.weight` (384, 384, 1, 1) float32
- Encoder (indices 83-85): `encoder.fwd.0/2.{bias,weight}` (3 tensors)
- Decoder (indices 80-82): `decoder.fwd.0/2.{bias,weight}` (3 tensors)
- Residual (index 86): `residual_transform.weight` (1 tensor)
- Sorted alphabetically: blocks (0-79), decoder (80-82), encoder (83-85),
  residual (86).
- 8 spectral conv weights are complex64; `np.issubdtype(complex, floating)`
  is FALSE, so they're skipped by `_perturb_array` and NEVER PERTURBED.
- Stats files (`global_means.npy`, `global_stds.npy`) are `.npy`, NOT in
  `_HANDLERS` (only `.npz` is), so they're NOT perturbed -- correct
  behavior, no bug here.

### GraphCast -- BUG: hyperparam-perturb
- File: `params/GraphCast_operational - ERA5-HRES 1979-2021 ... .npz`
- 320 total keys in stored order (`f.files`).
- Float keys (267):
  - `params:grid2mesh_gnn` 36 tensors (indices 0-35)
  - `params:mesh2grid_gnn` 30 tensors (indices 36-65)
  - `params:mesh_gnn` 198 tensors (indices 66-263)
  - **3 model_config float scalars** (BUGGY):
    - index 264: `model_config:resolution` (0.25)
    - index 269: `model_config:radius_query_fraction_edge_length`
    - index 270: `model_config:mesh2grid_edge_normalization_factor`
- Non-float keys (53): int64 hyperparams (mesh_size, latent_size, etc.),
  task_config string lists (variable names, pressure levels), description,
  license. All passed through unchanged by `_perturb_array`.
- Existing `_MODEL_LAYER_GROUPS["graphcast_operational"]` ranges (g2m
  0:36, m2g 36:66, m2m 66:264) are exactly correct for the weight section.
  These ranges do NOT include the 3 buggy hyperparam scalars.
- `--layer all` (range 0..320) DOES hit the 3 buggy scalars.
- Stats files (`stats/*.nc`) are `.nc`, NOT in `_HANDLERS`, NOT perturbed.

## Required fixes

```python
# Fix 1 (SFNO): cover complex tensors.
# In _perturb_array, change:
#   if not np.issubdtype(arr.dtype, np.floating):
# to:
#   if not np.issubdtype(arr.dtype, np.inexact):
# (np.inexact covers floating + complex)

# Fix 2 (GraphCast): skip 0-dim model_config scalars.
# In _perturb_npz, add a filter on key prefix or on 0-dim shape:
#   for i in targets:
#       k = keys[i]
#       arr = arrays[k]
#       if k.startswith(("model_config:", "task_config:")):
#           continue
#       arrays[k] = _perturb_array(arr, sigma, rng)
# Or: skip 0-dim float scalars.
# Safer to skip by key prefix since "model_config:" is unambiguously
# config, not weights.
```

## Phase 1 impact

| Model | Phase 1 status | Re-run required? |
|-------|----------------|------------------|
| Aurora | CLEAN | NO |
| SFNO | BROKEN -- 8 spectral conv filters never perturbed (the SFNO core) | **YES** |
| GraphCast | mostly correct, 3 hyperparam scalars perturbed at +-1% (sigma=0.01) | Recommended for cleanliness, not strictly required |

## Corrected `_MODEL_LAYER_GROUPS["sfno"]`

Old (incorrect, based on a buggy inspect_weights.py count of 79):
```
"sfno": {
    "early": (0, 26), "middle": (26, 53), "late": (53, 79),
}
```

New (architecturally correct, 87 total):
```
"sfno": {
    "encoder":   (83, 86),  # module.model.encoder.* (3 tensors)
    "processor": (0, 80),   # module.model.blocks.0..7 (80 tensors)
    "decoder":   (80, 83),  # module.model.decoder.* (3 tensors)
    "residual":  (86, 87),  # module.model.residual_transform (1 tensor)
}
```

Optional finer subdivisions for Phase 3 (still TBD pending verification
of the 240-dim spectral mode layout in `*.filter.filter.weight`):
- early_blocks (0, 30)  -- blocks 0-2
- mid_blocks   (30, 50) -- blocks 3-4
- late_blocks  (50, 80) -- blocks 5-7
- spectral_only -- every 10th index (0,10,20,30,40,50,60,70)
- nonspectral_only -- complement of spectral_only within blocks
