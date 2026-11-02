---
name: phase3bis-sfno-mode-skip
description: "2026-06-02 ablation testing whether perturbing the SFNO spectral conv weight at l=0 (the global constant mode, mass-conservation-bound for fields like surface pressure) spuriously inflates global-mean spread on rollout. New CLI flag --coarse-mode-skip-first added; first batch K=1 fired as 2454673-76."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Motivation

In the SFNO spectral conv weight tensors `(1, 384, 384, 240)`, the last axis is the total wavenumber `l, low-l-first` (verified diagnostic 2328355). Index 0 = `l=0` = the GLOBAL CONSTANT mode on the sphere (the single zeroth Legendre polynomial). For atmospheric fields:

- MSL / surface pressure: global mean is conserved by atmospheric mass conservation -- `l=0` should not change on rollout
- Meridional wind v: global mean ~ 0 by symmetry
- Divergence: global mean ~ 0 by mass conservation
- Temperature, humidity: global mean varies (seasonal cycle) but slowly relative to weather timescales

Our production Phase 3 SFNO modes10 perturbs `W[..., :10]` = `l=0..9` with `(1 + 0.25 * N_complex)` multiplicative noise. Perturbing `l=0` effectively randomly scales the network's RESPONSE to the global mean of every field -- which conceptually conflicts with mass conservation for the conserved quantities.

Empirically the production SFNO has the BEST spatial-mean SSR of our 3 perturbed baselines (0.70 vs aurora 1.65 vs graphcast 3.38), so the current Phase 3 isn't broken -- residual connections probably anchor the `l=0` response. But "not catastrophic" != "well motivated", and a clean physical argument matters for the paper.

## Hypothesis

Skipping the first K low-l modes (perturbing only `l=K..mode_cut-1`) should:
- Move spatial-mean SSR closer to 1.0 (mean-preserving)
- Clean up LSD at the very lowest wavenumbers
- Maintain CRPS / SSR / SSIM within noise (most of the perturbation budget is in `l=1..9`)
- Provide a cleaner physical narrative in the paper

## CLI change

```
--coarse-mode-skip-first K   Skip the first K low-l modes from --coarse-mode-cut.
                              Default 0 = perturb from l=0 (current behaviour).
                              K=1 perturbs l in [1, coarse_mode_cut).
```

Plumbed through cli.py -> e2s_inference.py -> e2s_perturbation.py:
- `_perturb_array_low_modes(arr, sigma, rng, mode_cut, mode_skip=0)` now slices `arr[..., mode_skip:mode_cut]` instead of `arr[..., :mode_cut]`
- Raises ValueError if `mode_skip < 0` or `>= mode_cut`
- All function signatures and call sites updated for `coarse_mode_skip_first`

## First experiment (2026-06-02 ~10:30)

Fired 4 jobs (one per init) testing K=1:
- 2454673  SFNO 20240215 mag_0.25_modes10_skip1
- 2454674  SFNO 20230515 mag_0.25_modes10_skip1
- 2454675  SFNO 20230815 mag_0.25_modes10_skip1
- 2454676  SFNO 20241115 mag_0.25_modes10_skip1

Same sigma=0.25, mode_cut=10 as production (so the only difference is whether `l=0` is perturbed). Output paths:
`$STORE/ablation/phase3/sfno/<init_tag>/mag_0.25_modes10_skip1/forecast.zarr`

ETA ~10 min per init on 4 GPUs.

## Next steps once forecasts land

1. **Submit SwissClim eval** for `mag_0.25_modes10_skip1` (4 inits combined). Pattern: same as the Phase 6 fresh evals (`tools/eval_phase6_sfno_v2.sh` template).
2. **Spatial-mean SSR**: run `tools/spatial_mean_ssr.py` on the 4 skip1 zarrs against WB2 truth. Compare vs production modes10 (no skip, also 4 inits available at `mag_0.25_modes10/`).
3. **Cross-variant intercomp** -- pull modes10 (production) and modes10_skip1 (this) into one SwissClim intercomp. Look at CRPS / SSR / spatial-mean SSR / LSD / FSS, especially for MSL and v_wind (predicted biggest improvement).
4. **If skip1 wins on spatial-mean SSR with negligible CRPS regression**: paper should switch to skip1 as the production baseline AND update Tab. 2 / Tab. 3.
5. **Possible follow-up**: K=2 (skip `l=0,1`) if K=1 helps but is incomplete. Could fire 4 more jobs at marginal extra cost.

## Results (2026-06-02 evening)

Per-pixel multi-metric at lead 240 h, ablation grid (4 inits), 7 paper vars:

| Config | CRPSS | SSR | SSIM | LSD | FSS95 | W1 |
|---|---|---|---|---|---|---|
| Production sigma=0.25 modes10 | **0.310** | 1.098 | 0.789 | 1.025 | 0.568 | 0.069 |
| Phase 3-bis K=1 (skip l=0) | 0.309 | 1.099 | 0.789 | 1.030 | **0.569** | 0.084 |

**Per-pixel: tied with production within rounding** (CRPSS 0.309 vs 0.310, SSR 1.099 vs 1.098, SSIM 0.789 vs 0.789). Slightly worse W1 (0.084 vs 0.069), slightly better FSS95 (0.569 vs 0.568). Per-pixel-only this would be a draw.

**Spatial-mean SSR result (2026-06-03 morning, batch 2459717 COMPLETED):** Production sfno_modes10 = **0.875**, Phase 3-bis K=1 = **0.930**. The K=1 variant moves spatial-mean SSR meaningfully closer to 1 (delta 0.055, ~44% of the gap to unity), at zero per-pixel cost. **The original physical hypothesis is supported**: perturbing the global-constant mode l=0 was dragging spatial-mean coherence under the calibration floor.

CSVs: `/iopsstor/scratch/cscs/sadamov/full_table3_sfno_240h.csv` + `phase6_joint_ssr_240h.csv`.

**Decision: K=1 is a real (modest) refinement.** Worth mentioning in the paper as a small architecturally-motivated cleanup. Probably not enough to switch production by itself, but it composes with Phase 6c (refresh-N) -- the two effects target orthogonal axes (mode content vs temporal correlation). Stacked variant **Phase 6e = sigma=0.35 N=20 + K=1** running as jobs 2461050-53 (inference) + 2461054 (eval). See [[phase6d-higher-anchor]] for the joint Phase 6 table and the variance-budget framing that makes Phase 6c the natural pair.

## Why not GraphCast / Aurora?

GraphCast Phase 3 uses runtime activation hooks on coarse mesh NODES, not spectral modes. There's no `l=0` analogue in mesh space. Aurora Phase 3 perturbs `unet_bottom` layer weights at the U-net bottleneck -- not a spectral basis. The `l=0` argument is SFNO-specific.

## Related

- [[calibration-winners]] -- production picks; may need update if skip1 wins
- [[paper-phase-structure]] -- Phase 3 motivation
- [[sigma-sweep-design]] -- variance budget; skip1 changes the active mode count from 10 to 9 (negligible)
