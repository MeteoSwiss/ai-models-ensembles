---
name: sfno-phase6-hook-bug
description: "SFNO fresh-per-step weight perturbation hook installs successfully (8/8 spectral conv modules, complex64 weight shape (1,384,384,240)) but produces bit-identical members across all 10 ensemble seeds. Forward pass apparently bypasses or caches around `module.weight` so the pre_hook's in-place delta does not propagate."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Symptom

Verified 2026-06-01 16:30 by reading `phase3/sfno/20230515/mag_0.60_modes10_fresh/forecast.zarr`:

- members 0 and 5, 2m_temperature, lead 96h, init 2023-05-15
- `(mem_0 - mem_5).max() == 0` (bit-identical)
- per-pixel stddev across 10 members = `1.8e-05` (numerical noise floor)
- spatial-mean SSR from `tools/spatial_mean_ssr.py` reports `spread ~ 5e-14`, ratio fresh/frozen ~ `1e-12` (clearly broken, not the predicted `1/√T ≈ 0.13` reduction)

All 8 SFNO Phase 6 fresh forecasts (4 inits × 2 sigmas) are effectively deterministic across the 10 members.

## Diagnosis already done

Debug job 2450649 (instrumented `install_fresh_modes_hooks` to dump module tree):

```
total modules: 134
model class: SFNO (earth2studio wrapper)
inner model: ModelWrapper
8 modules ending in `.filter.filter` found:
  model.model.model.blocks.0.filter.filter  shape=(1, 384, 384, 240)  dtype=torch.complex64
  ... blocks.1 through blocks.7 ...
[SFNO_FRESH] installed 8 hooks (sigma=0.6, mode_cut=10, base_seed=42)
installed 16 handles
```

So: hooks ARE attached to the 8 correct modules. weight shape and dtype are correct (complex64 (1, 384, 384, 240) -- (1 batch, 384 c_in, 384 c_out, 240 modes_l)).

## Root cause (diagnosed 2026-06-01 17:10)

**Debug job 2450698**: ran a 2-member inference with the instrumented pre_hook that prints on step 1. ZERO `[SFNO_FRESH]` prints in the log. The pre_hook never fired.

**Why**: `e2s_inference.py` had a `shared_model` short-circuit that bypassed `_model_for_member` entirely whenever `weight_magnitude <= 0 and graph_coarse_sigma <= 0`. With `SFNO_FRESH=1` we explicitly set `weight_magnitude=0` (to skip checkpoint perturbation in favour of runtime hooks), so the shared-model path fired and `_model_for_member` (where `sfno_fresh_perturbation.maybe_install_from_env` lives) was never called. The same hole existed on both inference paths:

- `_run_members_sequential` line 424: `use_shared_model = weight_magnitude <= 0 and graph_coarse_sigma <= 0`
- `_gpu_worker` (parallel) line 288: `if weight_magnitude > 0 or graph_coarse_sigma > 0:` (else load_model)

All 8 production Phase 6 SFNO forecasts ran via the parallel path, so they all hit this -- every member loaded a fresh unperturbed SFNO checkpoint, no hook was attached, so all members were bit-identical.

## Fix

Both branches now also check `SFNO_FRESH=1 && model_name=="sfno"` and route through `_model_for_member` when it's set, so each member gets a freshly-loaded model with its own forward_pre_hook attached at the right `base_seed = seed + member_id`. Same change in two places:

- `_run_members_sequential` `use_shared_model` predicate
- `_gpu_worker` per-member model-load conditional

Verification: debug3 job 2450713 (COMPLETED 2:39) confirmed:
- 8 [SFNO_FRESH] PRE_HOOK fired prints in the log per member
- `|noise|.mean = 0.886`, `|delta_weight|.mean = 0.083 - 0.110` per spectral conv (~10% of weight at sigma=0.6)
- Members 0 vs 1 at lead 96h: `|m0 - m1|.mean = 0.62 K, max = 7.1 K, per-pixel stddev = 0.31 K` (vs 0/0 / 1.8e-05 in the broken version)

Production rerun: 8 jobs 2451881-2451888 fired via `sfno_fresh_pilot.sh` after `rm -rf` of the 8 broken zarrs. 6 of 8 COMPLETED with working hook (PRE_HOOK fired prints confirmed in every log); 1 hit the cursed-init SFNO SIGSEGV pattern unrelated to the hook (2451883 20230815 σ=0.60, relaunched as 2451950); 1 still RUNNING (2451888 20241115 σ=1.93). Once both land the new Phase 6 SFNO spatial-mean SSR plot can be regenerated to verify the predicted √T spread reduction.

## Forecasts NOT deleted

The 8 SFNO Phase 6 fresh forecasts (~1.4 TB) are parked, not deleted. They cost GPU time; will be deleted/superseded once a fix lands and a rerun produces valid data.

## Paper impact

Phase 6 SFNO conclusion is NOT in the paper. The Phase 6a paper claim still rests on the GraphCast fresh-vs-frozen test (which used a JAX activation hook, not torch pre_hook, and verified produced spread reduction of ~3.7x in line with prediction). SFNO Phase 6 was the second leg to widen the claim across architectures -- still wanted, not blocking submission.
