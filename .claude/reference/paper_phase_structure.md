---
name: paper-phase-structure
description: "Four-phase experimental structure proposed for the weight-perturbation paper, with status and rationale for each"
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Paper structure: four perturbation phases

## High-level narrative

Weight perturbation as a probabilistic-ensemble mechanism for AI weather
forecasts. Compare three perturbation strategies (architectural,
physical, principled) against a uniform-noise baseline. Demonstrate that
physically-motivated targeting produces better ensemble calibration with
less small-scale noise than uniform perturbation -- and validate against
classical IFS ENS where possible.

## Phase 1 -- uniform sigma sweep (control, DONE)

- Multiplicative noise on every float tensor: w' = w * (1 + sigma*N(0,1))
- Magnitudes: 0.001 / 0.003 / 0.01 / 0.03 / 0.1
- 3 models (Aurora / SFNO / GraphCast) x 5 mags x 4 init times x 10 mems
- Status: inference complete, eval complete, intercomparison done.
- Diagnostic finding: signal degradation (wind extremes shrink) +
  small-scale noise injection observed. See [[phase1-signal-vs-noise]].

## Phase 2 -- layer-group ablation (architectural prior, PLANNED)

- Perturb only weights inside one named architectural sub-module.
- Aurora: backbone / decoder / encoder
- SFNO: early / middle / late (third-thirds of FNO stack)
- GraphCast: g2m / m2g / m2m
- Best magnitude per model from Phase 1 (consolidated sigma_full = 0.01),
  sqrt(N_full / N_partial) variance scaling.
- 3 models x 3 groups x 4 inits = 36 runs. Reuses Phase 1 layer=all
  result.
- Cost ~55 NH. Plan and results: `docs/EXPERIMENTS.md` and `.claude/logbook/phase2_aurora_findings.md` (the plan was in the pre-handover MEMORY.md, archived in `memory/`).

## Phase 3 -- physical-scale targeting (physical prior, PLANNED)

- Perturb only the parameters responsible for *coarse spatial scales*
  (lambda >= ~3000-4000 km, planetary/large-synoptic).
- SFNO: spherical-harmonic degree `l <= 10` (slice of spectral weights).
- Aurora: bottleneck Swin layers (encoder_layers.2 + decoder_layers.0).
- GraphCast: runtime hook on edges of mesh refinement levels 0-1
  (NOT weight perturbation -- edge weights shared across edge lengths,
  so the only way to target long edges is at activation level in the
  forward pass).
- Same compute scope as Phase 2 (~55 NH).
- Plan and threshold rationale: [[phase3-physical-perturbation]].
- **Strongest physically-defensible claim of the paper.**

## Phase 4 -- Fisher-information targeting (principled prior, DEFERRED)

- Perturb weights weighted by inverse diagonal Fisher info -> targets
  directions of low loss curvature.
- ~15-20 NH compute + ~3 days code + JAX-side complexity for GraphCast.
- **Deferred** -- not in this paper. Save for follow-up. Full reasoning:
  [[phase4-fisher-info]].

## Why this structure works for the paper

- Phase 1 establishes that uniform perturbation works *but with the
  noted side effects* (extreme degradation, small-scale noise).
- Phase 2 is the "obvious" architectural ablation -- shows that
  *where* you perturb matters, not just *how much*.
- Phase 3 is the headline contribution: physical reasoning about
  *which* scales to perturb leads to better-calibrated ensembles
  with less small-scale noise. Each model uses a different mechanism
  (spectral / depth / runtime-hook) all targeting the same physical
  scale threshold (~3000 km).
- If desired, the paper can mention Phase 4 in Discussion as a
  natural extension without doing the experiment.

## Status of supporting infrastructure (2026-05-18)

- Aurora/GraphCast weight layouts verified against checkpoints. SFNO
  dump pending (job 2293324, direct torch.load path after earth2studio
  OOM'd).
- Two perturbation bugs found and pending fix:
  - SFNO: weight-perturb also touches global_means.npy / global_stds.npy
    (input stats). Only SFNO is affected -- Aurora's stats are hardcoded
    in microsoft-aurora; GraphCast's stats are .nc (not in handlers).
  - GraphCast: weight-perturb silently touches 3 float model_config
    scalars (resolution, radius_query_fraction_edge_length,
    mesh2grid_edge_normalization_factor). Effect small but conceptually
    wrong.
- Fix scope: skip stats files (SFNO) + skip 0-dim float scalars
  (GraphCast). One re-run of SFNO ablation after fix; Aurora and
  GraphCast results stand.
