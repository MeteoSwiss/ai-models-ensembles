---
name: gc-phase3-fresh-frozen-audit
description: "Verified ablation inventory for GraphCast Phases 1-3 + the fresh-vs-frozen activation result at sigma=1.0. Empirical anchor for the Phase 3 asymmetry argument and a proposal to expand the GC frozen sweep before paper submission."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# GraphCast Phase 1-3 ablation inventory (verified 2026-06-10)

Listing of actual cell directories under `$STORE/ablation/<phase>/graphcast_operational/`. Confirms the per-phase mechanism:

| Phase | Cells found | Mechanism | Frozen-vs-fresh |
|---|---|---|---|
| Phase 1 | `mag_{0, 0.001, 0.003, 0.01, 0.03, 0.1}_layer_all` (6) | weight perturbation, all 264 NPZ tensors | frozen at member init |
| Phase 2 | `mag_*_layer_{m2m, g2m, m2g}` (3) | weight, GNN layer group | frozen |
| Phase 2b | `mag_*_layer_{g2m, m2g}` (4) | weight, GNN layer group refined | frozen |
| Phase 3 | `gcsigma_*_gcnodes42` (7 fresh + 1 frozen control at sigma=1.0) | activation on first 42 mesh nodes (level-0+1 icosahedral planetary mesh) | fresh per step by default; one frozen control |
| Phase 3b | `gcsigma_*_gcnodes{162, 642}` (2 fresh) | activation, broader mesh subset | fresh |

Production baseline (per [[calibration-winners]]): Phase 1 `mag_0.01_layer_all` = frozen WEIGHT perturbation on all 264 tensors. **GC production baseline matches the SFNO/Aurora/AIFS production convention exactly** -- frozen weight, full-state subset.

The asymmetry the paper has to defend is therefore not "GC mostly uses fresh, others use frozen" -- it is "for Phase 3 (coarse-scale subset target), the GC mechanism is activation+fresh because GNN weight sharing prohibits a weight-tensor scale slice, while SFNO/Aurora keep frozen weight perturbation restricted to a scale-structured subset".

## The single sigma=1.0 frozen control: what it actually shows

Pulled from `phase6_family/graphcast_operational/intercomparison/probabilistic/crps_summary_combined.csv` and `eval/<cell>/spatial_mean_ssr/spatial_ssr.csv`, lead 240 h, level 500:

| Cell | CRPS T2m | CRPS MSL | CRPS Z500 | SSR_sp T2m |
|---|---|---|---|---|
| mag_0.01_layer_all (frozen WEIGHT, prod) | 1.06 | 244 | 273 | (separate file) |
| gcsigma_1.0_gcnodes42_frozen (frozen ACT) | 1.12 | 285 | 302 | 0.288 |
| gcsigma_1.00_gcnodes42 (fresh ACT) | 1.28 | 325 | 349 | 0.155 |
| gcsigma_0.01_gcnodes42 (fresh ACT, low sigma) | 1.59 | 418 | 455 | -- |

Two findings that complicate naive narratives:

1. **At matched sigma=1.0, frozen activation BEATS fresh activation on raw CRPS** (1.12 vs 1.28 T2m). Cannot use "frozen activation performed poorly" as a reason for the asymmetry; the data goes the wrong way.
2. **Spatial-mean SSR direction is OPPOSITE to weight perturbation.** SFNO frozen WEIGHT has SSR_sp = 0.46 (collapsed); fresh-N=20 fixes it to 0.99. GC frozen ACTIVATION has SSR_sp = 0.288 (under-dispersed); fresh activation makes it WORSE (0.155). The random-walk decorrelation argument that motivates SFNO Phase 6c does not transfer from weight to activation perturbation in a straightforward way, because per-step activation noise is short-lived (damped by model dynamics in subsequent steps), so freezing it accumulates more coherent spread than refreshing it.

This is a genuine, paper-relevant finding -- it justifies why Phase 6c needs a dedicated experiment per architecture (weight domain) rather than just inheriting GC's Phase 3 (activation domain) result.

## Proposal: small frozen-activation sigma sweep for GC

Currently the GC Phase 3 grid is asymmetric: 7-point fresh sigma sweep vs 1-point frozen control at sigma=1.0. A reviewer asking "why only one frozen point?" has a fair question.

**Recommended addition: 2-3 frozen-activation cells** at sigma in {0.10, 0.312, 0.60} -- the CRPS-optimal range from the fresh sweep, matching cell naming `gcsigma_<sigma>_gcnodes42_frozen`. Cost:

- 3 cells x 4 inits x 10 mbr x 240h = 120 forecasts
- ~1.5-2h wall per cell = ~6h total in parallel
- Existing `submit_phase5_full.sh`-style launcher with GC_FROZEN=1 env var (already supported by [[phase6-fresh-per-step-weight]] note on `graphcast_coarse_perturbation.py`)

What this buys us:
- 4-point frozen vs 7-point fresh sweep -- balanced enough for honest reporting
- Empirical answer to "what is the CRPS-optimal frozen-activation sigma?"
- Test of whether the frozen-beats-fresh pattern at sigma=1.0 extends to lower sigmas (where the absolute CRPS is closer to the production weight baseline)
- Reviewer pre-emption: "we ran both fresh and frozen across the sigma range; here is the result"

What this risks:
- If frozen activation at sigma ~ 0.1-0.3 becomes competitive with the frozen-weight production baseline, the "weight perturbation is the natural choice for GC" narrative weakens. Need to verify the spatial-mean SSR + LSD + FSS pattern before committing to this in the paper.

## Status (2026-06-10)

- Aurora + AIFS Phase 6c probes set up + smoke-testing tonight (jobs 2512202, 2512203)
- GC frozen-activation expansion: NOT submitted yet, decision-gated on user input
- Paper sentences DRAFTED but NOT committed -- waiting for the GC expansion decision before finalising the asymmetry paragraph

## Related

- [[phase6-fresh-per-step-weight]] -- the Phase 6c plan + JAX vs PyTorch implementation notes
- [[calibration-winners]] -- production baseline picks per model
- [[noise-injection-taxonomy]] -- cross-model noise mechanism table
- [[spatial-mean-vs-pointwise-ssr]] -- the SSR coherence finding that motivates Phase 6c
