---
name: calibration-winners
description: "Per-model best post-hoc perturbation recipe across all phases (1,2,2c,3,3b) with rationale for argmin-CRPS decision rule"
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

Per-model best perturbation setting. Two stages of selection:

**Stage 1: argmin CRPS at lead 240h (initial pick, 2026-05-25):**

| Model | Mechanism | Threshold | sigma | CRPS | SSR | Source phase |
|---|---|---|---|---|---|---|
| Aurora | layer_group | encoder | 0.025 | 46.4 | 1.05 | Phase 2b |
| GraphCast | layer_group | g2m | 0.014 | 44.4 | 1.27 | Phase 2b |
| SFNO | coarse_modes | modes10 | 0.25 | 46.8 | 1.09 | Phase 3 |

**Stage 2: revised picks after inspecting ALL eval modules across phases (2026-05-27, what we actually use as the perturbation baseline):**

| Model | Mechanism | Threshold | sigma | Source phase | Notes |
|---|---|---|---|---|---|
| Aurora | layer_group | encoder | **0.025** | Phase 2b | UNCHANGED from Stage 1 |
| GraphCast | all_layers | all | **0.01** | Phase 1 | Looking at all phases, full-weight Phase 1 sigma=0.01 wins -- close call vs g2m sigma=0.014 |
| SFNO | coarse_modes | modes10 | **0.25** | Phase 3 | UNCHANGED from Stage 1 (user-confirmed 2026-05-27) |

User decision after seeing the per-phase table: Aurora encoder 0.025 (Phase 2b), GraphCast all_layers 0.01 (Phase 1), SFNO modes10 0.25 (Phase 3).

**Paper framing (decided 2026-05-31)**: the production-baseline selection is documented as a **multi-metric criterion applied symmetrically to all three models**, not pure argmin-CRPS. The criterion: for each per-phase argmin-CRPS winner, inspect the cross-phase intercomp across the full bundle (CRPS, SSR, LSD / energy spectra, FSS, deterministic RMSE/MAE, multivariate bivariate, PIT, maps). The configuration that performs best on the bundle is chosen.

- Aurora: the multi-metric pick coincides with argmin-CRPS (encoder @ 0.025, Phase 2b).
- SFNO: the multi-metric pick coincides with argmin-CRPS (modes10 @ 0.25, Phase 3).
- **GraphCast: the multi-metric pick differs from argmin-CRPS** -- it favours Phase 1 all-layers @ 0.01 (CRPS 30.7, SSR 1.20) over Phase 2b g2m @ 0.014 (CRPS 31.5, SSR ~1.27). The pure-CRPS gap is 0.8 -- within 4-init noise -- but the multi-metric view (SSR closer to 1, cleaner LSD, better FSS at long lead) breaks the tie in favour of Phase 1.

GraphCast is the worked example in the paper that motivates the multi-metric selection rule. Without it the narrative collapses to "we just took argmin-CRPS" which doesn't capture how the decision was actually made.

**Action item before paper submission**: pull the joint CRPS/SSR/LSD/FSS table for GraphCast Phase 1 all@0.01 vs Phase 2b g2m@0.014 to make the case rigorously, not just by assertion. Save as a supplementary table.

**Multi-metric robustness check (verified 2026-05-31, see Tab.~\ref{tab:calibration} = `figures/calibration_basis_table.tex`)**

20 metric x lead cells per model (5 metrics CRPS/SSR/SSIM/LSD/FSS95% x 4 leads 24/72/120/240). The production-baseline pick wins the most cells in each model:

| Model | Production baseline pick | Cells won (of 20) | Next-best candidate | Cells won |
|---|---|---|---|---|
| aurora | Phase 2b encoder@0.025 | **8 / 20** | Phase 3b enc_012@0.015 | 6 / 20 |
| graphcast | Phase 1 all@0.01 | **7 / 20** | Phase 3b n_coarse_162@0.159 | 5 / 20 |
| sfno | Phase 3 modes10@0.25 | **7 / 20** | Phase 3b Lcut_20@0.035 | 5 / 20 |

Pattern: in every model the production-baseline pick has the highest single-row count of column-wise wins. Phase 3b cells consistently come second on FSS95% (because the severely under-dispersive Phase 3b ensembles produce sharper ensemble means that score well on FSS even though CRPS/SSR are terrible). This is post-hoc validation that the multi-metric selection criterion (CRPS@240 + SSR + SSIM + LSD + FSS) gives a self-consistent answer for all three models.

Key cell-by-cell wins for the production picks:
- aurora encoder@0.025: CRPS@120,240; SSR@72,120,240 (3); SSIM@72,120,240 (3).
- graphcast all@0.01: CRPS@120,240; SSR@120; SSIM@72,120,240 (3); LSD@24.
- sfno modes10@0.25: CRPS@240; SSR@120; SSIM@24,72,120,240 (sweep 4); LSD@240.

This robustness check is the bedrock evidence that the production picks are not just argmin-CRPS@240h winners -- they dominate the multi-metric, multi-lead view too.

**CRPS -> CRPSS migration COMPLETE 2026-06-02 ~11:00 (commit `901a039`).** Tab. 3 first metric block now shows CRPSS = 1 - CRPS/CRPS_clim (climatology from WB2 2022-2024, per-pixel temporal std then cos-lat-weighted spatial mean). Per-variable averaging (one CRPSS per var, levels averaged for 3D vars) then mean across 7 paper-headline vars. Reason: variable-mean of RAW CRPS was geopotential-dominated because temperature / specific_humidity contribute 4-7 orders of magnitude less. **ALL 4 production picks remain the within-model argmax CRPSS at lead 240h** -- verified via `/iopsstor/scratch/cscs/sadamov/crpss_ablation_table3.csv`:

| Model | Production pick | CRPSS@240h | Gap to #2 |
|---|---|---|---|
| Aurora | Phase 2b encoder@0.025 | 0.317 | +0.029 |
| GraphCast | Phase 1 all@0.01 | 0.343 | +0.016 |
| SFNO | Phase 3 modes10@0.25 | 0.306 | +0.014 |
| AIFS | Phase 2 decoder@0.0275 | 0.349 | +0.035 |

So CRPSS is metric-only: presentation more honest, conclusions unchanged.

**Wasserstein (W$_1$) added to Tab. 3 as 6th metric block 2026-06-01 evening.** Variable-mean Wasserstein at 4 leads (24/72/120/240) computed from `wd_kde/wd_kde_wasserstein_averaged_enspooled.csv` per (model, phase). 17 cells total: 15 non-AIFS via reruns 2447965-84 + 2449978 (aurora P3 unet retry) + 2450144 (gc P3 1.80 retry), 2 AIFS via 2450519 (P1+P2). Pattern across all 4 models: W values are tight (~0.05-0.09 across phases, within $\pm$0.02), no single phase consistently wins. Production-baseline picks compete fairly: Aurora P2b wins W@72h, AIFS P2 wins W@72h+W@120h, GraphCast P1 and SFNO P3 win no W cell but are within 0.01-0.02 of the per-lead minimum. W column confirms the multi-metric picks aren't penalised on distributional shape — it doesn't move the ranking.

**AIFS production-baseline pick decided 2026-06-01: Phase 2 decoder@0.0275 (NOT Phase 1)**

Verified from raw CSV (full precision) of AIFS Phase 1 + Phase 2 intercomps:

| | Phase 1 (0.01/all) | Phase 2 (0.028/decoder) |
|---|---|---|
| Cell wins (of 20) | **5** | **15** |
| Wins on | FSS 95% all 4 leads + SSR@240h | CRPS all 4 + SSR short-mid + SSIM all 4 + LSD all 4 |

Phase 2 decoder dominates 15/20 -- by far the most decisive multi-metric winner of the four perturb baselines (aurora 8/20, graphcast 7/20, sfno 7/20). Driven by the SAME multi-metric inspection rule used for the other models. Earlier "Phase 1 wins SSR@240h" framing was a single-metric override that gave the wrong answer; the cell-count was rigorously recomputed from CSV to confirm.

Production-baseline config:
- `aurora_encoder`: encoder layer group, sigma=0.025 (Phase 2b)
- `graphcast_all`: all layers, sigma=0.01 (Phase 1)
- `sfno_modes10`: coarse spectral modes l<10, sigma=0.25 (Phase 3)
- `aifs_perturbed`: decoder layer group, sigma=0.0275 (Phase 2) **-- ADDED 2026-06-01**

submit_all_inference.sh updated to include `aifs_perturbed` with the corrected config (`--weight-magnitude 0.0275 --layer decoder`).

Following items closed by this decision:
- TODO "Generate AIFS allphases intercomp": fired 2026-06-01 (jobs 2448487-88) for additional cross-phase confirmation.
- TODO "Add AIFS row block to figures/calibration_basis_table.tex": done 2026-06-01 with the Phase 2 winner now starred.
- TODO "Add AIFS cell-win count to robustness check table above": done.

These are the baselines wired into [submit_all_inference.sh](../../../../pyprojects/ai-models-ensembles/scripts/submit_all_inference.sh) as `aurora_encoder`, `graphcast_all`, `sfno_modes10`. SFNO modes10 inference (112 forecasts) is already done from the prior run -- only Aurora and GraphCast need fresh inference.

Analysis scripts: `/tmp/sadamov_optimal_sigma.py` (per-mechanism breakdown), `/tmp/sadamov_best_perturb_per_model.py` (interp + argmin comparison).

**Why argmin CRPS, not interp-to-SSR=1**

Why: We considered interp-to-SSR=1 (linear in log-σ) as a "principled" recipe, but only 4 init_times means the σ-vs-(SSR,CRPS) curves are noisy enough that interp-extrapolation costs ~11 CRPS units on GraphCast (m2g at SSR=1 gives 55.3 vs g2m at SSR=1.27 gives 44.4) and ~2 on SFNO. With more inits we could justify interp; with 4 we cannot.

How to apply: When recommending a final recipe per model, use the observed argmin CRPS row even if SSR is slightly above 1. State the SSR honestly in the paper -- "modestly overdispersive" is not "miscalibrated" for ensemble forecasting, esp. compared to underdispersive baselines that look more confident than they are.

**Structural finding (paper claim)**

Each model rewards a different abstraction level:
- Aurora: input-projection (encoder MLP) layer-group perturbation wins
- GraphCast: g2m (encoder, grid-to-mesh) layer-group perturbation wins
- SFNO: low-frequency spectral modes (degree l < 10) coarse-modes perturbation wins -- the only model where Phase 3 physics-motivated targeting beats Phase 2 layer-group targeting

Two of three models reward architectural targeting of the input-projection / message-passing layer; only SFNO benefits from the physics-motivated low-frequency-modes story. See [[paper-phase-structure]].

**vs Phase 1 all-layer (sigma=0.01) baseline**: Aurora encoder σ=0.025 beats Phase 1 (CRPS 46.4 vs ~67 at SSR=1). GraphCast g2m σ=0.014 beats Phase 1. SFNO modes10 σ=0.25 beats Phase 1.

**Caveats**:
- All values are at lead 240h. CRPS curves differ at shorter leads (24h, 120h).
- SSR is var-averaged; per-variable SSR varies (geopotential is more overdispersive than 2m_temperature typically).
- "Test power" caveat: 4 inits is small; some of the small CRPS gaps between adjacent sigma points are within noise.

**Full Phase 2b sweep -- 12 experiments total, not the "Aurora encoder fine sweep" I had been describing** (verified by `find phase2b/<model>/<init>/` on disk, 2026-05-31). For each model Phase 2b refined sigma on the top two Phase 2 layer groups (not just one):

| Model | Run tags | CRPS (var+lead mean) |
|---|---|---|
| Aurora | mag_0.025_layer_encoder | **31.92 (Phase 2b winner)** |
|  | mag_0.044176_layer_encoder (Phase 2 anchor) | 32.82 |
|  | mag_0.040_layer_decoder | 35.49 |
|  | mag_0.060_layer_encoder | 36.74 |
|  | mag_0.085_layer_decoder | 42.78 |
| GraphCast | mag_0.014_layer_g2m | **31.46 (Phase 2b winner)** |
|  | mag_0.029665_layer_m2g (Phase 2 anchor) | 32.09 |
|  | mag_0.050_layer_m2g | 32.46 |
|  | mag_0.018_layer_m2g | 34.77 |
|  | mag_0.027080_layer_g2m (Phase 2 anchor) | 35.02 |
|  | mag_0.045_layer_g2m | 42.88 |
| SFNO | mag_0.053852_layer_encoder (Phase 2 anchor) | **36.63 (Phase 2b winner)** |
|  | mag_0.035_layer_encoder | 38.10 |
|  | mag_0.100_layer_decoder | 38.12 |
|  | mag_0.080_layer_encoder | 38.31 |
|  | mag_0.075_layer_decoder | 39.65 |

SFNO Phase 2b case is notable: the four finer sigmas (encoder 0.035/0.080, decoder 0.075/0.100) all underperformed the original Phase 2 sqrt(N) anchor on encoder (0.054). So SFNO's Phase 2 winner stays at the sqrt(N) anchor, not a refined value -- the dagger story for SFNO is the opposite of Aurora and GraphCast.
