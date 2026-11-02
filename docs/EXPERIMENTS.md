# Experiments

What was run, why, and what came out of it. Paper: "Stochastically Perturbed Weights: Ensembles from
Deterministic Machine-Learning Weather Models" (Adamov, Fuhrer, Knutti, Schemm; submitted to AMS AIES,
preprint on arXiv physics.ao-ph). Where the data of each experiment lives: [DATA.md](DATA.md). How each
figure and table is made from it: [REPRODUCING.md](REPRODUCING.md).

## Method: SPW

Stochastically Perturbed Weights (SPW) turns a frozen deterministic checkpoint into an ensemble without
retraining. Each member gets multiplicative Gaussian noise on (a subset of) the weights,

    w' = w * (1 + sigma * xi),   xi ~ N(0, 1) i.i.d. per weight,

drawn once per member with seed `seed + member_id` (`seed = 42`), then the member is rolled out
autoregressively like the deterministic model. Optional initial-condition (IC) noise uses seed
`seed + member_id + 10000`. Normaliser statistics are never perturbed (SFNO `.npy` stats, AIFS
pre/post-processors, GraphCast config scalars). Implementation:
[ai_models_ensembles/e2s_perturbation.py](../ai_models_ensembles/e2s_perturbation.py) (checkpoint
perturbation, `_MODEL_LAYER_GROUPS`), the `*_fresh_perturbation.py` modules (noise redrawn during the
rollout, Phase 6) and [graphcast_coarse_perturbation.py](../ai_models_ensembles/graphcast_coarse_perturbation.py)
(Phase 3 GraphCast activation noise).

Backbones perturbed:

| Name in code | Model | Checkpoint tensors (audit `.claude/reference/checkpoint_perturbation_audit.md`) |
|---|---|---|
| `aurora` | Aurora 0.25 deg pretrained (Microsoft) | 644 float32 tensors: backbone 0:594, decoder 594:611, encoder 611:644 |
| `graphcast_operational` | GraphCast operational (DeepMind, JAX) | 264 NPZ params: g2m 0:36, m2g 36:66, m2m 66:264 |
| `sfno` | SFNO 73ch small (NVIDIA) | 87 tensors, 8 complex64 spectral filters (idx 0,10,...,70); groups encoder 3 / processor 80 / decoder 3 |
| `aifs` | AIFS single 1.1 (ECMWF) | 250 keys / 255 M params: encoder, processor, decoder |

Comparison models: AIFS-ENS 1.0, Atlas, FourCastNet 3 (all trained probabilistically from scratch) and
IFS-ENS (10 stratified members of the WB2 archive) as the physics reference.

## Grids

- Ablation: 4 inits (2023-05-15, 2023-08-15, 2024-02-15, 2024-11-15 00 UTC), 10 members, 0-240 h, pressure
  levels 500 + 850 hPa. Used to choose the recipe per backbone.
- Production: 112 inits (days 2-8 of Jan/Apr/Jul/Oct 2023-2024, 00/12 UTC), 10 members, 0-360 h (paper
  evaluates to 240 h). Disjoint from the ablation inits, so the production scores are out of sample.

## Ablation phases

| Phase | Store path | What is perturbed | Result |
|---|---|---|---|
| 1 | `ablation/phase1/` | all weights, sigma in {0.001, 0.003, 0.01, 0.03, 0.1} (+ unperturbed control) | spread grows with sigma; uniform noise also degrades extremes and injects small-scale noise. GraphCast pick: all @ 0.01 |
| 2 | `ablation/phase2/` | one layer group at `0.01 * sqrt(N_total / N_group)` (variance-matched to Phase 1 sigma = 0.01) | where you perturb matters. AIFS pick: decoder @ 0.0275 |
| 2b | `ablation/phase2b/` | finer sigma on the two best Phase 2 groups per model | Aurora pick: encoder @ 0.025 |
| 3 | `ablation/phase3/` | coarse scales only: SFNO spectral modes l < 10; Aurora U-Net bottleneck; GraphCast activation noise on the 42 coarsest mesh nodes | SFNO pick: modes10 @ 0.25 (the one model where scale-targeting beats layer-targeting) |
| 3b | `ablation/phase3b/` | alternative cut-offs (SFNO l < 20, Aurora enc_012, GraphCast 162/642 nodes) | under-dispersive, runner-up on FSS only |
| 4 | - | Fisher-information-weighted noise | deferred, not run |
| 5 | `baselines/*_ic`, `*_ic_only` | production picks + IFS-ENS perturbed analyses as ICs; IC-only arm | IC spread dominates days 0-3 and decays; weight spread carries the medium range (Tab. ic_decomp) |
| 6 | `ablation/phase3/sfno/*refresh*`, `ablation/phase6_family*`, `baselines/*_p6c_reseed` | noise redrawn every N steps instead of once per member | refresh helps only the under-dispersed SFNO; barely moves Aurora (1.56 -> 1.43) and AIFS (2.54 -> 2.50) spatial-mean SSR at 240 h |

Production picks (rerun on the 112-init grid as `baselines/<name>`):

| Experiment | Backbone | Recipe | Chosen in |
|---|---|---|---|
| `aurora_encoder` | Aurora | encoder group, sigma = 0.025 | Phase 2b |
| `graphcast_all` | GraphCast | all weights, sigma = 0.01 | Phase 1 |
| `sfno_modes10` | SFNO | spectral modes l < 10, sigma = 0.25 | Phase 3 |
| `aifs_perturbed` | AIFS | decoder group, sigma = 0.0275 | Phase 2 |

Selection rule: a multi-metric bundle applied the same way to every model (CRPSS at 240 h, SSR, SSIM, LSD,
FSS, ES/VS/SIGK; Tab. calibration = `figures/calibration_basis_table.tex`), not argmin CRPSS. GraphCast is
the one case where the bundle and argmin CRPSS disagree (Phase 1 all @ 0.01 preferred over Phase 2b
g2m @ 0.014, a CRPS gap inside the 4-init noise). Bold cells in the tables were decided at full precision,
before rounding. Background: `.claude/logbook/calibration_winners.md`.

Robustness checks behind the picks (all in the paper):

- Rival validation: the runner-up cells within about 0.02 CRPSS of each pick were rerun on the held-out
  production grid (`baselines/{aurora_enc_s044,graphcast_m2g,graphcast_g2m,sfno_enc_s054,sfno_enc_s035,aifs_all_s010}`,
  `figures/rival_validation_table.tex`).
- Second seed: the four picks rerun with seed 43 on the ablation grid (`ablation_seed43/`); winner margins move
  by at most 0.006, so cells within that band are treated as ties.
- Paired block bootstrap over the 8 seasonal weeks (`tools/block_bootstrap_crpss.py`): each of the four
  headline CRPSS gaps has the same sign in all 8 blocks (exact sign-test p = 0.0078).

## Metrics

All verification runs through SwissClim Evaluations (pinned commit, see [ENVIRONMENT.md](ENVIRONMENT.md))
against WeatherBench 2 ERA5 at 0.25 deg; variables 2t, msl, z/t/u/v/q at 500 and 850 hPa.

- CRPSS = 1 - CRPS / CRPS_clim, with a 1990-2019 ERA5 climatology (`tools/compute_climatology_1990_2019.py`,
  cached in `tools/data/crps_clim_eval_*.json`).
- SSR (spread-skill ratio) with the Fortin finite-ensemble factor, reported as spatial mean
  (`tools/spatial_mean_ssr.py`) and per pixel (`tools/per_pixel_ssr.py`).
- ES / VS / SIGK (energy, variogram, signature-kernel scores) on channels standardised by a fixed 1990-2019
  climatological scale (`tools/data/channel_scale_1990_2019.json`), which keeps them proper.
- LSD (log-spectral distance), SSIM, FSS, Wasserstein-1, bivariate histograms from the SwissClim modules.
- IFS-ENS has whole (init, lead) NaN slabs (~20%, msl ~30%). It is excluded from ES/VS/SIGK/W1; the
  IFS-ENS-valid common-sample CRPSS check changes the ranking by at most 0.0022.

## Milton case study

Hurricane Milton (October 2024), 14 inits 2024-10-02..08 00/12 UTC: TC tracks with TempestExtremes
(DetectNodes/StitchNodes) for SPW AIFS weight-only, weight+IC, IC-only, AIFS-ENS and IFS-ENS, verified against
the NHC best track (IBTrACS). Pipeline: [tools/milton/README.md](../tools/milton/README.md). The IFS-ENS
detection rate is 71% (a 14% figure seen in some notes was a double member-filter bug).

## Settled decisions (do not undo without a reason)

- Headline = weight-only SPW for four backbones; `aifs_perturbed_ic` is not added to the headline (IC lives in
  Phase 5 and Milton).
- Refresh clause `sigma_N <= 1.93` is correct (not 1.58; SFNO SSR CSVs in `$STORE/diagnostics/sfno_phase6_ssr/`).
- Geopotential is shown in m2 s-2, never converted to height.
- Aurora's spectral spikes are patch-grid harmonics amplified by the decoder group only; they are not "noise
  entering deeper".
- Phase 6 seeding: before 2026-07-10 the fresh hooks used `seed + step`, which collapsed draws across members
  and tensors and faked a "refresh pulls over-dispersed models to 1" result. Fixed with `_mix_seed`
  (splitmix64); all paper Phase 6 numbers come from the `_reseed` runs.

## Reproducibility

- Seeds as above; PyTorch runs use `cudnn.deterministic=True`, `benchmark=False`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- GraphCast (JAX/XLA) is not bit-reproducible run to run (GPU reductions are nondeterministic): two identical
  runs differ by ~0.02 K mean / 0.43 K max in 2t at 24 h, far below the ensemble spread. AIFS is not
  bit-reproducible on GPU either (found 2026-10-16).
- Release used for the paper: tag `v1.0.0` (= `69e3f9f`). Later commits only add post-submission material
  (talk figures, diagnostics, this handover).

## Known open issues (for the revision)

- The related-work sentence on Whitaker et al. mischaracterises their noisy-subnetwork ensembles (they are
  post-hoc and retraining-free too, but additive, on a sparse subset, members selected on validation data).
- "Architecture excludes the weight-space refresh" for GraphCast is overstated: a refresh-every-20 GraphCast
  run exists (`baselines/graphcast_p6c_reseed`, 240 h spatial-mean SSR 3.20 -> 2.97, same "barely moves" result).
- Raster DPI of `milton_F3` (~261) and `milton_F4` (~279) is below the AMS 300 dpi bar.
- The submitted Milton F9 draws IFS-ENS from 2 of its 10 members (it predates the 2026-09-02 member-filter
  fix); the corrected figure is in `figures/`, the caption and text numbers are unaffected.
- No cited source for sigma = 0.01-0.03; it is empirical (span of the Phase 1 winners).
- Code-level: `evaluate_ablation.sh` has no guard against evaluating with fewer than 4 inits present;
  `e2s_data.py` `MergedDataSource` silently falls back to CDS on `KeyError`.
