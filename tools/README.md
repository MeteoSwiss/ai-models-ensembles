# tools/

Analysis, figure and table scripts, the Slurm wrappers that run the heavy ones, and project utilities.
Which script makes which paper item, and which run off-HPC: [docs/REPRODUCING.md](../docs/REPRODUCING.md).

Conventions:

- Paths come from [`_env.py`](_env.py) (`STORE`, `ANALYSIS = STORE/analysis`, `SCRATCH`, `BASELINES`,
  `WB2_*`, `IFS_ENS`, `FIGURES`, `DATA`), each overridable with an `AIENS_*` variable. Shell wrappers use
  `${AIENS_*:-<CSCS default>}` and `--account="${AIENS_ACCOUNT:-ab016}"`.
- Figure/table scripts (`plot_*`, `make_*`, `assemble_*`, `headline_*`, `per_variable_*`) only read small
  CSV/JSON/NPZ files; with `AIENS_STORE=$PWD/tools/data/store` they read the vendored copies in
  [data/](data/README.md) and run anywhere. They write into `../figures/`.
- Compute scripts (`compute_*`, `*_score.py`, `spatial_mean_ssr.py`, ...) read the zarr archives and run as CPU
  sbatch jobs through their `submit_*.sh` wrapper (not on a login node). Durable outputs go to
  `$STORE/analysis/` or next to the experiment; temp files to `$AIENS_SCRATCH/tmp`.
- `model_colors.py`: one colour/marker/line style per baseline and the shared matplotlib rcParams.

## Paper figures

| Script | `../figures/` output | Inputs |
|---|---|---|
| `../figures/draw_perturbation_schematic.py`, `../figures/draw_phase3_schematic.py` | `perturbation_schematic.pdf`, `phase3_schematic.pdf` | none |
| `plot_headline_crpss_vs_lead_8way.py` | `headline_crpss_vs_lead_8way.pdf` (`--with-aifs-det`: talk variant in `talk/`) | intercomparison CSV, AIFS CRPS CSVs, climatology + persistence JSON |
| `plot_7way_spatial_mean_ssr.py` | `tier1b_7way_spatial_mean_ssr.pdf` | `baselines/*/spatial_mean_ssr/spatial_ssr.csv` |
| `plot_refresh_ssr.py` | `refresh_frozen_vs_refresh_ssr.pdf` (`--with-graphcast --out <path>`: talk variant) | spatial-mean SSR of the frozen and `_p6c_reseed` runs |
| `plot_bivariate.py` (needs `swissclim-evaluations`) | `bivariate_Tq_500hPa_7way.pdf`, `bivariate_geostrophic_500hPa.pdf` | `eval/multivariate/*.npz`, `eval_milton/multivariate/*.npz` |
| `plot_spectrogram_delta_2row.py` | `spectrogram_delta_z500_7way.pdf` | `eval/energy_spectra/*bundle*.npz` |
| `plot_rank_histograms.py` | `rank_histograms_240h.pdf` | `analysis/rank_histograms/rank_hist_*.npz` (`compute_rank_histograms.py`) |
| `plot_spread_error.py` | `spread_error.pdf` | `analysis/spread_error_all.csv` (`spread_error_binned.py`) |
| `plot_milton_member_spread_maps.py` | `milton_F8_member_spread_maps_96h.pdf` | forecast zarrs (HPC only) |
| `milton/figures_milton.py`, `milton/plot_aifs_wt_vs_ic_spread.py` | `milton_F1..F5`, `milton_F9_*` | see [milton/README.md](milton/README.md) |

## Paper tables

| Script | `../figures/` output | Inputs |
|---|---|---|
| `headline_8way_table.py` | `headline_8way_table.tex` | as the headline figure |
| `per_variable_crpss_table.py` | `per_variable_crpss_table.tex` | as the headline figure |
| `assemble_production_metrics_table.py` | `table_c1_production_metrics.tex` | per-baseline eval CSVs + `analysis/table_metrics_fixedscale/` (refuses to run without the fixed-scale CSVs unless `ESVS_SCALE_TAG=''`) |
| `assemble_calibration_table.py` | body of `calibration_basis_table.tex` (stdout, pasted by hand) | ablation intercomparisons + per-run eval CSVs + fixed-scale CSVs |
| `patch_calibration_lsd.py [--write]` | LSD cells of `calibration_basis_table.tex` | per-run `energy_ratios_3d_*` CSVs |
| `make_ic_decomp_table.py` | `ic_decomp_table.tex` | `data/ic_decomp_4bb.csv` |
| `make_rival_validation_table.py` | `rival_validation_table.tex` | `analysis/per_init_crps_production_v2.csv` + per-init climatology JSON |

## Compute jobs (Clariden, via `submit_*.sh`)

- CRPS / CRPSS: `compute_per_init_crps.py`, `compute_climatology_crps_vs_eval.py`,
  `compute_climatology_1990_2019.py`, `crpss_common_sample.py`, `block_bootstrap_crpss.py`,
  `calibration_crpss.py`, `compute_perturbation_uplift.py` (+ `submit_uplift_multilead.sh`).
- Spread: `compute_rank_histograms.py`, `spread_error_binned.py`, `spatial_mean_ssr.py`, `per_pixel_ssr.py`,
  `compute_member_vs_mean_rmse.py`, `submit_ic_ssr.sh`, `submit_p6c_reseed_ssr.sh`, `submit_sfno_p6c_ssr.sh`.
- Multivariate proper scores: `energy_variogram_score.py`, `signature_kernel_score.py`,
  `aggregate_signature_kernel_score.py`, `compute_channel_scale.py`, `compare_fixedscale_ranking.py`;
  wrappers `submit_table_metrics_fixedscale.sh` (Tab. C1 + calibration, 128 CPU, 12 h),
  `submit_esvs_sigk_fixedscale.sh`, `submit_esvs_sigk_ic_decomp.sh`, `submit_gc_frozen_esvs_sigk.sh`,
  `submit_ablation_esvs_sigk.sh`, `submit_table_metrics_multilead.sh` (old per-init scale).
- IC vs weight spread: `ic_weight_decomposition.py`, `fill_ic_perturbed_levels.py`, `submit_ic_fill.sh`.
- Other: `compute_persistence_mae.py`, `compare_seed_robustness.py`, `submit_w1_wdkde.sh`,
  `submit_gc_p3frozen_w1.sh`, `submit_geostrophic_milton.sh` (Milton-box bivariate eval).
- Phase 6 refresh runs (inference): `submit_p6c_reseed.sh <sfno|aurora|aifs|graphcast> [spot|full]`,
  `submit_sfno_p6c_production.sh`, `submit_refresh_probes.sh`.
- Spectra re-evaluation after the SwissClim member-0 fix (2026-09-17): `submit_ablation_spectra_reeval.sh`,
  `run_ablation_spectra_batch.sh`.
- Member-distribution diagnostics (post-submission): `compute_ens_anomalies.py`, `compute_point_ensemble.py`,
  `plot_ens_anom_gaussianity.py`, `plot_point_ensemble.py` (+ `submit_ens_anomalies.sh`,
  `submit_point_ensemble.sh`, `submit_ens_anom_plots.sh`) -> `../figures/ens_anom/`.

One-to-one wrappers (`sbatch tools/<wrapper>`; each has its usage in the header):

| Wrapper | Runs | Writes |
|---|---|---|
| `submit_per_init_crps.sh`, `submit_per_init_crps_ifsens.sh` | `compute_per_init_crps.py` | `analysis/per_init_crps_*.csv` |
| `submit_compute_climatology.sh` | `compute_climatology_1990_2019.py` | `$AIENS_SCRATCH/{sigma_clim,empirical_crps_clim}_1990_2019.json` (intermediate) |
| `submit_compute_climatology_vs_eval.sh` | `compute_climatology_crps_vs_eval.py` | `data/crps_clim_eval_*.json` |
| `submit_channel_scale.sh` | `compute_channel_scale.py` | `data/channel_scale_1990_2019.json` |
| `submit_compute_persistence.sh` | `compute_persistence_mae.py` | `data/persistence_mae_112inits.json` |
| `submit_rank_histograms.sh [all|<baseline>...]` | `compute_rank_histograms.py` | `analysis/rank_histograms/` |
| `submit_spread_error.sh` | `spread_error_binned.py` | `$AIENS_SCRATCH/spread_error_binned.csv` (set `OUT_CSV`) |
| `submit_perpixel_ssr.sh` | `per_pixel_ssr.py` | `baselines/<x>/per_pixel_ssr/` |
| `submit_signature_kernel_score.sh` | `signature_kernel_score.py` (old per-init scale) | `$AIENS_SCRATCH/ai-models-ensembles/scratch/sigk/` |
| `submit_ic_weight_decomposition.sh` | `ic_weight_decomposition.py` | `$AIENS_SCRATCH/ic_weight_decomposition.csv` (pass `--out`; kept results: `analysis/ic_decomp_12ch.csv`, `data/ic_decomp_4bb.csv`) |
| `submit_milton_figs_regen.sh` | `milton/figures_milton.py` on a compute node | `../figures/milton_F*` |

## Utilities

| Script | Use |
|---|---|
| `inventory_store.py [--dust] [--publish]` | regenerates the dataset inventory `docs/data/STORE_INVENTORY.md` from the store and `docs/data/catalog.csv`; `--dust` adds sizes (via `dust`, ~2 min) and `docs/data/LARGE_DATASETS.md` (every dataset >= 1 TB, owner, keep/optional/delete); `--publish` copies the data docs into the store |
| `grant_group_read.sh` | read ACLs for one or more groups on the store (xfer partition, run by the owner) |
| `check_gpu.py` | CUDA / GPU visibility inside a container |
| `inspect_weights.py` | dump checkpoint tensor names/shapes/dtypes (source of `_MODEL_LAYER_GROUPS`) |
| `reshard_zarr.py` + `submit_reshard.sh` | rewrite a zarr to the SwissClim chunk/shard layout |
| `run_model_tests.sh`, `submit_model_tests.sh`, `test_model.py` | short-rollout smoke test per model inside its container |
| `_env.py`, `model_colors.py` | shared paths and plotting style |

## milton/

Hurricane Milton case study (tracking, verification, figures): [milton/README.md](milton/README.md).
