# tools/

Scripts that reproduce the paper's figures, tables, and derived numbers.

- **Figure / table scripts** resolve every machine path through
  [`_env.py`](_env.py) and read the cached inputs in [`data/`](data/), so
  they run anywhere (see "Running the analysis off the CSCS box" in the
  [main README](../README.md)).
- **`compute_*` / `*_score` jobs** and their `submit_*.sh` wrappers need the
  HPC and the multi-TB zarr archives; each writes a CSV/JSON the figure or
  table scripts then consume.

Shared: `_env.py` (paths), `model_colors.py` (per-baseline colour/style +
matplotlib rcParams).

## Paper figures

| Script | `../figures/` output |
|---|---|
| `../figures/draw_perturbation_schematic.py` | `perturbation_schematic.pdf` |
| `../figures/draw_phase3_schematic.py` | `phase3_schematic.pdf` |
| `plot_headline_crpss_vs_lead_8way.py` | `headline_crpss_vs_lead_8way.pdf` |
| `plot_7way_spatial_mean_ssr.py` | `tier1b_7way_spatial_mean_ssr.pdf` |
| `plot_bivariate.py` (needs `swissclim-evaluations`) | `bivariate_Tq_500hPa_7way.pdf`, `bivariate_geostrophic_500hPa.pdf` |
| `plot_spectrogram_delta_2row.py` | `spectrogram_delta_z500_7way.pdf` |
| `plot_rank_histograms.py` | `rank_histograms_240h.pdf` |
| `plot_spread_error.py` | `spread_error.pdf` |
| `plot_refresh_ssr.py` | `refresh_frozen_vs_refresh_ssr.pdf` |
| `plot_milton_member_spread_maps.py` | `milton_F8_member_spread_maps_96h.pdf` |
| `milton/figures_milton.py` | `milton_F1..F5` |
| `milton/plot_aifs_wt_vs_ic_spread.py` | `milton_F9_aifs_wt_vs_ic_spread.pdf` |

## Paper tables

| Script | `../figures/` output |
|---|---|
| `headline_8way_table.py` | `headline_8way_table.tex` |
| `per_variable_crpss_table.py` | `per_variable_crpss_table.tex` |
| `assemble_production_metrics_table.py` | `table_c1_production_metrics.tex` |
| `assemble_calibration_table.py` | calibration-basis table body |
| `make_ic_decomp_table.py` | `ic_decomp_table.tex` |
| `make_rival_validation_table.py` | `rival_validation_table.tex` |

## Compute jobs (HPC, via `submit_*.sh`)

- CRPS/CRPSS: `compute_per_init_crps.py`, `compute_climatology_crps_vs_eval.py`,
  `compute_climatology_1990_2019.py`, `crpss_common_sample.py`,
  `block_bootstrap_crpss.py`, `calibration_crpss.py`,
  `compute_perturbation_uplift.py`
- Spread / SSR: `compute_rank_histograms.py`, `spread_error_binned.py`,
  `spatial_mean_ssr.py`, `per_pixel_ssr.py`, `compute_member_vs_mean_rmse.py`
- Proper scores: `signature_kernel_score.py`, `energy_variogram_score.py`,
  `aggregate_signature_kernel_score.py`, `compute_channel_scale.py`,
  `compare_fixedscale_ranking.py`
- Spectra: `compute_psd_lead.py`, `compute_banded_lsd.py`,
  `compute_spectrogram_bug_impact.py`, `compute_lsd_bug_impact.py`
- Other: `compute_persistence_mae.py`, `ic_weight_decomposition.py`,
  `compare_seed_robustness.py`, `fill_ic_perturbed_levels.py`

Committed outputs (`data/`): `crps_clim_eval_*.json`,
`channel_scale_1990_2019.json`, `ic_decomp_4bb.csv`, `sigk_production.csv`.

## Zonal spectra (supplementary)

The routine eval runs on a 24 h lead stride, so short leads are absent from the
`energy_spectra` bundles; `compute_psd_lead.py` recomputes them from
`forecast.zarr` with the same processing.

    sbatch tools/submit_psd_lead.sh 6                            # Phase-1 sweep
    sbatch --partition=normal --time=02:00:00 \
        tools/submit_psd_lead.sh 6 phase2 phase2b                # layer groups

Each job writes `data/psd_<first phase>_lead<LLL>h.npz`, keyed `<phase>:<run>`,
always including the unperturbed Phase-1 run as the sigma=0 reference and
skipping the Phase-6 refresh/frozen/fresh/sub-step variants.

`plot_psd_lead.py` draws one column per backbone (raw PSD over log10 ratio to
ERA5) and takes several `--npz` files at once, so a figure can mix phases:

    python tools/plot_psd_lead.py --npz data/psd_phase2_lead006h.npz \
        --tag phase2 --variable geopotential@500hPa --harmonics 180

`--tag` names the output, `--models` subsets the columns, `--harmonics N` marks
multiples of zonal wavenumber N (180 = Aurora's patch grid), `--runs
<model>=<run>` picks individual runs (used for the production overlay) and
`--model-colors` draws them in the canonical per-baseline colour instead of the
sigma ramp. Colour otherwise encodes sigma and line style the perturbation
target.

## Utilities

| Script | Use |
|---|---|
| `check_gpu.py` | CUDA / GPU visibility inside a container |
| `inspect_weights.py` | dump checkpoint tensor shapes/dtypes (source of `_MODEL_LAYER_GROUPS`) |
| `reshard_zarr.py` + `submit_reshard.sh` | rewrite a zarr to the SwissClim chunk/shard layout |
| `run_model_tests.sh` / `test_model.py` | per-model short-rollout smoke test in a container |

## `milton/`

Hurricane Milton case study: `track_one_init.py` +
`submit_milton_tracker.sh [weight|ic_only|phase5]` (TC tracking),
`restitch_all.py` / `aggregate_tracks.py` (assemble master tracks),
`analyze_milton_stats.py` (summary numbers), then the two Milton figure
scripts above.
