# figures/

Generated figures (all available as PDF) and LaTeX tables, plus the two schematic generators. Every file
here is written by a script; the commands, inputs and off-HPC status are in
[docs/REPRODUCING.md](../docs/REPRODUCING.md). The paper repository keeps copies of the files its
`main.tex` includes (its `sync_figures.sh` copies them from here).

## In the paper (26 items included by `main.tex`)

| File | Paper element | Generator |
|---|---|---|
| `perturbation_schematic.pdf` | SPW schematic | `draw_perturbation_schematic.py` |
| `phase3_schematic.pdf` | coarse-scale (Phase 3) schematic | `draw_phase3_schematic.py` |
| `experiments_tables.tex` | experiment design tables | hand-written |
| `calibration_basis_table.tex` | ablation calibration table (selection of the production picks) | `tools/assemble_calibration_table.py` + `tools/patch_calibration_lsd.py` |
| `headline_crpss_vs_lead_8way.pdf` | headline CRPSS vs lead, 8 systems | `tools/plot_headline_crpss_vs_lead_8way.py` |
| `headline_8way_table.tex` | headline CRPSS table | `tools/headline_8way_table.py` |
| `tier1b_7way_spatial_mean_ssr.pdf` | spatial-mean SSR vs lead (Fig. 4) | `tools/plot_7way_spatial_mean_ssr.py` |
| `table_c1_production_metrics.tex` | Tab. C1, all metrics of the production baselines | `tools/assemble_production_metrics_table.py` |
| `inference_cost_table.tex` | inference and training cost | hand-written from Slurm accounting and the model papers |
| `per_variable_crpss_table.tex` | per-variable CRPSS (appendix) | `tools/per_variable_crpss_table.py` |
| `bivariate_Tq_500hPa_7way.pdf`, `bivariate_geostrophic_500hPa.pdf` | bivariate distributions | `tools/plot_bivariate.py` |
| `spectrogram_delta_z500_7way.pdf` | spectral error vs lead, z500 | `tools/plot_spectrogram_delta_2row.py` |
| `rank_histograms_240h.pdf` | rank histograms at 240 h | `tools/plot_rank_histograms.py` |
| `spread_error.pdf` | binned spread vs error | `tools/plot_spread_error.py` |
| `ic_decomp_table.tex` | IC vs weight spread decomposition | `tools/make_ic_decomp_table.py` |
| `rival_validation_table.tex` | runner-up configurations on held-out inits | `tools/make_rival_validation_table.py` |
| `refresh_frozen_vs_refresh_ssr.pdf` | noise refresh (Phase 6) | `tools/plot_refresh_ssr.py` |
| `milton_F1_track_spaghetti.pdf` ... `milton_F5_verification_vs_lead.pdf` | Milton tracks, intensity, cascading, composite, verification | `tools/milton/figures_milton.py` |
| `milton_F8_member_spread_maps_96h.pdf` | Milton member and spread maps | `tools/plot_milton_member_spread_maps.py` |
| `milton_F9_aifs_wt_vs_ic_spread.pdf` | AIFS weight vs IC track spread | `tools/milton/plot_aifs_wt_vs_ic_spread.py` |
| `milton_summary_table.tex` | Milton detection/error summary | transcribed from `milton_per_baseline_summary.csv` (`tools/milton/analyze_milton_stats.py`) |

## Not in the paper

| Path | What | Generator |
|---|---|---|
| `talk/headline_crpss_vs_lead_8way_aifs_det.{pdf,png}` | headline figure plus unperturbed AIFS (sigma = 0) | `plot_headline_crpss_vs_lead_8way.py --with-aifs-det` |
| `talk/milton_wind_frames/` | 10 m wind of Milton, 0-120 h: 21 frames (`milton_wind_fNNN.{png,pdf}`), `milton_wind.gif`, particle video `milton_wind_particles_1920.mp4` (+ poster PNG/PDF), `frames.json` | `tools/milton/plot_wind_frames.py` (the full-resolution `milton_wind_particles.mp4`, 76 MB, is not committed) |
| `slides/milton_z500_plume_box_{mean,median}[_nolegend].{pdf,png}` | z500 member plumes at landfall | `tools/milton/plot_z500_plume.py` |
| `ens_anom/` | member-distribution (Gaussianity) diagnostics: Q-Q plots, Shapiro-Wilk rejection rates, single-point member values, incl. the 50-member case (`*_m50*`) | `tools/plot_ens_anom_gaussianity.py`, `tools/plot_point_ensemble.py` |
| `experiments_tables_preview.*` | local LaTeX preview build of the tables (gitignored) | - |

Units: geopotential is shown in m2 s-2 (geopotential, not height). Colours and line styles per baseline come
from `tools/model_colors.py`, identical across all figures.
