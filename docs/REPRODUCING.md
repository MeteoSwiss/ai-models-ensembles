# Reproducing the figures and tables

Three levels, from cheapest to complete:

1. Any machine, minutes: regenerate the figures and tables from the 60 MB of inputs vendored in
   `tools/data/`. Covers 17 of the 26 paper items plus the talk/diagnostic extras below.
2. Clariden, hours: rerun the analysis jobs (`tools/submit_*.sh`) on the stored forecasts and evaluations in
   `$STORE`, then step 1. Needs `s83` access.
3. Clariden, days: rerun inference (`scripts/submit_*`) and SwissClim evaluation from scratch. Needs the
   containers, a Slurm account, CDS credentials for AIFS.

Data locations: [DATA.md](DATA.md). Environment: [ENVIRONMENT.md](ENVIRONMENT.md). Cluster usage: [HPC.md](HPC.md).

## 1. Any machine

```bash
git clone https://github.com/MeteoSwiss/ai-models-ensembles && cd ai-models-ensembles
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r env/host-requirements.txt && uv pip install -e . --no-deps
export AIENS_STORE=$PWD/tools/data/store AIENS_SCRATCH=$PWD/tools/data/store/analysis MPLBACKEND=Agg
python tools/plot_headline_crpss_vs_lead_8way.py      # -> figures/headline_crpss_vs_lead_8way.pdf
```

Every script writes into `figures/` (overwriting the committed file, so `git diff` shows any change). Run them
from the repository root. Verified on 2026-11-02 with all non-vendored paths pointed at `/nonexistent`:
every output marked "verified" below came out byte-identical to the committed file (PDFs compared with
the `CreationDate` stamp removed).

### Paper items

| Paper item | Output in `figures/` | Command | Off-HPC | Status |
|---|---|---|---|---|
| Fig. perturbation schematic | `perturbation_schematic.pdf` | `python figures/draw_perturbation_schematic.py` | yes | no inputs |
| Fig. Phase 3 schematic | `phase3_schematic.pdf` | `python figures/draw_phase3_schematic.py` | yes | no inputs |
| Fig. headline CRPSS vs lead | `headline_crpss_vs_lead_8way.pdf` | `python tools/plot_headline_crpss_vs_lead_8way.py` | yes | verified |
| Tab. headline (8-way) | `headline_8way_table.tex` | `python tools/headline_8way_table.py` | yes | verified (only the source-path comment line changes) |
| Tab. per-variable CRPSS (App.) | `per_variable_crpss_table.tex` | `python tools/per_variable_crpss_table.py` | yes | verified (same) |
| Fig. spatial-mean SSR | `tier1b_7way_spatial_mean_ssr.pdf` | `python tools/plot_7way_spatial_mean_ssr.py` | yes | see note 1 |
| Fig. refresh SSR | `refresh_frozen_vs_refresh_ssr.pdf` | `python tools/plot_refresh_ssr.py` | yes | verified |
| Figs. bivariate T-q and geostrophic | `bivariate_Tq_500hPa_7way.pdf`, `bivariate_geostrophic_500hPa.pdf` | `python tools/plot_bivariate.py` | yes (needs `swissclim-evaluations`) | verified |
| Fig. spectrogram delta z500 | `spectrogram_delta_z500_7way.pdf` | `python tools/plot_spectrogram_delta_2row.py` | yes | verified |
| Fig. spread-error | `spread_error.pdf` | `python tools/plot_spread_error.py` | yes | verified |
| Fig. rank histograms | `rank_histograms_240h.pdf` | `python tools/plot_rank_histograms.py` | yes, after vendoring `analysis/rank_histograms/` | inputs regenerated 2026-11-02 (job 3833540), see section 4 |
| Tab. C1 production metrics | `table_c1_production_metrics.tex` | `python tools/assemble_production_metrics_table.py` | yes, after vendoring `analysis/table_metrics_fixedscale/` | inputs regenerated 2026-11-02 (job 3833541), see section 4 |
| Tab. calibration basis | `calibration_basis_table.tex` | `python tools/assemble_calibration_table.py` (prints the body; pasted by hand) + `python tools/patch_calibration_lsd.py --write` (LSD cells) | yes, same | LSD patch verified; body as Tab. C1 |
| Tab. IC decomposition | `ic_decomp_table.tex` | `python tools/make_ic_decomp_table.py` | yes | verified |
| Tab. rival validation | `rival_validation_table.tex` | `python tools/make_rival_validation_table.py` | yes | verified |
| Fig. Milton F1 tracks | `milton_F1_track_spaghetti.pdf` | see note 2 | yes | verified |
| Figs. Milton F2, F5, F9 + Tab. Milton summary | `milton_F2_*`, `milton_F5_*`, `milton_F9_*`, `milton_summary_table.tex` | `tools/milton/figures_milton.py`, `plot_aifs_wt_vs_ic_spread.py` | no | intermediates scratch-purged, see section 5 |
| Figs. Milton F3, F4 | `milton_F3_cascading_combined.pdf`, `milton_F4_storm_relative_composite.pdf` | `tools/milton/figures_milton.py` | no (reads forecast zarrs) | HPC only; F4 also needs the regenerated ERA5 window |
| Fig. Milton F8 member/spread maps | `milton_F8_member_spread_maps_96h.pdf` | `python tools/plot_milton_member_spread_maps.py` | no (reads zarrs) | HPC only |
| Tab. experiment design | `experiments_tables.tex` | hand-written | - | static |
| Tab. inference cost | `inference_cost_table.tex` | hand-written from Slurm accounting (`sacct`) | - | static |

Note 1: the committed `tier1b_7way_spatial_mean_ssr.pdf` (and the paper's copy) was drawn at 11:24 on
2026-10-07 from an audit version of the IFS-ENS SSR CSV; the canonical CSV was rewritten at 14:43 the same day
by the repo tool. The two differ by at most 0.0008 in SSR (mean 0.00008), invisible in the figure. Running
the script now (vendored or on `$STORE`) gives the canonical version; vendored and `$STORE` runs are
byte-identical.

Note 2: `figures_milton.main()` first loads the purged master CSVs, so draw F1 alone:

```bash
AIENS_SCRATCH=$PWD/tools/data/store/analysis python -c "import sys; sys.path[:0]=['tools/milton','tools']; \
import figures_milton as fm, aggregate_tracks as at; fm.f1_track_spaghetti(at.load_master())"
```

### Talk, slide and diagnostic figures (not in the paper)

| Output | Command | Off-HPC |
|---|---|---|
| `talk/headline_crpss_vs_lead_8way_aifs_det.{pdf,png}` | `python tools/plot_headline_crpss_vs_lead_8way.py --with-aifs-det` | yes, verified |
| refresh SSR with GraphCast (talk) | `python tools/plot_refresh_ssr.py --with-graphcast --out <path>` | yes |
| `slides/milton_z500_plume_box_{mean,median}[_nolegend].{pdf,png}` | `python tools/milton/plot_z500_plume.py` | yes, verified (uses the cached netCDFs) |
| `ens_anom/point_members_zurich_*`, `point_member_qq_zurich_*` | `python tools/plot_point_ensemble.py --members-only` (`--case` for the m50 pair) | yes, verified |
| `ens_anom/point_grid_values_240h`, `point_grid_opposite_values_240h` | `python tools/plot_point_ensemble.py` | yes (it then stops at the band grid, which needs 4 GB of NPZ) |
| `ens_anom/member_shapiro_rate_m50_20231005_0000` | `python tools/plot_ens_anom_gaussianity.py --case` | yes, verified |
| `ens_anom/member_{values,anomaly_qq,zscore_qq}_240h`, `member_shapiro_rate`, `band_grid_values_240h` | `python tools/plot_ens_anom_gaussianity.py` / `plot_point_ensemble.py` | no: reads `analysis/ens_anom_gaussianity/*.npz` (551 MB per model) |
| `talk/milton_wind_frames/*` (21 frames as PNG + PDF, GIF, MP4) | `source tools/milton/env.sh && python tools/milton/plot_wind_frames.py` | no: 143 MB netCDF cache in `analysis/milton_wind_frames/`; needs ffmpeg. Rerun 2026-11-02 (job 3833547): PNGs/GIF byte-identical, PDFs added for every frame |

## 2. On Clariden from `$STORE`

Unset `AIENS_STORE` (defaults point at the CSCS paths) and run the same commands. To refresh an analysis
input, rerun its job; every job writes into `$STORE/analysis/` or next to the experiment:

| Input | Job | Resources |
|---|---|---|
| `baselines/intercomparison/...` | `bash scripts/evaluate_baselines.sh intercompare` | CPU, hours |
| `baselines/<x>/eval/<module>/` | `bash scripts/evaluate_baselines.sh all <x>` (one sbatch per module) | 800 GB, up to 12 h per module |
| `baselines/<x>/spatial_mean_ssr/` | `sbatch tools/submit_ic_ssr.sh` / `tools/submit_p6c_reseed_ssr.sh` (`tools/spatial_mean_ssr.py`) | CPU, ~12 min per experiment; not on a login node |
| `analysis/per_init_crps_*.csv` | `sbatch tools/submit_per_init_crps.sh`, `tools/submit_per_init_crps_ifsens.sh` | CPU |
| `analysis/spread_error_*.csv` | `sbatch tools/submit_spread_error.sh` (writes to scratch by default: set `OUT_CSV=$STORE/analysis/...`) | CPU |
| `analysis/rank_histograms/` | `sbatch tools/submit_rank_histograms.sh all` | 16 CPU, ~2 h |
| `analysis/table_metrics_fixedscale/` | `sbatch tools/submit_table_metrics_fixedscale.sh` | 128 CPU, 12 h |
| `tools/data/crps_clim_eval_*.json`, `channel_scale_1990_2019.json` | `sbatch tools/submit_compute_climatology_vs_eval.sh`, `tools/submit_channel_scale.sh` | CPU, 13-16 h (HTTPS to WB2) |
| `tools/data/persistence_mae_112inits.json` | `sbatch tools/submit_compute_persistence.sh` | CPU |

After regenerating anything that a vendored file mirrors, refresh the mirror (`tools/data/README.md`).

## 3. From scratch

```bash
bash containers/submit_build.sh all                       # 6 images -> $STORE/<model>.sqsh (or reuse them)
bash scripts/submit_ablation.sh {phase1|phase2|phase2b|phase3|phase3b} [model]
bash scripts/evaluate_ablation.sh <phase> && bash scripts/evaluate_ablation.sh intercompare <phase>
bash scripts/submit_all_inference.sh [experiment ...]     # production grid; PER_INIT=1 for SFNO
bash scripts/evaluate_baselines.sh all <experiment>       # per experiment; never without an experiment
bash scripts/evaluate_baselines.sh intercompare
# then the analysis jobs of section 2, then section 1
```

Cost orientation (GH200 node = 4 GPUs): one production experiment (112 inits x 10 members x 360 h) is about
112 node-jobs of 8-15 min with warm caches (cold ARCO/CDS caches: ~2 h per init); Atlas is the slow one
(~38 min per member). Evaluation of one experiment is 8 jobs of up to 12 h. The paper's per-model
inference costs are in `figures/inference_cost_table.tex`.

## 4. Items regenerated on 2026-11-02

The scratch purge had deleted the inputs of the rank-histogram figure (`rank_hist_*.npz`) and of the fixed-scale
ES/VS/SIGK columns of Tab. C1 and the calibration table. Both jobs were rerun with their outputs moved to
`$STORE/analysis/` (`ANALYSIS` in `tools/_env.py`) so they cannot be purged again:

- `sbatch tools/submit_rank_histograms.sh all` -> `$STORE/analysis/rank_histograms/rank_hist_<baseline>.npz`
- `sbatch tools/submit_table_metrics_fixedscale.sh` -> `$STORE/analysis/table_metrics_fixedscale/{esvs,sigk}_<label>_fixed_L{120,240}.csv`

## 5. Milton F2 / F5 / F9 and the summary table (HPC recipe)

The per-init tracks behind these were on scratch and are gone; only the 12 F1 tracks were rebuilt
(2026-10-07, into `$STORE/analysis/milton_case_study/`). To rebuild everything, run with
`AIENS_SCRATCH=$STORE/analysis` so the tracks land on the store:

1. `source tools/milton/env.sh` (TempestExtremes, udunits, the host venv).
2. Track the arms: `bash tools/milton/submit_milton_tracker.sh weight`, `... ic_only`, `... phase5`
   (Slurm arrays, one task per (baseline, init), ~5 min each). `aifs_perturbed` and `ifs_ens` are in no arm:
   build a pair list like the F1 rebuild did (`<baseline>|<YYYYMMDD_HHMM>` lines, see the script) and submit
   it the same way.
3. `python tools/milton/era5_control.py` (ERA5 truth track + `era5_milton_window.nc`, needed by F4 and the
   stats; copy `milton_2024_ibtracs.nc` into `$AIENS_SCRATCH/milton_case_study/` first).
4. `python tools/milton/restitch_all.py` (if tracker settings changed), `python tools/milton/aggregate_tracks.py`
   (master + verification CSVs), `python tools/milton/analyze_milton_stats.py` (summary CSV).
5. `python tools/milton/figures_milton.py` (F1-F5) and `python tools/milton/plot_aifs_wt_vs_ic_spread.py` (F9);
   transcribe `milton_per_baseline_summary.csv` into `milton_summary_table.tex`.

Expected: IFS-ENS detection rate 71% (140-pair denominator). Background: `.claude/logbook/2026-09-01_*milton*`,
`2026-09-02_milton_ifsens_detection_double_filter_bug.md`, `2026-10-07_milton_F1_weight_only_column.md`.

## 6. Paper repository

The LaTeX source is a separate private repository (Overleaf mirror). Its `figures/` folder holds copies of
the files in this repo's `figures/`; after regenerating, the paper repo's `sync_figures.sh` copies exactly the
files `main.tex` includes. Diff the legend of every regenerated figure against the committed one before
syncing (a regeneration once silently dropped a baseline from a legend).
