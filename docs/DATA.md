# Data

Every dataset this project produced or consumed: where it is, who can read it, what is inside,
how it was made, and how to remake it. The machine-generated per-experiment listing (counts, schema,
sizes, evaluation products) is [data/STORE_INVENTORY.md](data/STORE_INVENTORY.md); this page explains it.
Everything of 1 TB or more, with owner and a keep/optional/delete verdict for clearing space:
[data/LARGE_DATASETS.md](data/LARGE_DATASETS.md).
Experiment design (what each run tests) is in [EXPERIMENTS.md](EXPERIMENTS.md).

Snapshot: 2026-11-02. A copy of this page is placed as `README.md` at the root of the store
(`python tools/inventory_store.py --dust --publish`), so it can be found from the data side too.

## 1. Locations and access

| What | Path on CSCS Alps (Clariden) | Unix group with read access | Size |
|---|---|---|---|
| Project store (`$STORE`): forecasts, evaluations, analysis caches, container images | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` | `s83` (ACL, granted 2026-11-02 by `tools/grant_group_read.sh`), `a122` (owning group, allocation frozen) | 92 TB |
| IFS-ENS reference forecasts (WB2 archive, 0.25 deg) | `/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr` | `ab016` | 21.3 TB |
| IFS-ENS perturbed analyses (Phase 5 ICs) | `/capstor/store/cscs/swissai/ab016/IFS/ifs_analysis_perturbed_ic.zarr` | `ab016` | 2.2 TB |
| IFS control analyses | `/capstor/store/cscs/swissai/ab016/IFS/ifs_control.zarr` | `ab016` | 0.8 TB |
| ERA5 truth, WeatherBench 2 layout | `/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr`, `.../weatherbench2_2024_2025.zarr`, `.../weatherbench2_original` | `weatherbench` | shared dataset, not ours |
| Handover note with private details + local-only agent memory | `/capstor/store/cscs/swissai/ab016/sadamov/ai-models-ensembles/` | `ab016` | < 1 MB |
| Host Python venv / conda tools | `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles`, `/capstor/store/cscs/mch/s83/sadamov/miniforge3` | `s83` | |
| Small inputs of the figures and tables (vendored) | this repo, `tools/data/` | public | 60 MB |

Notes on access:

- `/capstor/store/cscs/mch/s83` is owned by root with mode `770` and group `s83`, so only `s83` members
  can reach `$STORE`. Members of other projects need CSCS to grant traverse there, or a copy.
- The three IFS zarrs were originally under `/capstor/store/cscs/swissai/a122/IFS/` (the paper runs used
  those). A122 is frozen; a colleague copied them to `ab016/IFS` on 2026-08-21. Verified 2026-11-02: same
  file list and sizes (the a122 copy additionally holds 5 hidden temp files from an interrupted write) and
  identical checksums on sampled chunks. All code defaults now point to the ab016 copies.
- Every path is overridable (`AIENS_STORE`, `AIENS_IFS_ENS`, `AIENS_WB2_22`, `AIENS_WB2_24`,
  `AIENS_WB2_ORIGINAL`, `AIENS_SCRATCH`; see [tools/_env.py](../tools/_env.py)), so a full or partial copy
  elsewhere works without code edits.
- To grant another group: `GROUPS_RW="<group>" sbatch --account=<acct> tools/grant_group_read.sh [dir]`
  (must be run by the file owner).

Durability rules (learned the hard way, do not relax them):

- Never delete a `forecast.zarr`. Each one is hours of GPU time. Superseded runs are kept (status
  "superseded" in the inventory) or moved to `_archived/`, never removed.
- `/iopsstor/scratch/cscs/<user>` is purged without notice. The purge deletes files but leaves the
  directory tree, so `ls` looks healthy; count files, not directories. As of 2026-11-02 every file under
  `/iopsstor/scratch/cscs/sadamov` is gone, including the earth2studio download caches and several analysis
  intermediates. Durable intermediates live in `$STORE/analysis/` and `tools/data/`.
- `$STORE/e2s_cache_backup/cds` (43 GB) is the only copy of the CDS GRIB downloads behind every AIFS and
  AIFS-ENS run. Rebuilding it needs a CDS key and 24-48 h of throttled downloads.

## 2. Store layout

```text
$STORE/
  baselines/<experiment>/<YYYYMMDD_HHMM>/forecast.zarr   production grid (112 inits) and its variants
  baselines/<experiment>/eval/<module>/...                SwissClim evaluation per experiment
  baselines/<experiment>/eval_*_config.yaml               the exact SwissClim config of each eval job
  baselines/<experiment>/{spatial_mean_ssr,per_pixel_ssr,energy_variogram_score,eval_milton}/
  baselines/intercomparison/<module>/                     cross-baseline SwissClim intercomparison
  baselines_m50/<experiment>/20231005_0000/forecast.zarr  50-member single-init diagnostics
  ablation/<phase>/<model_id>/<YYYYMMDD>/<run_tag>/forecast.zarr   4-init ablation grid
  ablation/<phase>/<model_id>/eval/<run_tag>/<module>/... per-run SwissClim evaluation
  ablation/<phase>/<model_id>/intercomparison/            per-phase intercomparison
  ablation/allphases/<model_id>/intercomparison/          cross-phase intercomparison
  ablation/phase6_family_v2/<model_id>/<run>/<YYYYMMDD_HHMM>/forecast.zarr   (different nesting)
  ablation_seed43/<model_id>/<YYYYMMDD>/<run_tag>/forecast.zarr    second-seed check
  analysis/            durable CSV / NPZ / netCDF caches of the analysis jobs
  diagnostics/         small CSV diagnostics (SFNO Phase 6 SSR)
  e2s_cache_backup/    persistent earth2studio caches (CDS GRIBs, AIFS checkpoints) - do not delete
  earth2studio_cache/  SFNO checkpoint package
  <model>.sqsh         the six container images used for every run (+ ai-ens.sqsh, legacy all-in-one)
  *_logs/, logs/       Slurm logs of inference and evaluation jobs
  _archived/           failed or truncated early runs kept for the record
  scratch/             two SFNO debug zarrs
```

`model_id` is the earth2studio-side name: `aurora`, `graphcast_operational`, `sfno`, `aifs` (and
`fcn3`, `atlas`, `aifsens` for the trained-probabilistic models).

## 3. Forecast format

All forecasts use the SwissClim Evaluations schema, written by
[ai_models_ensembles/swissclim_format.py](../ai_models_ensembles/swissclim_format.py) and
[e2s_inference.py](../ai_models_ensembles/e2s_inference.py):

- Zarr v3, one store per init (`init_time` dimension of length 1), consolidated metadata.
- Dimensions and order: surface fields `(ensemble, init_time, lead_time, latitude, longitude)`, pressure-level
  fields `(ensemble, level, init_time, lead_time, latitude, longitude)`.
- Coordinates: `ensemble` int 0..M-1, `init_time` datetime64, `lead_time` timedelta64 (6-hourly from 0 h),
  `level` integer hPa, `latitude` descending from 90, `longitude` 0..359.75.
- Grid: 0.25 deg global. 721 x 1440 (90 to -90) for every model except Aurora, which writes 720 x 1440
  (90 to -89.75, its native grid). SwissClim handles both.
- Variables (ECMWF long names, float32, no units attributes; units are the ERA5/ECMWF conventions):

  | Variable | Units | Levels |
  |---|---|---|
  | `2m_temperature` | K | surface |
  | `mean_sea_level_pressure` | Pa | surface |
  | `10m_u_component_of_wind`, `10m_v_component_of_wind` | m s-1 | surface |
  | `geopotential` | m2 s-2 (geopotential, not height; do not divide by g in plots) | 500, 850 |
  | `temperature` | K | 500, 850 |
  | `u_component_of_wind`, `v_component_of_wind` | m s-1 | 500, 850 |
  | `specific_humidity` | kg kg-1 | 500, 850 |

  Production and m50 stores hold exactly these 9 variables. Some ablation stores (SFNO, AIFS) carry extra
  native outputs (`100m_*` wind, `surface_pressure`, `total_column_water_vapour`, `2m_dewpoint_temperature`,
  `convective_precipitation_6hr`); evaluation ignores them.
- Chunking: one shard per variable (sharding codec, zstd), inner chunk = full ensemble x 1 level x 1 init x 1
  lead x full lat/lon (~40 MB). This matches SwissClim's `DESIRED_CHUNKS`; any other chunking makes the
  evaluation rechunk on read (hours instead of minutes). `_INNER_CHUNKS` in `e2s_inference.py` and
  `INNER_CHUNKS` in `tools/reshard_zarr.py` must stay in sync; `tools/submit_reshard.sh` fixes old stores.
- Unwritten chunks: templates are created with NaN fill (see `.claude/rules/zarr-v3-fill-value.md`); an
  all-NaN slab means the member/lead was never written.
- Leads: production 0-360 h (61 steps) for the headline baselines, 0-240 h (41 steps) for the Phase 5/6 arms
  and all ablation runs. The paper evaluates up to 240 h.
- No run configuration is stored in the zarr attributes. The configuration of every production experiment is
  in `scripts/submit_all_inference.sh` (`EXTRA_FLAGS`, `DATA_SRC`, `MEMBERS`), the ablation run tags encode
  theirs (below), and `baseline_logs/` / `ablation_logs/` keep the generated per-job scripts (`*.sh`, with
  the exact `python -m ai_models_ensembles.cli infer ...` calls) next to their `.out` / `.err`.

Minimal read example (host venv):

```python
import xarray as xr
ds = xr.open_zarr("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_encoder/20230102_0000/forecast.zarr")
z500 = ds.geopotential.sel(level=500).isel(init_time=0)          # (ensemble, lead_time, lat, lon)
spread = z500.std("ensemble", ddof=1)
```

## 4. Init grids

- Production grid, 112 inits: days 2-8 of January, April, July and October in 2023 and 2024, at 00 and 12 UTC
  (`2023-01-02T00` ... `2024-10-08T12`), matching the IFS-ENS archive (`WEEK_STARTS` in
  `scripts/submit_all_inference.sh`).
- Ablation grid, 4 inits: 2023-05-15, 2023-08-15, 2024-02-15, 2024-11-15, 00 UTC (mid-season, disjoint from the
  production grid, so the production picks are evaluated out of sample).
- Milton case study: the 14 production-style inits 2024-10-02 .. 2024-10-08 00/12 UTC of the production grid.
- m50 diagnostics: 2023-10-05 00 UTC only.

## 5. Experiment names and run tags

Production-grid experiments (`baselines/`) are named `<backbone>_<recipe>[_variant]`:

| Suffix | Meaning |
|---|---|
| `aurora_encoder`, `graphcast_all`, `sfno_modes10`, `aifs_perturbed` | the four SPW production picks (weights perturbed once per member) |
| `_ic` | weight noise of the pick plus IFS-ENS perturbed analyses as per-member initial conditions (Phase 5) |
| `_ic_only` | IFS-ENS perturbed ICs, weight noise off |
| `_p6c` / `_p6c_reseed` | weight noise redrawn every 20 steps (Phase 6c); `_reseed` = after the 2026-07-10 seeding fix (use these) |
| `aurora_enc_s044`, `graphcast_m2g`, `graphcast_g2m`, `sfno_enc_s054`, `sfno_enc_s035`, `aifs_all_s010` | runner-up ablation cells rerun on the production grid (rival validation) |
| `_SMOKE` | short smoke tests, ignore |

Ablation run tags encode the configuration:

| Pattern | Meaning |
|---|---|
| `mag_<sigma>_layer_<group>` | multiplicative weight noise `w' = w (1 + sigma xi)` on one layer group (`all`, `encoder`, `processor`, `decoder`, `backbone`, `g2m`, `m2g`, `m2m`, `unet_bottom`, `enc_012`) |
| `mag_<sigma>_modes<L>` | SFNO: noise only on spectral filter coefficients with degree l < L |
| `..._skip1` | SFNO: additionally leave mode l = 0 (global mean) unperturbed |
| `..._refresh<N>_<sigma>` | noise redrawn every N rollout steps (Phase 6); `_fresh` = every step |
| `gcsigma_<s>_gcnodes<n>[_frozen]` | GraphCast Phase 3: activation noise on the n coarsest mesh nodes (42 / 162 / 642 = mesh levels 0 / 1 / 2); `_frozen` = drawn once, otherwise every step |
| `mag_0_layer_all` | unperturbed control (1 member) |

Sigma values with six decimals (e.g. `0.044176`) are the Phase 2 rescaled values
`0.01 * sqrt(N_total / N_group)`; the run tag keeps them exact.

## 6. What each directory contains

The status column of [STORE_INVENTORY.md](data/STORE_INVENTORY.md) uses: `paper` (enters the submitted
paper), `talk only`, `diagnostic` (post-submission analysis), `superseded`, `smoke test`, `failed run`,
`external`, `cache`, `archive`, `debug`. Descriptions per experiment are in
[data/catalog.csv](data/catalog.csv).

### baselines/ (82 TB)

- Headline 8-way comparison: `aurora_encoder`, `graphcast_all`, `sfno_modes10`, `aifs_perturbed` (SPW) vs
  `aifsens`, `atlas`, `fcn3` (trained probabilistic) and `ifs_ens` (physics reference; evaluation outputs only,
  forecasts come from the IFS-ENS zarr). 112 inits x 10 members, 0-360 h each, ~3.4 TB per experiment.
- Phase 5 (IC vs weight spread): `*_ic` (weight + IC, 0-240 h) and `*_ic_only` (IC only, 0-360 h) for all four
  backbones.
- Rival validation: six runner-up configurations on the production grid.
- Phase 6c refresh: `{aurora,aifs,sfno}_p6c_reseed` (paper), `graphcast_p6c_reseed` (talk), and the superseded
  pre-fix `_p6c` runs.
- `aifs_det`: unperturbed single-member AIFS (talk figure). `atmllm`, `esfm`: evaluations of external models,
  not in the paper.
- `intercomparison/`: SwissClim cross-model outputs; `probabilistic/temporal_metrics_combined.csv` is the
  source of the headline CRPSS numbers.

Evaluation outputs per experiment (SwissClim modules, one sbatch per module via
`scripts/evaluate_baselines.sh all <experiment>`):

| Subdir | Content |
|---|---|
| `eval/probabilistic/` | CRPS, SSR (Fortin-corrected), PIT, per variable and lead (`crps_line_*`, `ssr_line_*` CSVs, NPZ) |
| `eval/deterministic/` | RMSE/MAE of the ensemble mean |
| `eval/energy_spectra/` | power spectra and LSD (`energy_ratios_3d_*_enspooled.csv`, per-lead bundles NPZ) |
| `eval/multivariate/` | bivariate histograms (T x q, u x v, geostrophic) |
| `eval/ssim/`, `eval/fss/`, `eval/wd_kde/`, `eval/maps/` | SSIM, fractions skill score, Wasserstein-1 / KDE, map plots |
| `eval/energy_spectra_member0_pre_2026-09-17/` | pre-fix spectra (SwissClim member-0 bug), kept for the record |
| `spatial_mean_ssr/spatial_ssr.csv` | spatial-mean SSR (`tools/spatial_mean_ssr.py`); `*_pre_fortinfix_2026-06-13.csv` = before the Fortin factor fix |
| `per_pixel_ssr/` | per-pixel SSR (`tools/per_pixel_ssr.py`) |
| `energy_variogram_score/score.csv` | ES/VS at the old per-init scale (superseded by `analysis/table_metrics_fixedscale/`) |
| `eval_milton/` | evaluation restricted to the Milton box |

### baselines_m50/ (699 GB)

Seven production picks rerun with 50 members for one init (2023-10-05 00 UTC, 0-240 h) for the
member-distribution diagnostics in `figures/ens_anom/*_m50*` (`.claude/logbook/2026-10-16_m50_single_init_gaussianity.md`).

### ablation/ (8.8 TB) and ablation_seed43/ (319 GB)

Four-init ablation grid, 10 members, 0-240 h; see [EXPERIMENTS.md](EXPERIMENTS.md) for what each phase
tests and which cell won. Phase 6 SFNO refresh runs are in `ablation/phase3/sfno/` (they reuse the Phase 3
tag convention); `phase6_family*/sfno/` hold only intercomparisons. `phase2c/` holds only evaluation leftovers
from before it was renamed to `phase2b/`. `ablation_seed43/` reruns the four production picks with seed 43.

### analysis/ (12 GB), diagnostics/

Durable outputs of the `tools/compute_*` and `tools/submit_*` jobs; each entry is described in the
inventory. `analysis/rank_histograms/` and `analysis/table_metrics_fixedscale/` were regenerated on
2026-11-02 after the scratch purge (their earlier copies on scratch were lost twice).

### Container images

`aurora.sqsh`, `graphcast.sqsh`, `sfno.sqsh`, `fcn3.sqsh`, `atlas.sqsh`, `aifsens.sqsh` (`aifs.sqsh` is a
symlink to it), 17-22 GB each, built from `containers/<model>.dockerfile`. These exact images produced every
forecast; their package freezes are `containers/locks/<model>.txt`. `ai-ens.sqsh` is the legacy all-in-one
image (only `tools/submit_model_tests.sh` still references it).

### Logs

`baseline_logs/` (inference), `baseline_eval_logs/` (evaluation and analysis jobs, including the 2026-11-02
handover jobs), `ablation_logs/`, `logs/` (reshard), `model_test_logs/`. The inference log directories hold
the generated job scripts (`<job>.sh`, exact CLI calls) plus `.out` / `.err`; use them to recover the exact
configuration of any run.

## 7. External inputs

| Input | Used for | Access |
|---|---|---|
| ARCO-ERA5 (`gs://gcp-public-data-arco-era5`, via earth2studio `arco` source) | ICs of Aurora, GraphCast, SFNO, FCN3, Atlas | public, anonymous HTTPS |
| CDS ERA5 (earth2studio `cds` source) | ICs of AIFS and AIFS-ENS (need land/soil fields ARCO lacks) | `~/.cdsapirc` key; cached in `e2s_cache_backup/cds` |
| WeatherBench 2 ERA5 (`weatherbench2_2022_2023.zarr`, `weatherbench2_2024_2025.zarr`) | verification truth | group `weatherbench` on Clariden |
| WB2 ERA5 1990-2019 (anonymous HTTPS to the WB2 bucket) | climatology for CRPSS and the fixed ES/VS/SIGK scale | public; results cached in `tools/data/*1990_2019*.json` |
| IFS-ENS (WB2 archive, `ifs_ens.zarr`) | reference forecast; ~20% whole (init, lead) slabs are NaN in every variable (msl ~30%) | group `ab016` |
| IFS-ENS perturbed analyses (`ifs_analysis_perturbed_ic.zarr`) | Phase 5 per-member ICs | group `ab016` |
| IBTrACS (NHC best track, fetched by `huracanpy`) | Milton verification | public; cached in `analysis/milton_case_study/milton_2024_ibtracs.nc` |
| Model checkpoints | see [ENVIRONMENT.md](ENVIRONMENT.md#model-checkpoints) | public (Hugging Face, NGC, GCS) |

## 8. Data vendored in this repository

`tools/data/` holds everything a figure or table needs that is small enough for git (see
[tools/data/README.md](../tools/data/README.md)):

- top-level JSON/CSV files: climatology CRPS, channel scale, persistence MAE, IC decomposition, SIGK;
- `tools/data/store/`: a 60 MB mirror of exactly those `$STORE` files the figure and table scripts read,
  with the same relative paths. `AIENS_STORE=$PWD/tools/data/store` makes the scripts run on any machine
  (verified 2026-11-02 against the committed outputs; see [REPRODUCING.md](REPRODUCING.md)).

## 9. Regenerating the inventory

```bash
python tools/inventory_store.py                  # facts only, ~1 min
python tools/inventory_store.py --dust           # + sizes (dust, ~1 min for 92 TB) and data/LARGE_DATASETS.md
python tools/inventory_store.py --dust --publish # + copy DATA.md, the inventory and the large-dataset table into $STORE
```

Sizes are disk usage measured with [dust](https://github.com/bootandy/dust) (`~/.cargo/bin/dust` on Clariden),
in decimal units (1 TB = 10^12 bytes).

New experiments show up as "Not in catalog.csv yet" until a line is added to
[data/catalog.csv](data/catalog.csv) (prefix keys are inherited by sub-paths).
