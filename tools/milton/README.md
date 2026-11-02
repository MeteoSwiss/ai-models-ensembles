# tools/milton/ - Hurricane Milton case study

Tropical-cyclone tracking and verification for Hurricane Milton (landfall Florida, 2024-10-10) on the 14
inits 2024-10-02 .. 2024-10-08, 00/12 UTC, of the production grid. Paper figures F1-F5, F9 and the Milton
summary table; talk figures (z500 plume, wind animation). Detector (`track_one_init.py`): TempestExtremes
`DetectNodes` on MSL minima with two closed-contour criteria (MSL rise of 200 Pa within 5.5 deg, warm core
via the 500-850 hPa thickness, -58.8 m within 6.5 deg), merged within 6 deg; `StitchNodes` with a 5 deg range,
at least 36 h, gaps up to 18 h (bridges the IFS-ENS NaN leads, a no-op for the ML models) and 10 m wind
>= 17 m/s on at least 3 steps. Verified against the NHC best track (IBTrACS).

## Environment

`source tools/milton/env.sh` puts TempestExtremes and udunits from the conda base on the path
(`AIENS_MILTON_CONDA`, default the Clariden miniforge base; elsewhere `conda env create -f env/conda-tools.yml`)
and sets `MILTON_PYTHON` (the host venv; `AIENS_PY` overrides). All scripts resolve paths through
`tools/_env.py`: working directory `BASE = $AIENS_SCRATCH/milton_case_study`. Run with
`AIENS_SCRATCH=$AIENS_STORE/analysis` so the results land on the store (scratch is purged; it already
deleted the 2026-09 tracks).

## Pipeline

All steps in one node job: `sbatch --account=${AIENS_ACCOUNT:-ab016} tools/milton/submit_milton_rebuild.sh`
(224 pairs, `PAR=20` at a time, then ERA5 control, tables, F1-F5 and F9; `SKIP_FIGS=1` stops after the tables).
Prefer it to the array of `submit_milton_tracker.sh`: nodes are allocated exclusively, so the array pays a full
node for every 2-4 min task (~11 node-hours vs ~1).

| Step | Script | Output (under `BASE`) |
|---|---|---|
| 1. track one (baseline, init), 10 members | `track_one_init.py <baseline> <YYYYMMDD_HHMM>`, normally via `submit_milton_tracker.sh [weight|ic_only|phase5]` (Slurm array, ~5 min per task) | `tracks/<baseline>/<init>/milton_tracks.csv` + raw detect/stitch files |
| 2. ERA5 control track | `era5_control.py` (needs `milton_2024_ibtracs.nc` in `BASE`) | `era5_stitch.txt`, `era5_milton_window.nc`, a sanity plot |
| 3. re-stitch with final settings (optional) | `restitch_all.py` | updated `tracks/` |
| 4. master + verification tables | `aggregate_tracks.py` | `milton_master_tracks.csv`, `milton_verification.csv` |
| 5. summary numbers | `analyze_milton_stats.py` | `milton_per_baseline_summary.csv` (transcribed into `figures/milton_summary_table.tex`) |
| 6. figures | `figures_milton.py` (F1-F5), `plot_aifs_wt_vs_ic_spread.py` (F9) | `figures/milton_F*.pdf` |
| talk | `plot_z500_plume.py`, `plot_wind_frames.py` | `figures/slides/`, `figures/talk/milton_wind_frames/` (cached inputs in `$STORE/analysis/milton_{z500_plume,wind_frames}`) |

Arms of `submit_milton_tracker.sh`: `weight` = aurora_encoder, graphcast_all, sfno_modes10, aifsens, fcn3,
atlas; `ic_only` = the four `*_ic_only`; `phase5` = the four `*_ic`. `aifs_perturbed` (weight-only AIFS) and
`ifs_ens` are in no arm: give them a hand-made pair list (`<baseline>|<init>` lines) as the 2026-10-07 F1
rebuild did. IFS-ENS uses the 10 stratified members 0,5,...,45 and keeps its WB2 NaN leads as missing
(`AIENS_MILTON_GAPFILL=1` would reconstruct them; the paper setting is off).

## State at handover (2026-11-02)

- Fully rebuilt 2026-11-02 with `submit_milton_rebuild.sh` (job 3833735, 35 min, 224 pairs, 0 failures):
  tracks, ERA5 control, master/verification tables and summary in `$STORE/analysis/milton_case_study/`; all but
  the ERA5 window netCDF vendored in `tools/data/store/analysis/milton_case_study/`. F1, F2, F5, F9 and the
  summary table rebuild anywhere ([docs/REPRODUCING.md](../../docs/REPRODUCING.md), note 2); F3 and F4 read
  forecast zarrs. Summary table: all 40 values reproduce; F1-F5 byte-identical.
- F9 corrected: the version up to `v1.0.0` (and in the submitted paper) drew IFS-ENS from 2 of 10 members
  (REPRODUCING.md note 3).
- Correct IFS-ENS detection rate: 71% (140-pair denominator). A 14% figure in older notes was a double member
  filter in `aggregate_tracks.py`, fixed 2026-09-02.
- F3 and F4 are rasterised below 300 dpi; regenerate with a higher `savefig` dpi for the revision.
