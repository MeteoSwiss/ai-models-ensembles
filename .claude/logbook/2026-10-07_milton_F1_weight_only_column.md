# 2026-10-07 Milton F1 track spaghetti gains an AIFS weight-only column; Milton scratch data purged again

One-line: added `aifs_perturbed` (weight-only) as the 2nd column of `milton_F1_track_spaghetti.pdf`
(IFS-ENS | AIFS weight-only | AIFS weight+IC | AIFS-ENS), re-tracked into a durable capstor dir.

- `/iopsstor/scratch/cscs/sadamov/milton_case_study/` was purged AGAIN (dirs left, 0 files): master/
  verification CSVs, IBTrACS nc, ERA5 control, every per-init track. Only F1 was rebuilt.
- `aifs_perturbed` was never in any `submit_milton_tracker.sh` arm; first tracked today.
- Rebuilt with `AIENS_SCRATCH=$STORE/analysis`, so BASE = `$STORE/analysis/milton_case_study/`:
  4 baselines (ifs_ens, aifs_perturbed, aifs_perturbed_ic, aifsens) x 3 F1 inits (02/04/06 00 UTC),
  job 3604419 (12 tasks, all COMPLETED), plus `milton_2024_ibtracs.nc` (62 records, min usa_pres 895 hPa).
  The `_env.SCRATCH` default still points at iopsstor; F2/F5/table regen needs the full 14-init x all-arm
  re-track + `era5_control.py` + `aggregate_tracks.py` first, ideally with the same AIENS_SCRATCH override.
- Reproduction check: all 9 pre-existing panels reproduce their member counts exactly (1/7/9, 3/2/8,
  8/5/10). Weight-only: 2/10, 2/10, 9/10.
- Fixes in `f1_track_spaghetti`: colour via `color_for` (the `BASELINE_COLORS` dict lacks aifs_perturbed
  and fell back to black, colliding with the IBTrACS line); fonts 9 -> 12 and row height 2.6 -> 2.1 so the
  wider 4-col figure keeps ~6 pt print size; dropped the per-row init prefix from panel titles (duplicated
  the row label, overflowed at 12 pt).
- Paper caption updated (main.tex Fig. milton-spaghetti); "other five ML baselines" still correct since
  aifs_perturbed is not one of the seven headline baselines.
- NOT propagated to the author's private copies of the figure (detail kept in the local notes).
