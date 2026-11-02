# 2026-09-17 - SwissClim member-0 align bug fixed upstream; three bib entries corrected; LSD/spectrogram regen pending

One-liner: the enspooled bug (`xr.align(join="inner")` collapsing the ensemble axis to member 0, found
2026-09-10 on `visuals`) is fixed in `SwissClim_Evaluations` `9d3a7fd` (research, local); the paper's
Tab. 4 / Tab. 5 LSD columns and Fig. C3 still carry the member-0 numbers because regenerating them needs
the forecast zarrs on capstor, which was unmounted all day. Bib: Selz2023 / Lang2024aifsens /
Bonev2025fcn3 corrected (paper `a9e3d83`) and mirrored in Zotero.

## SwissClim fix (`9d3a7fd` on `research`, not pushed)
- `_compute_spectra_pair` now squeezes any singleton `ensemble` axis (target analysis, deterministic
  prediction, or an `isel(ensemble=[m])` member) before `xr.align`, so it broadcasts instead of being
  intersected with the members. The two later aligns in `run()` (spectrogram paths) then never see a
  mismatched ensemble coordinate. `reduce_ensemble=True` (mean mode) was never affected.
- Regression test `tests/test_energy_spectra_pooled_members.py`: pooled LSD must equal the mean of the
  per-member LSDs, not member 0's, in both orientations. Suite: 120 pass in a throwaway uv venv on scratch
  (`/iopsstor/scratch/cscs/sadamov/tmp/swissclim-venv`, purge-safe to lose); the 2 failures in
  `test_lead_time_handling.py` are pre-existing and matplotlib-version related (`set_axes_locator`).

## What the fix means for the paper (NOT applied yet)
- After the fix, pooled mode yields **mean over members of per-member LSD**, which is exactly what
  App. B states ("per-member log spectral distance ... averaged over the ensemble").
- CAUTION: `tools/compute_lsd_bug_impact.py` (branch `visuals`, `f0cbe46`) computes a THIRD quantity,
  LSD of the ensemble-mean spectrum (`sp.mean("ensemble")` then LSD). Its `lsd_pooled` column is NOT what
  the fixed eval will print and must not be pasted into the tables. Its `lsd_member0` column does
  reproduce the printed Tab. 5 cells for the six ML baselines (ifs_ens differs in the 3rd decimal, NaN
  handling); `aifs_perturbed` was not in that run at all.
- Affected artefacts: Tab. 5 (`figures/table_c1_production_metrics.tex`, LSD 120/240, 8 rows), Tab. 4
  (`figures/calibration_basis_table.tex`, LSD 120/240 for all 20 ablation rows - never measured), and
  Fig. C3 (`figures/spectrogram_delta_z500_7way.pdf`; mean-of-member spectra, expected visually unchanged).
  No prose quotes an LSD value; the interpretation (FCN3 closest, GraphCast/AIFS accumulate small-scale
  power) survives per the 09-10 measurement.
- Regen recipe once capstor is back: `bash scripts/evaluate_baselines.sh all energy_spectra` (8 per-model
  jobs, host venv = editable SwissClim in this checkout), then `evaluate_baselines.sh intercompare`, then
  `tools/assemble_production_metrics_table.py` + `tools/plot_spectrogram_delta_2row.py`. For Tab. 4 the
  ablation eval is bundled (5 modules per phase, `scripts/evaluate_ablation.sh {phase1,...,phase3b}`);
  either accept the full re-eval or add a modules filter first (`EVAL_MODULES` at line 48 is dead, the
  YAML template's `modules:` block is what counts). Then `tools/assemble_calibration_table.py`,
  `sync_figures.sh`, recompile, check that no bold optimum moves.

(paper-process detail kept in the local notes)

## Evening update: re-eval running, Tab. 4 LSD refreshed, impact larger than the 09-10 estimate
- Capstor came back ~19:30. Old `energy_spectra/` outputs archived as `energy_spectra_member0_pre_2026-09-17/`
  next to each of the 8 production evals and the 17 Tab. 4 ablation runs (never deleted).
- Queue reality: `normal` is a 480-node QoS cap (`normalp`) with ~640 higher-priority jobs ahead; `preemptable`
  starts at once but was cleared every 5-18 min (14 of 25 jobs preempted in the first wave, and the cluster
  does NOT requeue by default - `--requeue` added to `_submit_module_sbatch` and the new scripts). Five-hour
  production evals cannot survive there, so they sit on `normal` (3422025-3422032). The 17 short ablation
  evals went through `tools/run_ablation_spectra_batch.sh` on `debug` (4-6 concurrent spectra-only evals per
  node, CSV only, idempotent "have" check on `_enspooled.csv`), batches 3422367 + 3422620.
- Traps hit on the way (all fixed in `tools/submit_ablation_spectra_reeval.sh`): stored ablation configs are
  historical (swissai/a122 paths, `phase2c` dir, `layer_coarse_encoder` run tag, and phase1/aurora's had
  `energy_spectra: mean`), so the script now rebuilds `paths.ml` from the zarrs on disk and forces `pooled`;
  SwissClim drops a config snapshot into the run dir, and the batch runner briefly read that stale copy
  instead of the eval-level one; `preemptable` allows node sharing and the prolog kills shared jobs, so
  whole-node (`--exclusive`) is mandatory there; sbatch spools the script, so `$BASH_SOURCE`-relative repo
  paths are wrong (hardcoded `AIENS_SRC`).
- Tab. 4 (`figures/calibration_basis_table.tex`) LSD column replaced via `tools/patch_calibration_lsd.py`
  (only those 34 cells; bold re-marked within model). Shifts are LARGER than the 09-10 production-grid
  estimate: aifs Phase 2 1.17/1.17 -> 1.22/1.23, sfno Phase 1 1.19/1.17 -> 1.15/1.13, aurora Phase 1
  1.33/1.33 -> 1.31/1.30; bold moves in three places (aurora 120 h to Phase 3 by a sub-rounding margin,
  graphcast 240 h to Phase 3, sfno to Phase 2 at 120 h and Phase 1 at 240 h). Root cause verified directly:
  per-member LSD of specific humidity varies 0.98-1.73 across the 10 AIFS decoder members at z500-equivalent
  scale (member 0 = 1.06, the second lowest), so a single member is a poor estimate for q, while z/t/u/v
  members agree to ~1%. The 09-10 script measured LSD-of-mean-spectrum, which smooths exactly this.
  The selection-rule claim in the Tab. 4 caption ("a rival within +-0.02 displaces the winner only by
  improving the full metric bundle, which none did") re-checked against the new LSD: still true (sfno
  Phase 2/2b beat the Phase 3 pick on LSD now but lose on SSR/ES).
- Still pending: production 8 (Tab. 5 LSD cells + Fig. C3 spectrogram), then `sync_figures.sh`, compile,
  commit both repos. `tools/assemble_calibration_table.py::lsd` now reads the per-run eval CSV.
- Side casualty: the untracked `figures/spw_overview_schematic.html`, `draw_overview_schematic.py` and
  `overview_schematic.{pdf,png}` vanished from `figures/` around the visuals->main checkout at 21:39 (no
  hook, not tracked on either branch; cause not found). Recreated and committed (7d46849), then REMOVED on
  user request 2026-09-18: commit rewritten as 6d4b2b8 (README part only), files deleted, not wanted in the
  repo. The HTML artifact was deleted too (2026-09-18); no copy remains.
