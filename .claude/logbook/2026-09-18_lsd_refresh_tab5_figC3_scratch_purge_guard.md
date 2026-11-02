# 2026-09-18 - Tab. 5 LSD + Fig. C3 refreshed after the member-0 fix; independent verification; fixed-scale CSVs purged

One-liner: production energy_spectra re-evals (3422025-3422032) delivered the pooled LSD; Tab. 5's 16 LSD
cells and Fig. C3 refreshed (paper `559c757`, code `2b5735e`), every number verified by a from-scratch numpy
LSD; a full Tab. 5 regeneration would have silently regressed ES/VS/SIGK because the fixed-scale scratch
CSVs are purged - both generators now refuse instead.

## What changed in the paper (AMS resubmission carries these)
- Tab. 4 (p. 15): 34 LSD cells (committed yesterday, code `e00096c`), 4 within-model bold marks move, no
  production pick changes; caption's selection-rule claim re-checked (graphcast Phase 2, sfno Phase 2/2b are
  within +-0.02 CRPSS and now win LSD but lose SSR/FSS/W1/ES/VS, so "none improved the full bundle" holds).
- Tab. 5 (p. 19): 16 LSD cells; only aifs_perturbed beyond the 3rd decimal (1.160/1.179 -> 1.206/1.240; its
  q LSD 0.892 -> 1.054). FCN3 keeps the optimum. Prose survives ("almost every metric", "FCN3 closest").
- Fig. C3 (p. 40): regenerated from the new `energy_spectra_per_lead_geopotential_500_bundle` NPZs; 0.25%
  of pixels differ at 60 dpi, same 8 panels and colour range.
- No prose quotes an LSD value. 60 pp, 0 overfull, 79 bibitems. `em_upload/` rebuilt (main.tex carries
  the new cells, main.pdf 60 pp).

## Verification (the "double and triple check")
- Eval configs: the 17 spectra-only configs differ from the May eval configs only in module switches,
  `output_mode`, spectrogram flag; same 4 zarrs, same WB2 ERA5 target, same levels/vars, `pooled` set.
- From-scratch numpy LSD (zonal rFFT, k=0 dropped, cos-lat weighted, per (init, lead, level, member),
  mean; scratchpad `indep_lsd.py` / `indep_lsd_prod.py`, not kept): all 17 Tab. 4 rows AND aifs_perturbed /
  fcn3 / sfno_modes10 on the 112-init grid (debug job 3430470, 2m25) agree with the new SwissClim CSVs to
  4 decimals at both leads; the same script restricted to member 0 reproduces the archived pre-fix CSVs to
  4 decimals. So old cells = member 0 exactly, new cells = mean of per-member LSD exactly (App. B's definition).
- Aurora zarrs are 720 lat rows (no -90); the config's `latitudes: [90, -89.75]` selection handles it.
- Old production CSVs reproduce every committed Tab. 5 LSD cell; `patch_calibration_lsd.py` idempotent.

## Trap: scratch purge + silent fallback in the table generators
- `/iopsstor/scratch/.../scratch/table_metrics*` (all 10 dirs) are purged: directories stand, 0 files.
  The fixed-1990-2019-scale ES/VS/SIGK rescore CSVs (internal review item 2) lived ONLY there and were never
  copied to `$STORE/analysis/`. `assemble_production_metrics_table.py` fell back to its hardcoded
  per-init-scale dicts and rewrote ES/VS/SIGK (e.g. aurora ES 0.762 -> 0.730) with no warning.
- Handling: Tab. 5 LSD cells patched in place (16 cells + header note, everything else byte-identical,
  asserted); both generators now `SystemExit` when `ESVS_SCALE_TAG` is set and the fixed CSVs are
  missing. To regenerate a full table again: rerun `tools/submit_table_metrics_fixedscale.sh` first and
  copy its output to `$STORE/analysis/` this time.

## Jobs
- 3422025-3422032 (`normal`, started ~03:10): the 3D lead-time CSVs and the per-lead NPZ bundles used by
  the paper were final by ~10:50; the jobs then spend ~15 min per remaining per-lead bundle (the May run
  took 6 h, not the 5h16 I estimated). Commit made from the final inputs before job exit; after-break
  check: `sacct -j 3422025,...,3422032 -X` COMPLETED and logs show "All 1 module(s) succeeded".
- Overview schematic (files + claude.ai artifact) removed on user request; 7d46849 rewritten as 6d4b2b8.

## Afternoon: pushes, tag move, resubmission package
- All three repos pushed (code `69e3f9f`, paper `559c757`, SwissClim `9d3a7fd`).
- `v1.0.0` on MeteoSwiss/ai-models-ensembles MOVED from `bdc525d` to `69e3f9f` on the user's explicit
  request (annotated tag re-created with the original message + a re-tag note, `git push --force
  origin refs/tags/v1.0.0`, GH release body appended, still Latest). Rationale: App. E states the tag
  is "the revision used to produce the reported results" and nobody consumes the repo before review.
  Default for any later change is a new tag; do not move v1.0.0 again without being asked.
- (paper-process detail kept in the local notes)
