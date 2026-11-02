# 2026-06-15 (attended) figures -> vector PDF, fresh-hook count verification, Zotero Buizza1995

Attended daytime session (Opus 4.8). Cleared most of the nightly "Next up" list with the user.

## Deferred nightly review items - RESOLVED (paper 4da7acd)
(paper-process detail kept in the local notes)

## Figures -> vector PDF (the big PNG->PDF item)
Milton F1-F5 (code c93f724, figures 4747d02; paper 10cba06):
- tools/milton/figures_milton.py F1-F5 savefig now emits png+pdf.
- F3 (cascading, cartopy) + F4 (storm-rel composite): cartopy GeoContourSet IGNORES rasterized=True
  (warns "Rasterization ... will be ignored"). Fix = ax.set_rasterization_zorder(0) + contourf/contour at
  zorder=-1 (coastlines/borders zorder=3 stay vector). 38 MB -> ~1 MB per F3 panel. F4 is plain axes so
  plain rasterized=True works there. Verified visually (rendered F3 aifsens: rasterized fills, vector
  coastlines/text/colorbar).
- main() F3 list: added atlas, fcn3; DROPPED ifs_ens (source /capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr
  no longer exists -> f3 crashes on all-None). ifs_ens F3 stays PNG in the paper (milton_F3_cascading_ifs_ens.png).
- pyproject.toml: added h5netcdf (the host venv had NO netcdf backend; netCDF4 + h5netcdf both missing).
  Installed wheel-only on aarch64 (uv pip install --only-binary=:all: h5netcdf h5py).

temporalCRPS x3 + z500 spectrogram (code cbfb3ef; paper f74966b):
- These figure names are RENAMED SwissClim intercomp outputs (committed long ago in 7b8414b), not produced
  by any in-repo generator. temporalCRPS = ablation phase1 per-model prob intercomp temporal_CRPS_2m_temperature;
  spectrogram = 7-way baselines energy_spectra geopotential_500 ..._spectrogram_delta_compare.
- The intercomp PLOT step reads cached per-run npz/csv (NO GPU, NO zarr re-read) -> cheap CPU re-plot.
- SwissClim save_figure (helpers.py) only saves the given path; energy_spectra uses plt.savefig DIRECTLY
  (bypasses save_figure). cli intercompare SUBPROCESSES `swissclim-evaluations-compare`, so an in-process
  monkeypatch in the parent does NOT reach it. SOLUTION: a sitecustomize.py on PYTHONPATH (auto-imported by
  the child) that patches Figure.savefig to dup a .pdf for every .png. Files (throwaway, scratch):
  /iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/pdf_regen/{sitecustomize.py,pdf_launcher.py}.
- Ran 4 short debug-partition CPU jobs (intercompare --module spectra / --module prob, --out-dir to scratch
  temp). 7-way = atlas,fcn3,aifsens,sfno_modes10,aurora_encoder,graphcast_all,aifs_perturbed (NOT ifs_ens).
  temporalCRPS phase1 prob = 5 perturbed mags (mag_0.001..0.1, mag_0 excluded). debug QOS caps ~2 jobs;
  cancelled jobs early once the one target PDF was written. Copied 4 PDFs into figures/, paper -> .pdf.
- NOTE: paper commit f74966b ALSO swept in the user's concurrent model-soups table (tab:member-vs-mean)
  because `git add main.tex` grabbed the whole file. User said leave-as-is; amended the commit message to
  cover both. Lesson: stage hunks, not whole files, when the user may be editing live.

main.pdf after all figure swaps: 17.3 MB, compiles clean, no undefined cites/refs.

## Fresh-hook per-AR-step count verification (item 3) - aurora + sfno CONFIRMED
Goal: confirm the Phase 6 fresh-per-step weight noise re-samples once per AR step (paper's matched-budget
claim sigma_p*sqrt(60)=1.93 only holds if the hook fires once/step -> 40 fires @240h, 60 @360h).
- GOTCHA: the hook log is guarded `if step == 1` (prints only the first fire) - step_counter increments
  silently every fire. So grepping the log undercounts. Temporarily un-guarded the per-step print in all 3
  fresh_perturbation.py modules (aurora/aifs/sfno) to count. THESE EDITS ARE TEMPORARY DEBUG - REVERT THEM.
- Ran 48h (8-step) 1-member mag_0 inference per model in each container (partition normal, gpu:4, output to
  scratch fresh_verify/). `python -m ai_models_ensembles.cli` uses the BIND-MOUNTED code so the edit applies.
- RESULTS: aurora 8 fires steps 1-8 (one per step). sfno 8 spectral hooks installed, each fires once/step =
  64 fires, 8 distinct steps. Both confirm per-step firing -> 40 @240h. aifs (job 2536113) still running at
  ~28 min (slow: downloads 948M ckpt + many grib2 fields; source=cds actually pulled ECMWF open-data dated
  2026-01-01 NOT the requested 2024-02-15 - open-data has no 2024 archive; harmless for the COUNT but not a
  clean init - confirms the user's "reuse the cached CDS init" point). Background watcher bd60qfl4j waits on it.
- item 4 (GraphCast bit-identical reproducibility): SKIPPED per user (determinism well established, no novel claim).

## Zotero Buizza1995 (item 5) - DONE
(paper-process detail kept in the local notes)

## STILL TO DO next session
- REVERT the temporary per-step logging in ai_models_ensembles/{aurora,aifs,sfno}_fresh_perturbation.py
  (the `else: print(... PRE_HOOK fired step=...)` branches). Held until aifs job 2536113 confirms its count.
- Read aifs 2536113 result from scratch/fresh_verify/logs/aifsb_*.out (expect 29-param decoder hook firing
  once/step; watcher bd60qfl4j reports).
- Optional cleanup: scratch/pdf_regen/ + scratch/fresh_verify/ + scratch/zotero_check_buizza.py are throwaway.
