# 2026-06-15 swissclim native PDF export; sitecustomize pdf-dup shim retired; ifs_ens F3 -> PDF

One-line: SwissClim research branch now emits vector PDFs natively (env-gated), replacing the throwaway
sitecustomize savefig monkeypatch; ifs_ens F3 panel re-rendered as PDF after its source zarr came back.

## SwissClim PDF export (swiss-ai/SwissClim_Evaluations, branch research, commit ff609b9)
- Before: only the bivariate/multivariate module wrote a .pdf (helpers.save_figure + the 11 intercomparison
  modules all called plt/fig.savefig with .png only). `main` has zero PDF export. That is why paper vector
  PDFs needed a downstream hack.
- Now: new `savefig_formats(out_path, *, fig=None, formats=None, **kwargs)` in helpers.py saves the figure in
  every extension in `SWISSCLIM_FIG_FORMATS` (default "png"). All 39 direct savefig sites across the 11
  modules + save_figure route through it. multivariate forces formats=("png","pdf") to keep its always-on pdf.
- Usage going forward: `export SWISSCLIM_FIG_FORMATS=png,pdf` before the ai-ens / swissclim compare run ->
  vector PDFs land next to every png. Default runs are unchanged (png only), so routine eval is not doubled.

## Workaround retired (delete + update)
- Deleted `scratch/pdf_regen/sitecustomize.py` and `scratch/pdf_regen/pdf_launcher.py` (the Figure.savefig
  monkeypatch that dup'd png->pdf, needed because energy_spectra savefig runs in a CLI subprocess). The rest
  of scratch/pdf_regen/ (logs, tmp yaml, output subdirs) is harmless scratch, left in place.
- The MEMORY.md note describing the sitecustomize shim is now historical - the mechanism is the env var.

## ifs_ens F3 panel (main repo commit fb6e6cf; paper commit ff4e5a5)
- Source zarr restored at `/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr` (old `mch/s83/IFS` path gone).
  Updated `_msl_ensmean_at` path in tools/milton/figures_milton.py + re-added f3_cascading_detection("ifs_ens")
  to main(). Rendered figures/milton_F3_cascading_ifs_ens.pdf (1.07 MB vector). Paper main.tex:1137 now .pdf.
  F3 is all-PDF.

## Pushes (explicit user instruction, overrode the standing no-push default)
- swiss-ai/SwissClim_Evaluations research ff609b9 (with Claude trailer).
- MeteoSwiss/ai-models-ensembles main fb6e6cf (with trailer).
- sadamov/ai-models-ensembles (paper) main ff4e5a5.

## Left UNCOMMITTED on purpose
(paper-process detail kept in the local notes)
