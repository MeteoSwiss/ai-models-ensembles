---
name: bivariate-colorbar-isoline-rerun
description: "Pre-submission reminder: re-run bivariate histogram plots in any model variant we cite in the paper, so the regenerated images carry the greyscale truth-density isoline marks on the shared horizontal colorbar (fix landed 2026-06-10 in SwissClim_Evaluations)."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Bivariate histogram colorbar isoline fix

## What was missing

The shared horizontal colorbar at the bottom of the multi-panel bivariate histogram grids was a bare colour gradient with no level cues. The per-panel single-eval colorbar carries the greyscale truth-density isoline marks (`cbar.add_lines(cs1)` in `plot_bivariate_histogram` at `plots/bivariate_histograms.py:1255`). The intercomparison-bivariate grid already had `cbar.add_lines(shared_target_cs)` (`intercomparison/modules/multivariate.py:323`). The per-eval multi-panel grid `_plot_bivariate_per_lead_grid` (`plots/bivariate_histograms.py:452`, shared cbar at line ~690) DID NOT.

## What was added (2026-06-10)

In `_plot_bivariate_per_lead_grid`:
- `shared_target_cs = None` before the lead-loop
- `return_contour_sets=True` passed to `plot_bivariate_histogram`
- `if shared_target_cs is None and isinstance(_result, tuple): shared_target_cs = _result[2]` after each panel
- `if shared_target_cs is not None: cbar.add_lines(shared_target_cs)` after the shared colorbar is built

NOT YET COMMITTED at the time of writing.

## Pre-submission action

Before paper submission, re-run these for the model variants we actually CITE in the paper (the bivariate joint distribution + the intercomp-bivariate figure already in Sec 4.4):
1. `bash scripts/evaluate_baselines.sh multivariate <model>` -- regen per-baseline bivariate PNGs
2. `bash scripts/evaluate_baselines.sh intercomparison_multivariate ...` -- regen the intercomp bivariate PNGs

Anything we DON'T cite doesn't need regeneration; the fix only affects rendered PNGs/PDFs going forward.

## Why grey isoline marks on the colorbar matter

The greyscale truncated-Greys colormap maps log-density to grey shade exactly. Without the isolines marked on the colorbar, the reader has to guess where each grey contour line on the panel falls on the density scale. With the marks, the reader can read the truth-density level for each isoline directly off the colorbar -- the same way the prediction colour at each colour fill level is read off the same colorbar.

Restoring symmetry between prediction (filled colour) and truth (isoline) on the colorbar key is a small but important readability fix for the bivariate joint-distribution figure.
