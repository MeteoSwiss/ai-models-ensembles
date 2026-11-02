---
name: paper-add-climatology-persistence
description: "Pre-submission task: add climatology and persistence as reference baselines to the paper figures. Fig. 1 (headline 8-way CRPSS) DONE 2026-06-04; Fig. 2 ablation panels, tier1b SSR, Phase 6 SSR, Fig. 4 bivariate still pending."
metadata:
  node_type: memory
  type: project
  originSessionId: 06ea53b7-020e-48a1-b173-7a0c0fcadfe2
---

Before submission, add **climatology** and **persistence** as two trivial reference baselines to most of the paper's per-lead figures. Standard NWP-verification practice; it grounds the absolute scale that CRPSS alone partly hides.

**Why:** the paper currently uses CRPSS (which normalises against a Gaussian climatology), but no figure shows climatology or persistence as a visible curve. Readers expect both as floor and ceiling references; reviewers will likely ask.

**Definitions to use:**
- **Climatology forecast:** use whichever climatology is current at the time the reference curves are drawn. **As of 2026-06-03 evening, the paper still uses the WB2 2022-2024 Gaussian climatology** (`sigma_clim_ablation.json`). If [[paper-revision-round-todos]] #9 + #10 (WB2 1990-2019 empirical climatology recompute) is executed before this task, use the 1990-2019 empirical climatology instead. The climatology forecast itself is then the climatological ensemble (30 years of per-(DOY, hour) values per pixel). As an "ensemble" with this many members it has CRPS = sigma_clim / sqrt(pi) for the Gaussian case (constant in lead), or the empirical leave-one-out fair-CRPS for the empirical case (also approximately constant in lead). Either way CRPSS = 0 (by construction), spread = sigma_clim, RMSE of mean = sigma_clim, so SSR = 1.
- **Persistence forecast:** forecast(t+h) = analysis(t=0). Deterministic (M=1), so CRPS = MAE, CRPSS rises monotonically from 1 at t=0 to ~0 at long lead. SSR = 0 (no spread).

**Figures that need the addition:**
- ✅ Fig. 1 DONE 2026-06-04, now [figures/headline_crpss_vs_lead_8way.pdf](../../../../pyprojects/ai-models-ensembles/figures/headline_crpss_vs_lead_8way.pdf) (Sec 4.2): climatology drawn as dashed-black axhline at CRPSS=0 (labelled "Climatology"); persistence drawn as dotted-black curve (labelled "Persistence") loaded from the JSON below. Tool `tools/plot_headline_crpss_vs_lead_8way.py` updated; ylim extended to (-1.6, 1.0) so the persistence dive at long lead fits.
- Fig. 2 (Sec 4.1 per-model ablation CRPS panels): add climatology CRPS = sigma_clim / sqrt(pi) (horizontal dashed) and persistence MAE vs lead (thin line) to each panel.
- Fig. (tier1b, [figures/tier1b_7way_spatial_mean_ssr.pdf](../../../../pyprojects/ai-models-ensembles/figures/tier1b_7way_spatial_mean_ssr.pdf)) and Phase 6 SSR panels: add SSR=1 reference (already implicit but draw explicitly) and persistence SSR=0 floor.
- Fig. 4 (bivariate joint distribution): overlay climatology joint density (thin black contour) so reviewer can read climatology + truth + each baseline in one panel.
- Spectrogram delta (Fig. 3): probably not -- the metric is forecast-truth spectral difference; climatology / persistence don't add a useful reference there. Skip.

### Persistence source (compute_persistence_mae.py)

Persistence MAE was computed empirically (not modelled analytically), so the reference curve reflects the actual ERA5 structure function:
- Script: `tools/compute_persistence_mae.py` (committed 2026-06-04 alongside the figure update).
- Truth source: `/capstor/store/cscs/swissai/a122/ERA5-2022-2025.zarr` (same source SwissClim eval uses).
- Sample: 20 init dates from 2024 weekly production grid (Wed 00 UTC), `random.seed(42)`.
- Leads: 6 h .. 360 h in 6 h steps.
- Variables: same 6-paper set (2m_temperature, geopotential, temperature, u/v wind, q at 500+850).
- Output: `/iopsstor/scratch/cscs/sadamov/persistence_mae_2024.json` (15 K, JSON keyed `var` or `var_lvl` -> {`lead_h`: persistence_MAE}).
- CRPSS conversion in the plot script uses the same `sigma_clim_ablation.json` denominator (= sigma / sqrt(pi)) as the model curves -- apples-to-apples.
- Rerun cost: ~5 min wallclock on a login node.

**How to apply (for the remaining figures):**
- Climatology baseline: build a synthetic "model" entry from `sigma_clim_ablation.json` + WB2 mean; add a row to `temporal_metrics_combined.csv` so downstream plotting picks it up without code changes.
- Persistence baseline: for figures off the 7/8-way CRPSS axis, either reuse the JSON above (CRPSS / CRPS) or extend `tools/compute_persistence_mae.py` to write SSR (=0) / bivariate stats. Either way the persistence forecast itself is identical across baselines because they all start from the same ERA5 IC.
- Then re-run the intercomp plot modules (or the in-house plot scripts) -- they consume the CSV.

**Visualisation rule (matches existing palette in `tools/plot_headline_crpss_vs_lead.py`):**
- climatology: thin **black** dashed line
- persistence: thin **black** dotted line
- both labelled in the legend below the other 7

**Scope marker:** do this in the same pre-submission pass as the [[paper-presubmission-pdf-regen]] step, since it touches the same plot regeneration pipeline. Don't bother during drafting -- it's only needed once the paper is locked.

**When this triggers:** user signals "ready to submit", "freeze figures", "camera-ready", or asks for a pre-submission checklist.
