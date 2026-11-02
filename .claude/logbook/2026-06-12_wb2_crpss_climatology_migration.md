---
name: wb2-crpss-climatology-migration
description: CRPSS denominator migrated from analytical-Gaussian to exact WB2 probabilistic climatology (2026-06-12); new tool + denominators + all paper numbers updated
metadata:
  node_type: memory
  type: project
  originSessionId: 9e1561ce-5130-4cbc-b5e6-f35b39f773a4
---

2026-06-12: replaced the analytical-Gaussian CRPSS denominator
(`CRPS_clim = sigma_clim/sqrt(pi)`) EVERYWHERE with the exact WB2/ECMWF
probabilistic climatology. The Gaussian was a temporary workaround; it used the
UNCONDITIONAL per-pixel std (annual cycle included), which inflated the
denominator and the CRPSS. The user asked to follow the exact ECMWF procedure.

**Method (verified vs Rasp2024 WB2, arXiv:2308.15560):** the 30 years 1990-2019
of ERA5 at each forecast valid time's calendar (month, day, hour) form a
30-member ensemble (no smoothing), scored with the SAME fair (M-1) CRPS against
the actual 2023/2024 eval truth, cos(lat)-weighted, resolved PER LEAD, over the
same inits as the numerator. NOT leave-one-out within the reference period.
Forecast CRPS was already WB2-exact (weatherbenchX `CRPSEnsemble(fair=True)`).

**Tool:** `tools/compute_climatology_crps_vs_eval.py` (`--grid production|ablation`),
`tools/submit_compute_climatology_vs_eval.sh [grid]`. Jobs 2522467 (production,
2h12m) + 2522480 (ablation, 1h2m) COMPLETED 2026-06-12. Slow on 3D wind/humidity
fields (I/O on the hourly clim zarr); ~43 GB/field, --parallel 2.

**Denominators (canonical, version-controlled):**
`tools/data/crps_clim_eval_1990_2019.json` (112-init headline) and
`tools/data/crps_clim_eval_ablation_1990_2019.json` (4-init ablation/calibration),
keyed `{var_or_var_level: {lead_str: CRPS_clim}}`. Scratch copies + `_provenance`
in `/iopsstor/scratch/cscs/sadamov/`.

**SUPERSEDED / dead for the paper:** `compute_climatology_1990_2019.py` (Gaussian
sigma + LOO empirical), `sigma_clim_1990_2019.json`, `sigma_clim_ablation.json`,
`diff_climatology_recompute.py`. No `sigma/sqrt(pi)` remains in any LIVE CRPSS
tool (headline_8way/7way tables, plot_headline_crpss_vs_lead{,_8way},
compute_perturbation_uplift all rewired to `clim_for(var,lvl,lead)`).
`tools/calibration_crpss.py` reproducibly recomputes the Tab. 3 CRPSS columns.

**Impact (all paper numbers updated in `main.tex` + regenerated
`figures/headline_8way_table.tex` + `headline_crpss_vs_lead_8way.pdf`):**
- Absolute CRPSS ~halved. Headline gap vs AIFS-ENS 0.07-0.10 -> 0.09-0.13 @240h,
  within 0.04 -> within 0.06 @360h. Per-model @240h: AIFS-ENS 0.256, graphcast
  0.167, aurora 0.134, sfno 0.125.
- Ablation picks @240h: 0.317/0.343/0.306 -> 0.151/0.190/0.145. ALL FOUR
  production picks remain the within-model argmax CRPSS@240h (selection intact).
  Only Tab.3 bold change: razor-thin SFNO 72/120h flip Phase1->Phase2.
- Uplift (Tab): SFNO Delta +0.59 -> +0.73 (still largest); aurora +0.67, gc +0.55.
- Phase 6c sfno 0.303 -> 0.143 (delta -0.0017, "within 0.002"/"no measurable
  cost" still hold); Phase 6d drop 0.06-0.10 -> 0.08-0.12; sigma=1.93 fresh
  -1.93 -> -3.04.
- Narrative shifts: best-calibrated var now geopotential (was T850); Aurora 2m_t
  collapsed to 0.022 (its old skill was mostly the seasonal cycle the conditional
  clim removes); Milton Phase-5 footnote "<=0.02 at any lead" was false at short
  lead (0.10 @6h) -> narrowed to "<=0.01 at 240h".
- Methods def + framing rewritten (Hamill2001 Gaussian -> Rasp2023wb2 prob clim).

Persistence (`compute_persistence_mae.py`) audited: already correct + WB2-consistent
(init state held constant, MAE = deterministic CRPS, cos-lat, same grid/truth);
no change, just re-derives against the new denominator in the figure.

Committed 2026-06-12 (main repo 5d08b70, paper repo ee7f72c); after the 2026-06-13
nightly-branch retirement both are on `main`. Supersedes the Gaussian parts of
[[crpss-migration]] and the sigma_clim notes in [[metric-audit]] /
[[headline-7way-crpss]].

(Migrated 2026-06-13 from the auto-memory dir into the repo logbook, per the
auto-memory-is-pointer-only policy.)
