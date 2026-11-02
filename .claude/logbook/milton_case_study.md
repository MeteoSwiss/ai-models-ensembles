---
name: milton-case-study
description: "2026-06-03 evening: Hurricane Milton (October 2024) case study setup -- TempestExtremes tracker via tctrack + conda-forge binaries, Walsh 2007 thickness-anomaly warm-core check (because forecasts only have 500+850 hPa), ERA5 validation (22 km mean position error, +40 hPa MSL bias is reanalysis resolution-limit not tracker error), SLURM batch 2465829 tracking all 6 baselines x 14 inits x 10 members."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why Milton (October 2024)

User's choice. Analogue to Doksuri (2023) used by ESFM Ozdemir et al. 2026 (arXiv:2605.00850); same-direction story (rapid-intensification tropical cyclone + landfall) but in our 112-init production grid week 2024-10-02 to 08, so zero new compute needed for the 7-way baseline comparison.

Milton: 24-30 September - 11 October 2024. Peak intensity 2024-10-07 18 UTC at 895 hPa / 155 kt (Cat 5). Genesis in Gulf of Mexico ~10-04 18 UTC. Landfall near Siesta Key, FL ~10-09 21 UTC. Source: IBTrACS v4 NA basin (loaded via huracanpy).

All 14 production inits 20241002_0000 to 20241008_1200 cover Milton's lifecycle at progressively shorter leads, supporting the **cascading-detection** "how early did each model see RI?" analysis.

## Variable constraint

Forecast.zarrs have only 500 + 850 hPa levels (production inference saves `--output-levels '500,850'`). ERA5 truth (WB2) has full 13 levels.

This rules out canonical TempestExtremes warm-core check (Z300-Z500 anomaly per Ullrich & Zarzycki 2017). Substituted with **Walsh 2007 (J. Climate)** lower-tropospheric thickness anomaly: Z(500)-Z(850) >= +58.8 m^2/s^2 (~5 m thickness) within 2 deg of SLP-min candidate. Bourdin et al. 2022 (GMD) intercomparison confirms ~80% detection rate baseline at ERA5 resolution.

Other consequence: forecasts can't represent Cat 5 intensity. ERA5 itself shows only ~989 hPa peak vs IBTrACS 895 hPa -- the 0.25 deg grid physically can't resolve a 25 km eye. So forecast intensity is compared against ERA5 (apples-to-apples), with IBTrACS as ultimate truth + resolution caveat (Knaff et al. 2019 WAF context).

## Tracker setup (verified 2026-06-03)

Stack:
- **huracanpy** v1.4.0 (pip) -- IBTrACS loader + xarray-native TC analysis (Lawton 2024 JOSS)
- **TempestExtremes** v2.1 (`DetectNodes` + `StitchNodes`) via conda-forge `tempest-extremes` -- Ullrich & Zarzycki 2017 GMD; Ullrich et al. 2021 GMD v2.1
- **tctrack** v0.2 Python wrapper (cf-python / cfdm / cfunits dependency chain, requires conda udunits2 + UDUNITS2_XML_PATH env var; we mostly bypass tctrack and call DetectNodes/StitchNodes directly via subprocess for speed)
- udunits2 + libudunits2 via miniforge3 conda base env

Uses the shared CAPSTOR venv `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python` plus the capstor miniforge3 for DetectNodes/StitchNodes and udunits. Env setup: `tools/milton/env.sh` (LD_LIBRARY_PATH + UDUNITS2_XML_PATH + PATH). The former dedicated venv on iopsstor scratch (`venvs/milton_case`) was destroyed by a scratch purge and deleted 2026-08-30 - never put a venv on scratch.

## Tracker parameters with citation per knob

```
DetectNodes:
  --searchbymin      psl                                          # SLP min seed (UZ17)
  --closedcontourcmd psl,200.0,5.5,0                              # SLP rises >=2 hPa within 5.5 deg (UZ17)
  --closedcontourcmd z_thick_500m850,-58.8,6.5,1.0                # Walsh 2007 thickness +5 m
  --mergedist        6.0                                          # merge adjacent candidates (UZ17)
  --latname latitude --lonname longitude --regional               # regional subset

StitchNodes:
  --range            5.0                                          # max 5 deg / 6 h (UZ17)
  --mintime          36h                                          # Walsh 2007 lifetime
  --maxgap           6h
  --threshold        v10,>=,17.0,3                                # NHC tropical-storm-force
```

Citation chain: Ullrich & Zarzycki 2017 (algorithm), Ullrich et al. 2021 (v2.1 docs), Walsh et al. 2007 (lower-tropospheric warm-core substitution), Knapp et al. 2010 (IBTrACS truth), Marks & Houze 1987 + Rogers et al. 2013 (storm-relative composites), Knaff et al. 2019 (TC structure verification), Bengtsson et al. 1995 (early TC tracking foundations).

All 10 entries added to `paper_refs.bib` (now 50 hand-curated entries) AND to the project Zotero collection (Post-hoc Weight Perturbation 2026) with tags `milton-case-study` + `method`.

## ERA5 validation (the green light)

Run on WB2 truth zarr 2024-10-04 to 2024-10-12, +-7 day window around Milton, regional subset 240-300 E / 5-40 N:
- **Mean position error vs IBTrACS: 22 km** across 19 matched 6-hourly fixes (excellent; IBTrACS own uncertainty ~15 km)
- **Mean MSL bias vs IBTrACS: +40.3 hPa** -- this is the ERA5 grid limit, NOT tracker error. At Milton's Cat 5 peak ERA5 shows 989 hPa where IBTrACS says 895 hPa (94 hPa gap is the 0.25 deg grid intensity ceiling)

Validation plots committed in `fd615fe`:
- [figures/milton_aifsens_m0_sanity.png](../../../../pyprojects/ai-models-ensembles/figures/milton_aifsens_m0_sanity.png)
- [figures/milton_era5_control.png](../../../../pyprojects/ai-models-ensembles/figures/milton_era5_control.png)

## First model result (1 task in)

Aurora_encoder, init 2024-10-02 00 UTC, lead 0 to 240 h:
- 9 of 10 members miss Milton entirely (no track found)
- 1 of 10 (member 5) produces a weak track with min MSL 990 hPa

This is the headline ensemble-spread story playing out: at 8-day lead, perturbed Aurora rarely sees Milton at all.

## SLURM batch in flight: 2465829

6 baselines x 14 inits = 84 array tasks. Lean resources (20G mem, 4 CPU, 15 min walltime, %20 throttle). Wall-time per task ~ 4 min wall (per task 1). Started 2026-06-03 ~20:30 CEST. ETA ~3-4 h.

Per-init output: `/iopsstor/scratch/cscs/sadamov/milton_case_study/tracks/<baseline>/<init_tag>/milton_tracks.csv` (one row per member per timestep).

ifs_ens NOT in this batch -- needs separate pipeline because (a) it's classical NWP with different zarr layout; (b) we subsample 10 of 50 members; (c) reshard pass 3 still running (2462505).

## Scripts

All on scratch (not in repo because they're case-study specific, not paper infra):
- `track_one_init.py` -- per-task tracker (one baseline x init x 10 members)
- `era5_control.py` -- the validation control
- `sanity_plot.py` -- AIFS-CRPS m0 verification figure
- `aggregate_tracks.py` -- concat all CSVs + NHC verification (track + intensity vs both truths)
- `figures_milton.py` -- F1 track spaghetti / F2 intensity vs lead / F5 verification vs lead (figures F3/F4 still to add: cascading detection + storm-relative composites)
- `add_milton_refs.py` -- the bib + Zotero sync (already executed)

## Pending paper figures

1. **F1 track spaghetti**: 3 init rows x 6 baseline cols, 10 members each + IBTrACS truth. Eulerian, paper-ready.
2. **F2 intensity vs lead**: min MSL per member per init, faceted by baseline. ERA5 + IBTrACS reference lines.
3. **F3 cascading detection**: ensemble-mean MSL at fixed Oct 7 18 UTC peak time, 14-panel grid (rows = init day, cols = init UTC), one baseline per figure. The "how early did the ensemble see RI" diagram.
4. **F4 storm-relative composites**: Cartesian visualisation (recentered MSL + 10 m wind + 850 q + 500 Z), one panel per baseline at peak intensity time.
5. **F5 verification vs lead**: track position error + MSL bias vs lead, one curve per baseline, both truths.

F1 / F2 / F5 scaffold written. F3 + F4 still to scaffold.

## Status 2026-06-04 morning

All 5 paper figures generated + committed (commit `a881769`):
- `figures/milton_F1_track_spaghetti.png` -- 3 inits x 6 baselines, 10 members each
- `figures/milton_F2_intensity_vs_lead.png` -- min MSL vs lead, faceted by baseline
- `figures/milton_F3_cascading_aifsens.png` + `_aurora_encoder.png` -- 14-panel grid, ensemble-mean MSL at 2024-10-09 12 UTC (landfall window), one per init
- `figures/milton_F4_storm_relative_composite.png` -- 6 baselines, +-10 deg storm-relative MSL composite at lead 120 h init 2024-10-04 00 UTC. Per-baseline minimum 993.7-999.0 hPa (atlas-graphcast), confirming ~990 hPa ERA5 grid-resolution ceiling.
- `figures/milton_F5_verification_vs_lead.png` -- track position error vs IBTrACS + MSL bias vs ERA5

Detection rate per baseline (from `aggregate_tracks.py` master CSV, 140 cells = 14 inits x 10 members each): aifsens 85.7%, atlas 87.9%, fcn3 84.3%, sfno 77.9%, aurora 73.6%, graphcast 52.1%. graphcast under-detects because its Phase 1 production sigma=0.01 is the smallest perturbation magnitude across baselines, so members converge most tightly and many miss Milton.

Aggregator outputs: `/iopsstor/scratch/cscs/sadamov/milton_case_study/milton_master_tracks.csv` (10954 track rows) + `milton_verification.csv` (same row count, joined with IBTrACS + ERA5 truth).

## Paper section TODO

Add to main.tex after the calibration results: 1-2 paragraph qualitative discussion + the 5 figures. Conservative interpretation per user note ("we'll probably redo with perturbed ICs anyway"). Highlight: ensemble-spread coverage of RI events, member-by-member detection rate, position-vs-intensity-error trade-off across baselines, methodological caveat re ERA5 grid intensity ceiling.

## Pending follow-up (added 2026-06-04): geostrophic wind balance bivariate

User wants a sixth Milton-domain figure: the geostrophic wind-balance bivariate histogram from the SwissClim `multivariate` eval module.

**Why:** the multivariate module already produces 2D density plots; reusing that infrastructure gives a clean physics-consistency check over the Milton domain (geostrophic balance: f*v_g = +dphi/dx, f*u_g = -dphi/dy, so actual wind vs |grad phi|/|f| should hug the 1:1 line away from the eye). Hurricanes are the classic geostrophic-breakdown case (cyclostrophic / gradient-wind regime in the eyewall), so the departures themselves are the story. This is a natural complement to the storm-relative composite (F4) and lets us claim physical-consistency evaluation, not just track position/intensity.

**How to apply when running:**
- Run SwissClim multivariate eval with `--region milton_domain` (the same +-10 deg box used for F4) and the pair (u_wind_850, geopotential_500) or (wind_speed_850, |grad_phi_500|/|f|). Pick whichever pair the multivariate module already supports natively.
- Output goes into the standard `multivariate/bivariate_*.{pdf,png}` slot; copy into `figures/milton_F6_geostrophic_balance.{pdf,png}` after the run.
- One panel per baseline at a fixed valid time (e.g. 2024-10-09 12 UTC landfall window) -- 6 baselines side-by-side, plus an ERA5 truth panel.
- Reference line: 1:1 diagonal (perfect geostrophic balance).
- Domain restriction matters: keep it to the Milton +-10 deg box so the eyewall ageostrophic signal dominates; over the whole globe the curve degenerates to the synoptic-scale balance everyone already has.

**Status:** not started -- recording for the next Milton-case-study session.

## Related

- [[paper-folder-inventory]] -- where the .tex + .bib live (`ai-models-ensembles-paper/` sibling clone)
- [[reference-scoring-rule-jargon]] -- CRPS / fair-CRPS definitions for the case-study verification
- [[spatial-mean-vs-pointwise-ssr]] -- the wider paper narrative this case-study illustrates concretely
