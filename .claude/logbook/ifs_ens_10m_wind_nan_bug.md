---
name: ifs-ens-10m-wind-nan-bug
description: "2026-06-06 final verdict: WB2 IFS-ENS dataset is upstream-dirty. ALL 5 2D surface vars (MSL, 2m_t, 10m_u/v_wind, 10m_wind_speed) have 10-30% NaN per (init, var) cell. Verified byte-identical NaN counts across our local source, our resharded copy, and the public GCS bucket gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr. This is NOT a reshard or download artefact -- the missing values are an ECMWF/WB2 dataset property. User had previously checked and remembered the ~20% missing. Dominant cause of the Milton 13% IFS-ENS detection rate."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## What we know

Confirmed 2026-06-06 by direct inspection of the resharded zarr at
`/capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr` for the Milton init
2024-10-04 00 UTC over the Milton bounding box (lat 8 to 40, lon 258 to 305):

| Variable | NaN fraction (Milton box, all leads + members) |
| --- | --: |
| `10m_u_component_of_wind` | 19.7\% |
| `10m_v_component_of_wind` | 21.3\% |

Lead-time breakdown: 12 of 61 6-hourly leads have **100% NaN** for 10m winds across all 50 members and the Milton box. The affected leads are
`[30, 114, 120, 174, 186, 204, 240, 252, 258, 282, 324, 330]` h.

The pattern is irregular -- not every Nth lead, not a contiguous block. This rules out a simple per-week or per-day skip.

## Why it matters

The 13\% Milton detection rate for IFS-ENS (Sec 4.6 of the paper) is almost entirely an artefact of this missing data. The TempestExtremes StitchNodes filter requires `v10 >= 17 m/s` for 3+ consecutive non-missing points; with 12/61 leads entirely NaN, those chain conditions are impossible to satisfy. The non-NaN IFS-ENS winds are also weaker than the ML baselines (3% of candidates cross 17 m/s vs 12% for aifsens), but the missing-leads effect dominates.

Earlier draft prose in Sec 4.6 attributed the low rate to "spectrally smoother IFS-ENS members" -- this was wrong and has been corrected. IFS-ENS actually has a LOWER MSL bias (+24.6 hPa) than the ML cluster (+39 to +47 hPa), meaning it resolves sharper pressure gradients, the opposite of "smoother".

## Verified provenance (2026-06-06)

Side-by-side NaN fractions at three (init, var) cells across three independent sources:

| init | var | local source<br>`a122/IFS/ifs_ens.zarr` | local reshard<br>`mch/s83/IFS/ifs_ens.zarr` | GCS WB2<br>`weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr` |
| --- | --- | ---: | ---: | ---: |
| 2023-01-02 00 | MSL | 18.03% | 18.03% | **18.03%** |
| 2023-01-02 00 | 2m_t | 22.95% | 22.95% | **22.95%** |
| 2023-01-02 00 | 10m_u | 19.67% | 19.67% | **19.67%** |
| 2023-01-04 00 | 2m_t | 27.87% | 27.87% | **27.87%** |
| 2023-01-04 00 | 10m_u | 29.51% | 29.51% | **29.51%** |
| 2024-10-04 00 | MSL | 13.11% | 13.11% | **13.11%** |

All three sources byte-identical at every cell sampled. The NaN is in the upstream WB2 IFS-ENS dataset itself.

Per-lead inspection at init 2024-10-04 00 UTC: 12 of 61 leads (~20%) have 100% NaN. Affected leads at `[30, 114, 120, 174, 186, 204, 240, 252, 258, 282, 324, 330]` h, irregularly distributed. Per-ensemble: identical 18.03% NaN across all 50 members for MSL at the 2023-01-02 init -- the NaN mask is independent of ensemble member.

## What we tried before drawing this conclusion

- 2D var audit on the resharded zarr: 10-30% NaN per (init, var) for all 5 2D vars sampled across 9 inits.
- Source-vs-reshard byte-identical comparison: byte-identical NaN counts (the reshard preserves the bug).
- GCS WB2 comparison: byte-identical NaN counts (the bug is upstream of our download).

This is consistent with the user's prior memory of having checked once before and seeing the ~20% missing.

## How to apply / future actions

- Sec 4.6 prose corrected 2026-06-06 to state explicitly that the missing data is an upstream WB2 / ECMWF property, not anything our pipeline introduced. The 13% IFS-ENS Milton detection rate is therefore not a forecast-quality statement.
- We cannot repair this upstream of our paper -- it would need an ECMWF reissue of the WB2 IFS-ENS dataset. Until that happens, all paper evaluations that use IFS-ENS share the same NaN-affected denominator. Variable-mean CRPSS in the headline 8-way handles this gracefully (NaN-aware aggregations); per-lead, per-variable, per-event tracker analyses (like Milton) are more sensitive.
- The wider headline / tier1b / ES-VS / bivariate evaluations in the paper are all aware of NaN via skipnan defaults in the SwissClim verification stack. The numbers shown in those tables and figures are computed on the ~80% non-NaN subset, not the full grid; this is the standard WB2 evaluation convention.

## Related

- [[wb2-truth-zarr-broken-msl-uwind]] -- the truth-zarr-side variant of the same general class of bug (broken metadata for MSL and u_component_of_wind in WB2 consolidated mode). The IFS-ENS-side bug is independent but rhymes.
- [[ifs-zarr-reshard]] -- the reshard pipeline state (where the NaN may have been introduced).
- [[milton-case-study]] -- the case study where the consequence shows up.
- [[ifs-ens-msl-bug]] -- earlier MSL bug at 14 July 2024 inits (separate, already patched to NaN in-place).
