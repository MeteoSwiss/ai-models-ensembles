# 2026-09-02 IFS-ENS Milton detection rate: the 14% "correction" was itself the bug (71% was right)

One-line: the 2026-09-01 "correction" of the IFS-ENS Milton detection rate from 71% to 14% was caused
by a stale double member filter in `aggregate_tracks.py` that silently kept only 2 of the 10 tracked
members; the original 71 / 317 / 173 / +36.6 / 252 row is exact and fully reproducible, and a separate,
real WeatherBench-2 data defect (~20% of IFS-ENS fields stored as global NaN) suppresses it further to
a true ~80% when repaired.

## The bug

`tools/milton/track_one_init.py` selects the stratified IFS-ENS 10-member subset up front
(`ds.isel(ensemble=[0,5,...,45])`) and then labels the survivors `member = 0..9`, exactly like every
other baseline. `tools/milton/aggregate_tracks.py:load_master()` then re-applied the *original*
stratification to those already-relabelled ids:

```python
stratified_ifs_ens = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45}
df = df[(df.baseline != "ifs_ens") | (df.member.isin(stratified_ifs_ens))]
```

On members 0..9 that set matches only `{0, 5}`, so 80% of the IFS-ENS tracks were dropped, while the
denominator stayed at 140. The filter was correct when the tracker still ran all 50 members with native
ids; moving the subsetting upstream made it a double filter and nobody removed it.

Arithmetic, straight off the archived per-init CSVs (`tracks_archive/ifs_ens_nogapfill_2026-09-02/`,
written by the 08-31 regeneration - i.e. the very run the 09-01 audit called "fresh"):

| counted over | (init, member) cells with a Milton track |
| --- | --- |
| all 10 members | **99 / 140 = 71%** |
| members 0 and 5 only | **19 / 140 = 14%**  <- what the audit measured |

The filter is gated on `baseline == "ifs_ens"`, so no ML row was ever affected - which is why every
other row reproduced and only this one looked anomalous.

## Why the 09-01 audit concluded "untraceable"

It re-derived the number through `aggregate_tracks.py` (carrying the bug) and compared against the
raw log of the *original* job 2473058. That job predates the `--maxgap 6h -> 18h` change, so its 12.6%
(88/700) is a genuine but since-fixed artefact of the tighter gap bridge, not independent confirmation
of 14%. Two corrupted-in-the-same-way numbers agreeing was read as reproducibility. The June 2026
"13%" in `[[ifs-ens-10m-wind-nan-bug]]` is the same pre-18h-maxgap regime.

Running the pipeline on the archived tracks with the filter removed returns the original row to four
significant figures: detection 71%, mean IBTrACS position error 316.5 km, median 172.7 km, MSL bias
+36.55 hPa, ERA5-axis position error 251.6 km - i.e. the published 71 / 317 / 173 / +36.6 / 252.

## The real, separate WB2 data defect

Independently confirmed while chasing this, and *not* what the 09-01 audit tested. In
`/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr` (downloaded whole-variable from
`gs://weatherbench2/datasets/ifs_ens/2016-2024-1440x721.zarr` by `IFS/tools/download_ifs_ens_wb2.py`,
144 dask workers), ~20% of `(variable, init, lead)` inner chunks hold a **globally NaN field**:

- Verified two ways: direct reads (lead 30 h `10m_u` is 100% NaN over the whole globe and all 50
  members while `mean_sea_level_pressure` at the same lead is complete; lead 96 h the reverse), and
  cheaply for every init from the zarr v3 shard indexes (all chunks are *written*, but the NaN ones
  compress to <2% of the median chunk size).
- Rate is 18-21% for **every** variable, randomly placed, **independent per variable and per init**.
  The union over the six fields the tracker needs (msl, z500, z850, u10, v10, u850/v850) leaves only
  **353 / 854 = 41%** of Milton leads usable, vs ~100% for every ML baseline.
- Not the store migration: a122 and ab016 missing-lead sets are identical across 42 (variable, init)
  cells. Not a physical archive property either - the companion `ifs_control.zarr` from a different
  pipeline is 0.0% NaN, and no ECMWF archive drops `10m_u` globally at step 30 h while keeping MSL.
  Most likely a lossy download; **not yet checked against the GCS source** (no network from this
  session; needs a compute node with `SSL_CERT_FILE` set).

Mechanism, which the 09-01 note got wrong ("gaps are short and already bridged"): the 18 h `--maxgap`
bridges gaps *between candidates*, but a lead with NaN msl or geopotential yields **no candidate at
all**, and 34% of the candidates that do survive carry `v10 = nan`, which makes the StitchNodes
`--threshold v10,>=,17.0,3` unsatisfiable - and StitchNodes then discards the **whole track**. Proof:
re-running StitchNodes on the untouched archived `detect.txt` files with only that threshold removed
takes IFS-ENS from 121/140 to 140/140 files with a track and to **126/140 = 90%** Milton detections,
while the ML baselines move only 10-29 points (95-100%).

## The repair and the corrected numbers

`_fill_missing_leads()` added to `track_one_init.py` (ifs_ens only): whole-field NaN leads are
reconstructed by linear interpolation along `lead_time`. Justified by the gap-length distribution -
82.8% of gaps are a single 6 h step, 97.3% are <= 12 h (n=953 gaps over 14 inits x 7 field-series).
Reconstructed leads are flagged (`interpolated` column, carried through `aggregate_tracks.py`) so
position/intensity stats can exclude them; doing so shifts almost nothing (pos err 326 -> 336 km,
MSL bias +33.6 -> +32.4 hPa), which is the check that the interpolation is not driving the result.

| IFS-ENS variant | detection | pos err IBTrACS | median | MSL bias | pos err ERA5 |
| --- | --: | --: | --: | --: | --: |
| **as published pre-09-01 (correct)** | **71%** | 317 km | 173 km | +36.6 hPa | 252 km |
| WB2 dropout repaired (gap-filled) | **80%** | 326 km | 173 km | +33.6 hPa | 271 km |
| 09-01 "correction" (2 members) | 14% | 348 km | 169 km | +40.2 hPa | 287 km |

Under both correct variants IFS-ENS has the **lowest MSL bias of any baseline** (+33.6 / +36.6 vs
+37.2 to +44.2), so the claim the 09-01 session deleted from `main.tex` sec.~4.5 as "false" was true;
it only looked false on the 2-member sample.

## Fixed in code

- `tools/milton/aggregate_tracks.py` - stale double member filter removed; `interpolated` carried into
  the verification frame.
- `tools/milton/analyze_milton_stats.py` - stale comments about the 50-member denominator corrected.
- `tools/milton/track_one_init.py` - `_fill_missing_leads()` + `TRACKED_FIELDS`; the ifs_ens load is
  restricted to the six tracked fields and the Milton box before `.load()`.

Tracks on scratch: `tracks/ifs_ens` = archived non-gap-filled (matches the paper);
`tracks_archive/ifs_ens_gapfill` = gap-filled; `tracks_archive/ifs_ens_nogapfill_2026-09-02` = the
untouched 08-31 originals.

## Still open

- Paper not yet touched: `figures/milton_summary_table.tex` (+ vendored copy) and `main.tex` sec.~4.5
  still carry 14% and the removed lowest-MSL-bias sentence. Awaiting the user's call on 71% vs 80%.
- Verify the ~20% NaN against the GCS source, and re-download the affected chunks if the source is
  clean. Everything else in the paper that uses IFS-ENS runs through skipna aggregation
  (`main.tex` already states the 18-20% NaN caveat), so only the tracker was structurally exposed.

## Related

- `2026-09-01_milton_pipeline_regen_ifsens_detection_fix.md` - the entry this one supersedes.
- `[[ifs-ens-10m-wind-nan-bug]]` - the NaN property, correctly identified in June, wrongly cleared as
  an explanation on 09-01, and quantified properly here.
