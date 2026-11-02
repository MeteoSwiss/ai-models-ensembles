---
name: aifs-predownload-cache
description: "2026-06-02 morning: built a parallel CDS pre-download script that writes ERA5 fields directly into earth2studio's on-disk grib cache (sha256-keyed) so AIFS inference becomes compute-bound instead of CDS-API-bound. Resolves the 1.2h-per-init bottleneck. Script + cache-key match verified."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Bottleneck root cause

aifs_perturbed jobs took 1.2h per init. Originally I (Claude) blamed parallel-jobs-throttling-CDS; that was wrong. The real cause is per-init **prefetch**: the inference driver calls `source(fetch_times, variables)` to warm cache for both T-6h and T0 of one init. earth2studio's CDS class issues those requests **sequentially** (the multiprocessing pool is commented out in the upstream code). 18 unique (var, time) pairs × 2 timestamps = ~36 sequential CDS requests at ~60 s each (CDS queue) = ~36 min just to warm the IC cache, before any inference happens. The remaining ~30-40 min per init is the actual 10-member 4-GPU rollout.

The cache mount we added 2026-06-01 (`/iopsstor/scratch/cscs/sadamov/e2s_cache:/workspace/.cache/earth2studio`) DOES work -- the 10 members in a job share cache (only member 0 pulls; members 1-9 hit cache in <1s). Misleading "11 Fetching CDS" log lines per (var, time) are NOT cache misses -- the log message is printed BEFORE the cache check, so 1 real fetch + 10 instant cache hits all log as "Fetching".

## Why aifsens was "fast" (debunked)

aifsens completed-week wallclocks of 1.5-6h were misleading. Per-init time was the SAME 30-60 min as aifs_perturbed; aifsens just had most of its weeks' inits already on disk from prior runs (auto-skip). The first-round aifsens jobs (long before our STORE migration) paid the full CDS cost too.

## Solution

Pre-download all 5376 CDS requests (224 unique fetch times x 25 var-groups) IN PARALLEL into the cache. The cache key is deterministic so as long as we match earth2studio's hash exactly, the inference will hit cache and never call CDS.

Cache key (verified against existing on-disk files at `/iopsstor/scratch/cscs/sadamov/e2s_cache/cds/`):
```python
sha256(f"{dataset_name}_{variable}_{'_'.join(level)}_{time}".encode())
```
- `dataset_name`: `reanalysis-era5-single-levels` or `reanalysis-era5-pressure-levels`
- `variable`: human-readable name (e.g. `2m_temperature`, `geopotential`)
- `level`: list of level strings, joined with `_`. For PL vars, the order is the iteration order of `AIFS.VARIABLES` (alphabetically sorted), giving `['100', '1000', '150', '200', '250', '300', '400', '50', '500', '600', '700', '850', '925']`. For surface vars, `['']`.
- `time`: str(datetime) = "YYYY-MM-DD HH:MM:SS"

Hash match verified 2026-06-02 06:13 by predicting `2m_temperature 2023-01-04 18:00` and `geopotential 13-level 2023-01-04 18:00` -- both predicted files exist on disk with sizes 2 MB and 26 MB.

## Variables AIFS pulls per timestamp (25 CDS request groups)

6 PL vars x 13 levels (one request each, packed): geopotential, specific_humidity, temperature, u_component_of_wind, v_component_of_wind, vertical_velocity.

19 surface vars (one request each): 10m_u/v, 100m_u/v, 2m_dewpoint, 2m_temperature, geopotential (sfc), land_sea_mask, mean_sea_level_pressure, skin_temperature, slope_of_sub_gridscale_orography, soil_temperature_level_1/2, standard_deviation_of_orography, surface_pressure, total_column_water, volumetric_soil_water_layer_1/2.

18 AIFS vars are NOT in CDS and earth2studio computes them locally: cos/sin julian_day, cos/sin latitude/local_time/longitude, cp06, insolation, ssrd06, strd06, sf, tcc, mcc, hcc, lcc, ro. We do not need to fetch these.

`total_precipitation_06` (tp06) is a special case earth2studio synthesizes from 6 individual tp pulls -- we skip it in the pre-download; it's cheap on the inference path.

## Script

`/iopsstor/scratch/cscs/sadamov/pre_download_aifs_ic.py`

- ThreadPoolExecutor with --workers=5 (CDS quota is 5 concurrent per user; raising to 6+ risks 429 / queue ejection).
- Atomic write: temp file in same dir then `os.rename` so a partial write can't be mistaken for cache hit by a concurrent inference.
- Idempotent: skips any file already present at the target hash path. Safe to resume.

## Status (2026-06-02 morning, updated through ~09:40)

- 06:15: Launched PID 191412 (5 workers) in scope `aifs_predl_1780373751.scope`.
- 09:30 check: 45 OK in 3:01h = 15/h = ~20 min per CDS request through the queue. Way slower than the 78 s/req single-test rate. CDS throttled us hard at 5 concurrent (plus aifs_perturbed single test 2452954 was ALSO eating a CDS slot).
- 09:32: scancel test 2452954, kill PID 191412, **relaunch as PID 283926 with 3 workers** in scope `aifs_predl_1780384683.scope` (16G).
- 09:40 check: 21 min elapsed, 0 OK so far. 3 active TCP connections to CDS object storage 160.79.104.10:443 confirmed -- script is blocked waiting on CDS queue, not crashed. Threads=4 in /proc/<pid>/status (3 workers + main). Need to be patient -- ECMWF queue assigns slots on its own timeline.

If 3 workers also stalls for hours: **drop to 1 worker** as the last resort. Single thread = no concurrency penalty. ETA at 78 s/req = 84h serial -- bad but predictable.

## Stage 4 (pending until cache full)

Once `du -sh /iopsstor/scratch/cscs/sadamov/e2s_cache/cds` flatlines:
1. scancel single test 2452954
2. `bash scripts/submit_all_inference.sh aifs_perturbed` to refire all 8 weeks
3. Each week should now run at compute-bound speed: 14 inits x ~5 min compute = ~1h wall per week

## Rate-limit watch

User flagged 2026-06-02 ~06:30 the concern that running pre-download + 5 parallel SLURM eval + IFS download all at once might trigger CDS / MARS throttling. Mitigations if signals appear:
- HTTP 429 from CDS -> drop pre-download workers from 5 to 3
- MARS "queue position" suddenly increasing -> pause IFS IC download
- New aifs_perturbed weeks ejected -> wait for pre-download to mostly finish before resubmitting

## Related

- [[e2s-cds-cache-mount]] -- the bind-mount that makes this cache shareable across jobs
- [[ifs-ens-ic-experiment]] -- parallel IFS download with similar memory-cap pattern
