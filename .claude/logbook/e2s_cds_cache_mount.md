---
name: e2s-cds-cache-mount
description: "2026-06-01 fix: bind-mount /iopsstor/scratch/cscs/sadamov/e2s_cache -> /workspace/.cache/earth2studio in submit_all_inference.sh so the earth2studio CDS GRIB cache survives across jobs, weeks, AND models (aifsens, aifs_perturbed). Without it, CDS-bound inference jobs re-download tens of GBs per init from a slow queued API."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Symptom

aifs_perturbed production jobs (V1: 2448601, 03-09 + 2451874) ran 9 hours and only completed 53/112 inits, on track to time out with ~7/14 inits per week. Logs showed endless `state: accepted` polling from earth2studio CDS:

```
2026-06-01 21:58:23.628 | DEBUG | earth2studio.data.cds | _download_cds_grib_cached:325 |
  Request ID:42c0f228-..., state: accepted
... 30+ identical accepted-polls over ~60s, then state: completed
```

Each init does ~30-40 separate per-variable CDS calls; each call sits in CDS's queue for ~30-60s before Copernicus schedules it. With aifsens having already pulled the same 112 inits the day before, this was pure waste.

## Diagnosis (2026-06-01 22:15)

`from earth2studio.data import CDS; CDS().cache` returns `/workspace/.cache/earth2studio/cds`. That path was NOT in `submit_all_inference.sh`'s container `--container-mounts` list, so it lived inside the ephemeral container layer:

- Within a single job: members 1-9 hit cache for inits member 0 downloaded.
- Across jobs (different weeks): no sharing -- cache vanished when container exited.
- Across models (aifsens completed all 112 inits, then aifs_perturbed reran the same): no sharing.

Aifsens had already pulled tens of GBs of GRIBs for the exact same (init, var) combinations aifs_perturbed needed. Throwing all of that away was the inefficiency.

## Fix

Edited `scripts/submit_all_inference.sh` to add:

```bash
E2S_CACHE_DIR="/iopsstor/scratch/cscs/sadamov/e2s_cache"
mkdir -p "$E2S_CACHE_DIR"
mounts="...,${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"
```

Now every container instance sees a persistent host dir at the CDS cache path. First job that pulls a (init, var) GRIB writes it there; every subsequent job (same or different model, same or different week) hits cache instead of CDS.

V2 jobs 2452954-62 submitted 2026-06-01 22:24 use the new mount. Expected wall-clock ~3-6h per week (mostly cache hits after the first job populates) instead of 12+h (CDS-bound).

## Verification

After the first V2 job's first init lands, `ls /iopsstor/scratch/cscs/sadamov/e2s_cache/` should show GB-scale GRIB files. If empty, the bind-mount syntax in `submit_all_inference.sh` is broken (most likely cause: `--container-mounts` doesn't accept `host:container` for non-existent host paths -- but we `mkdir -p` first so this should be fine).

## Why not switch AIFS to ARCO?

ARCO doesn't ship all of AIFS's required input fields (particularly `soil_temperature_level_2` and a few other land-surface fields). Earth2studio defaults AIFS to CDS for that reason. The cache mount is a strictly-better fix because it preserves the upstream defaults while eliminating the redundant API hits.

## Related

- [[ifs-ens-ic-experiment]] -- the other login-node download that also went through CDS / MARS API queues.
- `scripts/submit_all_inference.sh:138` -- where the mount is configured.
