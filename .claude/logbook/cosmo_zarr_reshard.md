---
name: cosmo-zarr-reshard
description: "COSMO regional training zarr being resharded from v2 (~19 TB, 27 vars at /capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data.zarr) to v3 with sharding codec to match SwissClim/AI-models-ensembles inner-chunk convention. Smoke test passed 2026-06-01 20:35, full reshard in flight as job 2451970."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Source

- Path: `/capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data.zarr`
- Format: zarr v2
- Size: ~19 TB
- 27 variables, mix of 2D (HSURF, FR_LAND, etc.) and 3D (T, QV, U, V, etc. with time/y/x dims, time=43133 timesteps, y=390, x=582).

## Target

- Path: `/capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data_sharded.zarr`
- Format: zarr v3, sharding codec
- Inner chunks per `tools/reshard_cosmo_zarr.py`: `{time: 1, z: 1, y: -1, x: -1}` -- one time-slab per inner chunk, full y/x in each chunk. Same principle as `tools/reshard_zarr.py` uses for the forecast.zarr stores.
- Streaming write: 3D vars (~2.3 TB each) processed in time-slabs (default 128 timesteps per slab) to keep peak RAM bounded.

## Status

- Smoke test 2451937 (smoke output `cosmo_ml_data_sharded_smoke.zarr`, 21 GB): COMPLETED 3:11, chunks verified (HSURF dims=(y,x) chunks=(390, 582); T_2M dims=(time,y,x) chunks=(1, 390, 582)).
- Full reshard 2451970: **COMPLETED 2026-06-02 03:14 (6:39h elapsed).**
- User verified locally 2026-06-02 ~06:30 that the resharded zarr contains the full arrays. Source `/capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data.zarr` is SAFE TO DELETE (user will rm).
- Smoke artefact `cosmo_ml_data_sharded_smoke.zarr` (21 GB) also safe to delete.

## Started in a parallel session

`tools/reshard_cosmo_zarr.py` was authored in a parallel Claude Code chat. Committed in [3c0c8c0] follow-on alongside MEMORY.md updates. Source on a122 stays until full reshard verifies + user moves/removes it.

## Why

Matches the per-project convention from [[checkpoint-perturbation-audit]] / `tools/reshard_zarr.py`: avoid SwissClim `enforce_chunking` rechunk-on-read penalty (4h-per-member ETS vs 12 min with correct chunks). Same payoff applies to COSMO when used as training input or as ground truth for limited-area-model verification.
