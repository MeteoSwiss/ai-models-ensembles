---
name: ifs-zarr-reshard
description: Reshard IFS source zarrs (a122 v2 layout) to v3 sharded at /capstor/store/cscs/mch/s83/IFS/. ifs_control DONE 2026-06-01 (smoke-test verified). ifs_ens in flight.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# IFS zarr reshard to v3 sharded

## Status (2026-06-07 evening)
- **ifs_control.zarr resharded: DONE.** Output at `/capstor/store/cscs/mch/s83/IFS/ifs_control.zarr` (731 GB, 10 vars x 224 inits x 24 leads x 12 levels). Smoke-tested 2026-06-01. Source `/capstor/store/cscs/swissai/a122/IFS/ifs_control.zarr` (697 GB) is SAFE TO DELETE.
- **ifs_ens.zarr reshard: DONE.** Resume sbatch `2485238` COMPLETED 06:28 on 2026-06-07 (3h 34m wall). All 16 vars at 112/112 inits, `.reshard_state.json` final, metadata consolidated, log ends "DONE". Sharded output 20 TB at `/capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr`; **9085 files total** (vs millions in the v2 source - sharding worked as designed). Structural cross-check vs old v2: identical dims `{init_time:112, ensemble:50, lead_time:61, level:3, latitude:721, longitude:1440}` and identical 15 vars (10m_u/v/wind_speed, 2m_t, geopotential, MSL, q, RH, T, t_p, t_p_24/6hr, u/v_wind, wind_speed).
- **ifs_ens.zarr cross-project relocate: DONE** (verified 2026-06-08). rsync from `/capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr/` -> `/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr/` (`-a --remove-source-files`) completed; dest = 20T, 15 vars, opens with dims `{init_time:112, ensemble:50, lead_time:61, level:3, latitude:721, longitude:1440}`, first/last slabs 100% finite. Sharding = 1 shard/init_time so ~112-113 files per var (geopotential = 112 shards + zarr.json); dest has ~1680 files total. NOTE: the earlier "9085 files" figure (line above) was actually the source HUSK inode count (7411 dirs + ~1674 files), NOT a file count - no data loss. mch/s83 husk `rmdir`'d (was 0 files, 7411 empty dirs). `evaluate_baselines.sh:65` already points at the swissai/a122 dest - no edit needed (reshard-then-relocate-back-under-original-name means the original path is correct again). `.reshard_state.json` left in dest root (harmless, benign open warning). Relocate ran via setsid on a DIFFERENT login node and was 99% done when checked this session - node-local `ps` gave a false negative, see [[feedback-check-all-login-nodes]].
- **`.OLD_v2` deletion: DONE** (2026-06-08 20:15:01, `rm` PID 196012 on ln003 ended `DONE rc=0`, ~600k dirs removed in ~13 min). `ls -d ....OLD_v2` confirms gone. capstor inode pressure dropped from 98% back to comfortable. The 20 TB v3 relocate + v2 backup cleanup is fully complete; nothing left on this thread.

The original ifs_ens.zarr (19 TB) and ifs_control.zarr (697 GB) live at `/capstor/store/cscs/swissai/a122/IFS/` in old zarr-v2 layout (high inode count, no sharding). Reshard to v3 with `INNER_CHUNKS` policy matching SwissClim `DESIRED_CHUNKS` (ensemble/lat/lon full, init/lead/level=1).

**Why:** evaluate_baselines reads the ifs_ens zarr; chunk mismatch hurts perf (see [[zarr-v3-fill-value]] note about chunking). Also frees ~500K inodes on a122 once source is removed.

**How to apply:** target output is `/capstor/store/cscs/mch/s83/IFS/` (separate quota). Once reshard + verify is clean, update `evaluate_baselines.sh:IFS_ENS_ZARR` path to the new location. Don't `rm -rf` source until both zarrs are verified end-to-end.

## Scripts
- `/iopsstor/scratch/cscs/sadamov/reshard/reshard_ifs_streaming.py`
  - Streams per (variable, init_time) slabs.
  - State file `<dst>/.reshard_state.json` records completed slabs for resume.
  - Bit-equality verify per slab via `np.array_equal(..., equal_nan=True)`.
  - Modes: default = full conversion, `--paranoia N` = stop after N inits per zarr (smoke test), `--verify-only` = re-read source and compare against existing dst.
  - INNER_CHUNKS policy: ensemble/lat/lon full, init/lead/level=1. 1 shard per init_time.
  - Source NEVER auto-deleted.
- `/iopsstor/scratch/cscs/sadamov/reshard/submit_reshard_ifs.sh`
  - SLURM wrapper, 1 node x 72 cpus x 800G x 12h (full) or 1h (paranoia).
  - Both zarrs in parallel via background subprocesses + wait.
  - Usage: `bash submit_reshard_ifs.sh paranoia both` / `bash submit_reshard_ifs.sh full both` / `bash submit_reshard_ifs.sh verify ifs_ens`.

## Job log
- Paranoia test: job 2434064 (1h walltime, normal partition), submitted 2026-05-30.
- Logs at `/iopsstor/scratch/cscs/sadamov/reshard/logs/reshard_ifs_paranoia_both_<JID>.{out,err}` + per-zarr `reshard_{ifs_control,ifs_ens}_paranoia.log`.

## Next step
If paranoia verifies clean for both zarrs: launch `full both` (12h walltime). If 12h insufficient, resume picks up from `.reshard_state.json`.

## Related
- [[store-relocation-2026-05-30]] (mch/s83 destination)
- [[feedback-zarr-v3-fill-value]] (zarr v3 sharding pitfalls)
- [[feedback-never-use-tmp-on-hpc]] (TMPDIR/DASK redirection enforced in `COMMON_ENV`)
