---
name: ifs-ens-ic-experiment
description: IFS ENS perturbed analyses as IC for Aurora / GraphCast / SFNO baseline runs -- physical-perturbation alternative to weight perturbation. MARS download in flight 2026-05-28.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

Hybrid experiment: use real IFS ENS perturbed analyses as the initial conditions for the post-hoc weight-perturbation baselines, instead of the single ERA5 IC currently used. Combines aleatoric (real EDA-derived IC spread) with epistemic (our weight perturbation) uncertainty.

Motivation: the current baselines initialize ALL 10 weight-perturbed members from the SAME ERA5 IC. Trained probabilistic models (atlas/fcn3/aifsens) effectively get both via their stochastic training. Real IFS ENS analyses add the missing aleatoric component without changing the perturbation method.

## What we're downloading

Script: `download_ic_perturbed.py` in `/capstor/store/cscs/swissai/a122/IFS/tools/repo-download-ifs/`. Submit via `cd <repo> && sbatch submit_ic_perturbed.sh` — purpose-built SLURM wrapper, 800G/12h on `normal` partition, already sets TMPDIR + DASK_TEMPORARY_DIRECTORY → iopsstor, SSL_CERT_FILE → certifi bundle, PYTHONUNBUFFERED=1. NOT `~/pyprojects/ifs_download/` (that's a different / older repo without the IC variant).

- Stream: `class=od stream=enfo type=pf step=0`, `number=1..50` -- the actual per-member perturbed analyses, ready to feed AI models as ICs.
- Upper-air on pressure levels (13 incl. 600 hPa): T/U/V/Q/Z/W (params 130/131/132/133/129/135)
- Surface: MSL, 2T, 10U, 10V, 100U, 100V, SP, TCWV (151/167/165/166/246/247/134/137)
- Init time grid: 8 weeks × 28 6-hourly samples = **224 init times**, extended 6h backward of the baseline 00/12Z grid to cover Aurora/GraphCast t-6h autoregressive input.

Output: single Zarr v3 sharded store at `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr`. Dims `(ensemble=50, init_time=224, level=13, lat=721, lon=1440)` for 3D vars, no `lead_time` dim. ~3.8 TB.

## Why type=pf step=0 (not type=icp)

PF step=0 = the per-member perturbed analysis itself, ready to use. ICP is the perturbation delta (`member - control`) and needs the control analysis added back per member. Direct is simpler.

## Why this stream over model levels (ml)

User initially suggested `levtype=ml type=icp` which has all 137 model levels but: (a) only T/U/V/Q + VO/D/LNSP (we'd need to derive Z, do ML->PL interpolation), and (b) the example showed `time=00:00:00/06:00:00/18:00:00` (no 12Z). The `levtype=pl` stream in `stream=enfo` has 14 levels including 600 hPa, all variables we need (including W vertical velocity), and all 4 daily times. Skips ML->PL preprocessing entirely.

## Status (2026-06-08 ~20:45) -- 222/224, RESUMED for last 2 inits

- **Download died at 222/224.** The ln004 scope (`ifs_ic_resume4.scope`, PID 264642) completed idx 0..221 (222 inits, all finite) then VANISHED with NO error in the log right after `writing slice idx=221` (2024-10-08 00:00) at 20:11:22 -- likely an external/node kill at the loop boundary (idx 221's 18 GB load had just succeeded, so not the usual mid-init OOM). The "Done" at 20:10:40 was a MARS GRIB transfer, NOT job completion. Verified via zarr: idx 222 (2024-10-08 06:00) + idx 223 (2024-10-08 12:00) are 100% NaN; idx 0..221 finite.
- **RESUMED 2026-06-08 ~20:45 on clariden-ln003**, scope `ifs_ic_resume5.scope`, MemoryMax=64G, python PID **243450** (bash 243449). Same `--skip-init-store`; confirmed skipping the 222 done inits then will fetch idx 222 + 223. ETA ~1-1.5h (2 inits x ~18-36 min MARS). Log `..._ic_nohup.log`. **After-break**: confirm `[223/224]` + `[224/224] OK`, then re-check the zarr has all 224 init_times finite.
- zarr `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr` = ~1.9T (final ~1.93T, NOT the 3.8T earlier estimate).

## ZARR FORMAT (verified 2026-06-08) -- MARS short names, NOT SwissClim
- dims `(ensemble=50, init_time=224, level=13, latitude=721, longitude=1440)`; levels `[1000,925,850,700,600,500,400,300,250,200,150,100,50]`.
- 14 data_vars in **MARS shortName** form: upper-air (13 levels) `z, t, u, v, q, w`; surface `2t, 10u, 10v, 100u, 100v, msl, sp, tcwv`.
- **Consequence for Phase 5 wiring**: `XarrayDataSource.from_swissclim` maps upper-air `z/t/u/v/q/w` -> `z500/t850/...` correctly (heads already match e2s), and `msl/sp/tcwv` pass through, BUT surface `2t,10u,10v,100u,100v` do NOT remap to the `t2m,u10m,v10m,u100m,v100m` tokens the models request. Needs a MARS->SwissClim (or MARS->e2s) rename, either as a cheap metadata rename of the zarr (also makes it eval-pipeline-consistent) or a MARS-aware branch in `ifs_ens_member_ic_source`. See [[phase5-perturbed-ic-plan]].
- **CROSS-NODE LESSON** still applies: the dead process was found GONE via `ssh clariden-ln00{1..4}`. See [[feedback-check-all-login-nodes]].

## Status (2026-06-01 evening update)

- **Login-node nohup launches (PIDs 234010, 263264) all eventually died on the login node.** Last attempt PID 263264 ran from 2026-05-30 20:04 through 2026-06-01 10:29, made it to init 78/224, then died while "loading merged dataset into memory (~18 GB)" -- login-node OOM-killer chose us as victim. Verified dead via `kill -0` 2026-06-01 20:30.
- **Decision 2026-06-01 21:25: stay on the login node BUT cap memory via user systemd scope** (`systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0`). User preference: preserve scarce SLURM node hours. The cgroup hard-cap means if the process exceeds 32 GiB, the kernel kills ONLY our process, not the login node.
- **Relaunched 2026-06-01 21:28 as PID 286466 in systemd transient scope `run-r96d773b34bce4e8e900751ad253610ca.scope`** (verified `MemoryMax=34359738368 = 32 GiB`, `MemorySwapMax=0`). Script reads progress.log on startup and SKIPs the 78 inits already on disk; resumes from init 79. Logs append to `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic_nohup.log`. PID file at `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.pid`.
- **PID 286466 also OOM-died** by ~22:00 -- last log line was init 18/224 SKIP at 21:35:27, the next iter would have been the first real MARS request and never finished. 32 GiB still too low.
- **Root cause + fix 2026-06-01 ~22:25**: `download_ic_perturbed.py:fetch_and_write_one` was missing explicit GC after the per-init `to_zarr`. The previous iteration's `merged` ds (~18 GB) survived into the next iter until Python rebound the name, peaking ~36 GB during the next merge step. Added `import gc; del merged; del pl_ds, sfc_ds, pl_src, sfc_src; gc.collect()` immediately after `_cleanup_earthkit_cache()`. Committed in (this batch).
- **Relaunched 2026-06-01 22:29 as PID 294037 in NEW systemd scope `ifs_ic_perturbed_1780345787.scope`** (verified `MemoryMax=51539607552 = 48 GiB`, `MemorySwapMax=0`). Cap bumped 32 → 48 GiB belt-and-braces alongside the GC fix. Same `--skip-init-store`; auto-resumes from init 19 (next non-SKIP after init 18). **After-break: `ps -p 294037` to confirm alive; tail `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic_nohup.log` to confirm log appending past init 18 SKIP and showing real MARS PL/SFC requests.**
- **DO NOT submit `submit_ic_perturbed.sh` (SLURM) without user OK** -- the script is correct (800G/12h, all envs proper) but consumes node-hours that are tight.

## Status (2026-05-29 morning, historic)

- **Paranoia (2419254) completed 2026-05-28 evening.** Real timing: PL request 18-23 min wall (server post-processing dominant), 1 active task per user, only **11 of 13 requested PL levels archived** (150 + 600 hPa missing across all 6 params uniformly).
- **Fill strategy: pure log-p interpolation** for 150 + 600 hPa (user-decided 2026-05-29). Preserves member spread at both levels at cost of slight accuracy loss; control-fill alternative would have given physical levels but zero spread at those 2 cells. The relevant 4-day download budget plus per-init cleanup makes interpolation the simpler win.
- **First launch (2026-05-29 ~08:55 on ln004) burned 3 minutes with a fill_value bug**: `init_zarr_store` created the template with `np.empty(...)` arrays, no explicit `fill_value`. zarr v3 defaulted float fill_value to 0.0, the np.empty zeros leaked into the zarr at to_zarr time, and `already_written`'s `np.isfinite` check passed for every init -> all 224 logged as SKIP, no MARS requests made, process exited cleanly in 3 min. ln004 went down later, unrelated -- our footprint was 139 MB RSS.
- **Fix (2026-05-29)**: replaced `np.empty` with `dask.array.full(..., np.nan)` (lazy, no RAM) + explicit `fill_value=NaN` in encoding. Also hardened `already_written` to reject all-zero slabs as defense-in-depth. Verified: empty template reads as NaN everywhere, `already_written` returns False.
- **First-launch bug 2 (region-write coord rejection, 2026-05-29 ~10:11)**: after init 1 MARS PL+SFC completed cleanly, `merged.to_zarr(region={'init_time': slice(...)})` failed with `ValueError: ... must have at least one dimension in common with the region's dimensions ['init_time']`. xarray's region write rejects any coord variable that doesn't share the region's dims. After `expand_dims({'init_time': ...})`, the ensemble/level/latitude/longitude coord variables still only carry their own dims. The script's per-init try/except logged FAIL and moved to init 2; that one fetched fine from MARS but the process died around 10:32 (likely ln003 churn, was on shared login node). Total cost: ~17 min wasted MARS quota on init 2 PL.
- **Fix 2 (2026-05-29 10:45)**: drop ensemble/level/latitude/longitude coords from `merged` before region-write. Template's `init_zarr_store` already wrote those coord values. Committed as `f8a96b5` on `sadamov/ifs_download`.
- **Relaunched 2026-05-29 10:46 on clariden-ln002** (PID 234010, `--skip-init-store` since zarr already exists with correct NaN template). Earthkit cache HIT for init 1 PL+SFC -> init 1 wrote OK in 1:15 (vs the usual 17 min PL + few min SFC). Init 2 onwards goes through MARS normally (cache cleanup after each success nukes all cached GRIBs). Estimated remaining: ~95h.
- **Output**: `/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr` (Zarr v3 sharded). Final size ~3.8 TB.

## MARS constraints + batching strategy

**1 active task per user** on the ECMWF MARS web API. So our 448 sequential requests (224 PL + 224 SFC) all serialize, no 2x parallelism available. At ~13 min per request that's ~4 days end-to-end at the current per-init batch granularity. The script is already a blocking 1-active loop so no code change needed -- just budget accordingly.

Rebatching options (decide after paranoia gives real per-request timing):

| Batching       | # requests | rough wall time | retry granularity |
|----------------|------------|-----------------|-------------------|
| per (init, levtype) -- current | 448 | ~4 days | per-init failure = ~30 min loss |
| per (week, time, levtype) | 64 | ~1.5-2 days | per-batch failure = ~3 hr loss |
| per (week, levtype) | 16 | ~1-1.5 days | per-batch failure = ~12 hr loss |

Tape extraction is typically sub-linear (one mount, then bulk read), so batching probably wins on total wall time. Tradeoff: bigger requests fail catastrophically.

## Open questions for paranoia review

- Confirm `type=pf step=0` returns per-member perturbed analyses with same values as `cf + icp`.
- Confirm dimensions / units / NaN content all look physical (T ~ 250 K at 500 hPa, MSL ~ 101000 Pa, etc).
- 3,300 fields per PL request observed via MARS web UI = 50 members × 6 params × **11** levels. Two of our 13 requested levels (probably 10 and 50 hPa) are NOT archived for type=pf step=0. Adjust PL_LEVELS constant after paranoia confirms which levels are missing.
- Decide on batching strategy based on actual per-request timing from paranoia.
