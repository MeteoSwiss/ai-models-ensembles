---
name: store-relocation-2026-05-30
description: "a122 hit 100% inode quota 2026-05-30. STORE root migrated to /capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles. Migration in flight, scripts updated, fresh jobs already write to new location."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# STORE relocation 2026-05-30

a122 inode quota (50M) hit 100% on 2026-05-30 from accumulated `_e2s_work` zarr-v2-style files across failed AIFS Phase 1/2 jobs. Decision: migrate entire `ai-models-ensembles` tree to `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` (mch/s83 has 5M inode quota, only ~31% used).

**Why:** independent inode quota on mch/s83; same Lustre filesystem so bandwidth is identical; no waiting for a122 cleanup tickets.

**How to apply:** any new script/path reference must use the new STORE root. The old `/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles` will be removed once Phase 3 of the migration completes (rsync verify -> rm source).

## Migration job
- Script: `/iopsstor/scratch/cscs/sadamov/migration_to_mch_s83.sh`
- PID: 108050 (started 2026-05-30 19:38)
- Log: `/iopsstor/scratch/cscs/sadamov/migration_to_mch_s83.log`
- Phases: rsync copy (Phase 1) -> rsync -anHc verify (Phase 2) -> rm -rf SRC (Phase 3, only if verify clean)
- Crash-safe: restartable via `rsync --partial --inplace`; state preserved between attempts.

## Scripts updated (commit 4f22e5a)
14 files had STORE updated via sed: `scripts/submit_all_inference.sh`, `scripts/submit_ablation.sh`, `scripts/evaluate_ablation.sh`, `scripts/evaluate_baselines.sh`, `containers/submit_build.sh`, `tools/run_model_tests.sh`, `tools/dump_sfno_keys.py`, `tools/submit_reshard.sh`, `scripts/relaunch_phase1_sfno_graphcast.sh`, etc.

## Fresh jobs already write to new STORE
- 24 AIFS Phase 1+2 sbatch jobs (2434025-2434047) resubmitted 2026-05-30 with IPC cleanup fix (d210b9d).
- IFS download (PID 263264) writes elsewhere (a122/IFS/ifs_analysis_perturbed_ic.zarr -- not migrated, separate dataset under a122).
- IFS zarr reshard (job 2434064) writes to `/capstor/store/cscs/mch/s83/IFS/` -- separate from the ai-models-ensembles tree but using the same s83 quota.

## Related
- [[feedback-verify-kill]] (zombie process discipline)
- [[feedback-never-use-tmp-on-hpc]] (TMPDIR/DASK redirection still applies)
