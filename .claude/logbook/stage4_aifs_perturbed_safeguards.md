---
name: stage4-aifs-perturbed-safeguards
description: "2026-06-04 Stage 4 aifs_perturbed production launch + safeguards around the CDS GRIB cache (sentinel file + capstor backup) that the run depends on. 8 SLURM jobs 2467808-15 PENDING at submission, cache warm at 3821/3821, backup at /capstor/.../e2s_cache_backup/."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Stage 4 launch (2026-06-04 ~08:25)

Submitted with `bash scripts/submit_all_inference.sh aifs_perturbed`. Eight SLURM jobs for the production 8 weeks:

| job | week |
| --- | --- |
| 2467808 | 2023-01-02 |
| 2467809 | 2023-04-02 |
| 2467810 | 2023-07-02 |
| 2467811 | 2023-10-02 |
| 2467812 | 2024-01-02 |
| 2467813 | 2024-04-02 |
| 2467814 | 2024-07-02 |
| 2467815 | 2024-10-02 |

All PENDING (Priority) on `normal` partition, 12 h wall, --mem=800G. Will start as nodes free up.

**Container cache mount is wired** in `scripts/submit_all_inference.sh:143-145` (commit `7116001`):
```bash
E2S_CACHE_DIR="/iopsstor/scratch/cscs/sadamov/e2s_cache"
mkdir -p "$E2S_CACHE_DIR"
mounts="...,${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"
```

Stage 4 depends critically on the cache being warm; otherwise every (init, var) pair re-queries CDS (30-60 s/file at the public queue rate -> ~6-12 h/week of CDS download time alone). See [[e2s-cds-cache-mount]] + [[aifs-predownload-cache]] for the full pre-download history.

## Cache warm-state at launch

```
/iopsstor/scratch/cscs/sadamov/e2s_cache/
  aifs-single-1.1/                       (model assets)
  aifs-single-1.1_interpolation_matrix/  (model assets)
  aifs-single-1.1_inverse_interpolation_matrix/
  cds/                                   (the 5400 GRIBs)
  ifs-opendata/
  README_DO_NOT_DELETE.md
```

Total live: 44 GB, ~5400 files. Predownload finished 3821/3821 OK earlier today (see [[aifs-predownload-cache]] + [[feedback-no-uv-pip-install-project-venv]] for the certifi/cdsapi recovery story behind the last 407).

## Safeguard 1: sentinel `README_DO_NOT_DELETE.md`

Wrote `/iopsstor/scratch/cscs/sadamov/e2s_cache/README_DO_NOT_DELETE.md` to protect against any future scratch-cleanup job or rm-by-mtime sweep. Contents document:
- what populated the cache (aifsens + aifs_perturbed)
- file count + GB total
- recreation cost (CDS API key + 24-48 h + cooperation from CDS)
- the bind-mount entry in `submit_all_inference.sh` so anyone investigating can follow the trail
- explicit "DO NOT: rm -rf / prune by mtime / include in scratch cleanup / tarball without keeping live copy" list

## Safeguard 2: capstor backup

Rsynced the entire live cache to `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/e2s_cache_backup/` (5393 files / 44 GB). Verified completion by finding 5 top-level subdirs + the README + 5393 files under the backup, matching live. iopsstor is project-scratch (purge-eligible long-term); capstor/store is the durable home.

If iopsstor ever loses the cache, restore via:
```bash
rsync -ahS --info=progress2 /capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/e2s_cache_backup/ /iopsstor/scratch/cscs/sadamov/e2s_cache/
```

## Cache-hit verification protocol (when jobs start)

When a Stage 4 job moves PENDING -> RUNNING, tail
`/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baseline_logs/bl_aifs_perturbed_<init>_<jobid>.out`
for either:
- WARM (good): no `earth2studio.data.cds | _download_cds_grib_cached:325 ... state: accepted` polling. Forecast member init proceeds directly to model setup.
- COLD (bad): repeated `state: accepted` polls (30-60 s each) before `state: completed`. Means the bind-mount is not seeing the cache. Cancel jobs, debug mount, do not rerun.

The first job's first init is the highest-signal moment. After that, members 1-N within the same job always hit cache (intra-job behaviour was fine pre-mount).

## Standing rules (forever)

- NEVER `rm -rf /iopsstor/scratch/cscs/sadamov/e2s_cache/` or any subdir.
- NEVER include this path in scratch cleanup, mtime pruning, or scratch-quota recovery.
- NEVER delete `README_DO_NOT_DELETE.md` from this dir.
- If iopsstor scratch hits quota, talk to sadamov first; do not free space by touching the cache.
- The backup at capstor is the recovery source; keep it intact unless explicitly asked.

## Related

- [[e2s-cds-cache-mount]] -- the bind-mount itself
- [[aifs-predownload-cache]] -- how the cache got populated
- [[feedback-no-uv-pip-install-project-venv]] -- the cdsapi/certifi mishap that almost stalled the predl
- [[ifs-ens-ic-experiment]] -- the parallel IFS perturbed-IC download for Phase 5 hybrid
