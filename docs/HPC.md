# Running on CSCS Alps / Clariden

Everything heavy (inference, SwissClim evaluation, analysis over the zarr archives) ran as Slurm jobs on
Clariden. This page collects what you need to run it again and the failure modes we hit.

## Accounts, partitions, limits

- Slurm account: `ab016`. The older allocation `a122` is frozen (QOS `stop`): submitting with it fails with
  `AssocMaxSubmitJobLimit`. Check yours with `sacctmgr show assoc user=$USER format=account,qos`.
- The submitters pass `--account="${AIENS_ACCOUNT:-ab016}"`; scripts that carry `#SBATCH --account=ab016`
  are overridden on the command line: `sbatch -A <account> tools/submit_x.sh`.
- Partitions: `normal` (12 h max, the default for everything), `debug` (1.5 h, one job per user), `xfer`
  (data movement, no GPU, 24 h; used for rsync/ACL jobs). Per-node maximum: `--time=12:00:00 --mem=800G`.
- Node: 4 x GH200 (aarch64, SM 9.0). Inference jobs use one node with 4 GPU workers; GraphCast (JAX) runs
  members sequentially on one GPU.
- Login nodes: no builds, no installs, no zarr-heavy analysis (processes get killed). Use a CPU sbatch.
  For long login-node processes use `setsid nohup CMD < /dev/null > LOG 2>&1 & disown` and unbuffered
  output (`python -u`); the four login nodes do not share process tables.

## Paths and environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AIENS_STORE` | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` | project store (forecasts, evals, images) |
| `AIENS_SCRATCH` | `/iopsstor/scratch/cscs/sadamov` | scratch for TMPDIR, dask spill, earth2studio cache (purged without notice) |
| `AIENS_PY` | `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python` | host interpreter used by the Slurm scripts |
| `AIENS_REPO` | `/users/sadamov/pyprojects/ai-models-ensembles` | repo checkout that jobs `cd` into and bind-mount |
| `AIENS_ACCOUNT` | `ab016` | Slurm account |
| `AIENS_IFS_ENS` | `/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr` | IFS-ENS reference |
| `AIENS_WB2_22`, `AIENS_WB2_24`, `AIENS_WB2_ORIGINAL` | `/capstor/store/cscs/swissai/weatherbench/...` | ERA5 truth |
| `AIENS_MILTON_CONDA` | `/capstor/store/cscs/mch/s83/sadamov/miniforge3` | TempestExtremes + udunits for Milton |
| `AIENS_MILTON_GAPFILL` | `0` | `1` reconstructs IFS-ENS NaN leads in the tracker (paper setting: off) |

A successor working from their own checkout sets at least `AIENS_REPO` and `AIENS_SCRATCH` (and
`AIENS_STORE` if writing to a copy). Always redirect temp files to scratch: the scripts export
`TMPDIR=$AIENS_SCRATCH/tmp` and `DASK_TEMPORARY_DIRECTORY` (never `/tmp`).

HTTPS from compute nodes needs `export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')`.

## Running inference in a container

The submitters assemble this; by hand it looks like:

```bash
srun -A ab016 -p normal --time=02:30:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=800G --gres=gpu:4 \
  --container-image=$AIENS_STORE/aurora.sqsh --container-workdir=/workspace/ai-models-ensembles \
  --container-mounts=$AIENS_REPO:/workspace/ai-models-ensembles,$AIENS_REPO/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,$AIENS_STORE:$AIENS_STORE,$AIENS_SCRATCH/e2s_cache:/workspace/.cache/earth2studio \
  python -m ai_models_ensembles.cli infer --model aurora --init 2023-01-02T00:00 --lead-hours 240 \
    --members 10 --weight-magnitude 0.025 --layer encoder --data-source arco --output /path/forecast.zarr
```

- Use `python -m ai_models_ensembles.cli` (the bind-mounted code), not the `ai-ens` entry point (the copy
  baked into the image).
- AIFS / AIFS-ENS: `--data-source cds`, mount `~/.cdsapirc` and `~/.ecmwfapirc` into both `~` and `/root`,
  and mount `$AIENS_STORE/e2s_cache_backup` as the cache (warm CDS GRIBs; a cold cache costs ~2 h per init).
- Phase 6 noise refresh is switched on by environment variables read inside the container:
  `{SFNO,AURORA,AIFS,GRAPHCAST}_FRESH=1`, `*_FRESH_SIGMA`, `*_FRESH_REFRESH_EVERY` (and
  `SFNO_FRESH_MODE_CUT`, `SFNO_FRESH_MODE_SKIP`); `tools/submit_p6c_reseed.sh` sets them.

## Submitters

| Script | Submits | Notes |
|---|---|---|
| `scripts/submit_all_inference.sh [experiment ...]` | production grid, one sbatch per (experiment, week) | `PER_INIT=1` one sbatch per init (SFNO), `CHAIN=1` sequential (CDS), `WEEKS=`, `AFTER_JOB=`, `DRY_RUN=1`; skips existing zarrs |
| `scripts/submit_ablation.sh <phase> [model]` | ablation grid | phases `phase1 phase2 phase2b phase3 phase3b all` |
| `scripts/evaluate_baselines.sh all <experiment>` | 8 SwissClim jobs (one per module), 800 GB / 12 h each | without an experiment it submits the whole matrix (100+ jobs): always name one |
| `scripts/evaluate_baselines.sh intercompare` | cross-baseline intercomparison | |
| `scripts/evaluate_ablation.sh <phase> [model]` / `intercompare <phase>` | per-run eval / per-phase intercomparison | count forecasts first (below) |
| `scripts/submit_ic_backfill.sh`, `scripts/submit_seed_robustness.sh` | IC backfill of the `_ic` runs; seed-43 ablation reruns | |
| `tools/submit_*.sh` | analysis jobs (CPU) | listed in [REPRODUCING.md](REPRODUCING.md) |
| `containers/submit_build.sh <model|all>` | container builds on a compute node | 1 h, 444 GB |
| `tools/grant_group_read.sh` | ACLs on the store (xfer partition) | run by the file owner |

Keep concurrent inference to about 12 nodes (convention, shared allocation).

## Failure modes and fixes

- `SIGSEGV (-11)` in all GPU workers at once: state accumulating when one process runs many inits (CUDA
  contexts, leaked `/dev/shm/sem.mp-*`). Failed members are retried automatically (`E2S_ROUND_RETRIES`);
  for SFNO use `PER_INIT=1` (one init per job).
- `blosc decompression: 0` reproducibly on one init: a truncated download poisoned the fsspec cache.
  `find $E2S_CACHE_DIR/arco -maxdepth 1 -type f -size -1k -delete`.
- Jobs hang 45-90 min in "preparing data + model", then `Errno 19` / `transport endpoint shutdown`: degraded
  capstor mount, not the code (`time ls -d $AIENS_STORE/baselines/<x>/*/forecast.zarr` should be sub-second).
  Do not resubmit into it; a 2026-08-28 incident truncated one init this way.
- AIFS ~2 h per init instead of ~8 min: cold CDS cache. ARCO cannot replace CDS for AIFS (no soil/land
  fields). Parallel CDS jobs throttle each other: `CHAIN=1`.
- Evaluation slow (hours per module): zarr chunking does not match SwissClim's; reshard with
  `tools/submit_reshard.sh`.
- `evaluate_ablation.sh` with fewer than 4 forecasts present writes configs whose output names embed the
  wrong init range, which later breaks the intercomparison. Count forecasts before evaluating; after a
  re-run delete the stale `eval/<run_tag>` and its config and evaluate again.
- Scratch purge: files vanish, directories stay. Check with `find <dir> -type f | wc -l`. Keep anything you
  need beyond a few weeks on `$AIENS_STORE`.
- A venv on `/iopsstor` gets half-deleted by the purge (packages' top-level files gone, imports fail while
  `site-packages` looks populated). Keep venvs and conda on capstor.
- A job leaving the queue is not success: check `sacct -j <id> -o JobID,State,ExitCode,Elapsed` and the log.
- Bash: `((count++))` returns 1 when `count` is 0 and kills a `set -e` script; use `count=$((count+1))`.
- `pkill -f <pattern>` over ssh matches its own remote shell; use `pgrep -f` and `kill <pid>`.

More operational notes (each a short incident write-up): `.claude/reference/hpc/` and
`.claude/reference/project-conventions.md`.
