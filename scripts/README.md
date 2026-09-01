# Workflow Execution Scripts

Slurm submitters that wrap `python -m ai_models_ensembles.cli infer` /
`verify` / `intercompare` and run them inside the per-model containers under
[../containers/](../containers/). Everything is configured by editing the
constants at the top of each script - there is no shared `config.sh`.

Each script sets `STORE` and the interpreter to a CSCS default that can be
overridden with `AIENS_STORE` / `AIENS_PY` (see "Running the analysis off
the CSCS box" in the [main README](../README.md)); `$STORE` on Clariden
login nodes points at the wrong project, hence the explicit default.

## submit_all_inference.sh

Probabilistic baseline inference for the IFS ENS eval period:

- 3 models: `fcn3`, `atlas`, `aifsens`
- 8 weeks (Jan/Apr/Jul/Oct 2023 + 2024), 14 init times each
- 10 members, 360 h lead, levels 500/850
- One sbatch job per (model, week); 4 GPUs in parallel inside the job.

```bash
bash scripts/submit_all_inference.sh                  # all 3 models, all 8 weeks
bash scripts/submit_all_inference.sh fcn3 atlas       # subset of models
CHAIN=1 bash scripts/submit_all_inference.sh aifsens  # chain via afterany (CDS throttling)
AFTER_JOB=2119857 bash scripts/submit_all_inference.sh fcn3
WEEKS=2023-01-02,2024-07-02 bash scripts/submit_all_inference.sh
```

Output: `$STORE/baselines/<model_id>/<YYYYMMDD_HHMM>/forecast.zarr`. Existing
zarrs are skipped automatically.

## submit_ablation.sh

Weight-perturbation ablation across 3 deterministic models (`aurora`,
`graphcast_operational`, `sfno`), reproducible from constants in the script.

Phases:

- `phase1`: magnitude sweep (5 magnitudes x 3 models x 4 init times = 60 runs).
- `phase2`: layer-group sweep at each model's best Phase 1 magnitude.
- `phase2b`: fine magnitude refinement on each model's best layer group.

```bash
bash scripts/submit_ablation.sh phase1               # all models
bash scripts/submit_ablation.sh phase1 aurora        # one model
bash scripts/submit_ablation.sh phase2
bash scripts/submit_ablation.sh all                  # phase1 + phase2 + phase2b
```

Output: `$STORE/ablation/<phase>/<model_id>/<init_tag>/<run_tag>/forecast.zarr`.

## evaluate_ablation.sh

SwissClim Evaluations driver for the ablation outputs. Generates a per-run
YAML and submits one sbatch per (model, run). After per-run eval, an
`intercompare` mode runs cross-run comparisons.

```bash
bash scripts/evaluate_ablation.sh phase1                       # eval all Phase 1 runs
bash scripts/evaluate_ablation.sh phase1 aurora                # per-model
bash scripts/evaluate_ablation.sh intercompare phase1          # cross-run plots
bash scripts/evaluate_ablation.sh intercompare phase1 aurora
```

Eval modules: `maps, wd_kde, energy_spectra, multivariate, probabilistic,
deterministic, ssim, fss`.

Output: `$STORE/ablation/<phase>/<model_id>/eval/<run_tag>/...` and
`.../intercomparison/...`.

## submit_ic_backfill.sh / submit_seed_robustness.sh

`submit_ic_backfill.sh` fills missing pressure levels in the IFS-ENS
perturbed-IC store; `submit_seed_robustness.sh` reruns the production picks
under alternate seeds for the seed-sensitivity check.

## `$STORE` output tree

```text
$STORE/
  baselines/
    {fcn3,atlas,aifsens,ifs_ens}/                  trained-probabilistic + physical
    {aurora_encoder,graphcast_all,sfno_modes10,aifs_perturbed}/   post-hoc weight perturbation
      <YYYYMMDD_HHMM>/forecast.zarr
    <id>/eval/...                                  per-baseline SwissClim modules
    intercomparison/                               cross-model plots + temporal_metrics_combined.csv
  ablation/
    <phase>/<model_id>/                            phase in {phase1,phase2,phase2b,phase3,phase3b}
      <init_tag>/<run_tag>/forecast.zarr
      eval/<run_tag>/{maps,wd_kde,energy_spectra,multivariate,deterministic,probabilistic,ssim,fss}/
      intercomparison/
    allphases/<model_id>/intercomparison/          cross-phase ablation summary
```

`forecast.zarr` uses the SwissClim chunk/shard layout; a store written with
the wrong chunking reads slowly in eval - reshard with
[../tools/submit_reshard.sh](../tools/submit_reshard.sh).

## Typical sequence

```bash
# 1. Build containers once (or after Dockerfile changes)
bash containers/submit_build.sh all

# 2. Ablation inference (Phase 1)
bash scripts/submit_ablation.sh phase1

# 3. Ablation eval after Phase 1 inference completes
bash scripts/evaluate_ablation.sh phase1
bash scripts/evaluate_ablation.sh intercompare phase1

# 4. Probabilistic baselines (independent of ablation)
bash scripts/submit_all_inference.sh

# 5. Pick best magnitudes and run Phase 2 / 2b
bash scripts/submit_ablation.sh phase2
bash scripts/submit_ablation.sh phase2b
```

## Troubleshooting

- **Job won't start**: check `sinfo` and your Slurm account access.
- **Container launch fails**: confirm `$STORE/<model>.sqsh` exists; rebuild
  with `bash containers/submit_build.sh <model>`. Builds must go to a compute
  node - the login nodes cannot build.
- **`import earth2studio` fails on the host**: expected; it lives only in the
  containers. Run inside `$STORE/<model>.sqsh`.
- **CDS throttling on `aifsens`**: rerun with `CHAIN=1` (sequential per-week
  via `--dependency=afterany`).
- **AIFS inference ~2 h/init instead of ~8 min**: cold CDS GRIB cache. ARCO
  cannot substitute for AIFS (no land-surface fields); `submit_all_inference.sh`
  mounts the persistent cache `$STORE/e2s_cache_backup` for `cds` runs.
- **`SIGSEGV (-11)` from every GPU worker at startup**: transient GH200/UCX
  race; failed members relaunch automatically, raise `E2S_ROUND_RETRIES`.
- **`blosc decompression: 0` reproducibly on one init**: a truncated write
  poisoned the fsspec cache. `find $E2S_CACHE_DIR/arco -maxdepth 1 -type f -size -1k`
  and delete the zero-length entries.
- **Jobs hang in "preparing data + model" 45-90 min then `Errno 19` /
  `transport endpoint shutdown`**: degraded capstor mount, not the code.
  `time ls -d $STORE/baselines/<run>/*/forecast.zarr` (healthy is sub-second);
  do not resubmit into it.
- **Pre-fix zarrs read slowly in eval**: chunking changed; reshard with
  [../tools/submit_reshard.sh](../tools/submit_reshard.sh).
- **`envsubst: command not found`**: install `gettext-base`.

## See also

- [../README.md](../README.md) - top-level overview
- [../tools/README.md](../tools/README.md) - host venv + inspection tools
- [../containers/](../containers/) - per-model Dockerfiles and build helpers
