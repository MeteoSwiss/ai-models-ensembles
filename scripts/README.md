# Workflow Execution Scripts

Slurm submitters that wrap `python -m ai_models_ensembles.cli infer` /
`verify` / `intercompare` and run them inside the per-model containers under
[../containers/](../containers/). Everything is configured by editing the
constants at the top of each script - there is no shared `config.sh`.

All scripts share the same hard-coded `STORE` root
(`/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles`) since
`$STORE` on Clariden login nodes points at the wrong project.

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

Eval modules used:
`energy_spectra, probabilistic, deterministic, multivariate, histograms`
(plus `ets`, `fss`, `ssim` via the temp scripts below until merged into the
main pipeline).

Output: `$STORE/ablation/<phase>/<model_id>/eval/<run_tag>/...` and
`.../intercomparison/...`.

## submit_etsfss_phase1.sh, submit_ssim_phase1.sh (temporary)

Phase 1 ETS/FSS and SSIM eval as standalone passes (run before these modules
were integrated into the main pipeline). Will be removed once the next eval
pipeline run absorbs them.

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

- **Job won't start**: check `sinfo` and account `a122` access.
- **Container launch fails**: confirm `$STORE/<model>.sqsh` exists; rebuild
  with `bash containers/submit_build.sh <model>`.
- **CDS throttling on `aifsens`**: rerun with `CHAIN=1` (forces sequential
  per-week submission via `--dependency=afterany`).
- **Pre-fix zarrs read slowly in eval**: chunking changed; reshard with
  [../tools/submit_reshard.sh](../tools/submit_reshard.sh).

## See also

- [../README.md](../README.md) - top-level overview
- [../tools/README.md](../tools/README.md) - host venv + inspection tools
- [../containers/](../containers/) - per-model Dockerfiles and build helpers
