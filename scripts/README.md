# scripts/

Slurm submitters for inference and SwissClim verification. They wrap `python -m ai_models_ensembles.cli`
(`infer`, `verify`, `intercompare`) and run it inside the per-model containers ([../containers/](../containers/)).
Each script is configured by the constants at its top plus a few environment variables; there is no shared
config file. Paths default to the CSCS locations and are overridable (`AIENS_STORE`, `AIENS_SCRATCH`,
`AIENS_PY`, `AIENS_WB2_*`, `AIENS_IFS_ENS`), the Slurm account with `AIENS_ACCOUNT` (default `ab016`).
Cluster usage and failure modes: [../docs/HPC.md](../docs/HPC.md). Output layout: [../docs/DATA.md](../docs/DATA.md).

## submit_all_inference.sh - production grid

```bash
bash scripts/submit_all_inference.sh                          # the 7 headline experiments
bash scripts/submit_all_inference.sh aurora_encoder sfno_modes10
PER_INIT=1 bash scripts/submit_all_inference.sh sfno_modes10  # one sbatch per init (SIGSEGV-safe)
CHAIN=1 bash scripts/submit_all_inference.sh aifsens          # sequential (CDS throttling)
WEEKS=2023-01-02,2024-07-02 bash scripts/submit_all_inference.sh fcn3
DRY_RUN=1 bash scripts/submit_all_inference.sh atlas          # print, do not submit
```

- Grid: 8 weeks (`WEEK_STARTS`: Jan/Apr/Jul/Oct 2023 + 2024) x 14 inits = 112 inits, 10 members, 360 h,
  levels 500/850. One sbatch per (experiment, week) runs the 14 inits with 4 GPU workers; existing
  `forecast.zarr` are skipped.
- Experiments (`EXTRA_FLAGS`, `DATA_SRC`, `MODEL_IDS`, `MEMBERS`): the headline set `fcn3 atlas aifsens
  sfno_modes10 aurora_encoder graphcast_all aifs_perturbed`; the IC-only arms `*_ic_only` (`--ic-zarr` with the
  IFS-ENS perturbed analyses); the rivals `aurora_enc_s044 graphcast_m2g graphcast_g2m sfno_enc_s054
  sfno_enc_s035 aifs_all_s010`; `aifs_det` (unperturbed, 1 member). Any of them can be named on the command line.
- AIFS / AIFS-ENS use `cds` and the persistent cache `$STORE/e2s_cache_backup`; the CDS credentials
  (`~/.cdsapirc`, `~/.ecmwfapirc`) are mounted into the container.
- Output: `$STORE/baselines/<experiment>/<YYYYMMDD_HHMM>/forecast.zarr`, logs and the generated job scripts in
  `$STORE/baseline_logs/`.

Related: `submit_ic_backfill.sh` (fills the IC-augmented `*_ic` runs, weight + IFS-ENS perturbed ICs) and
`../tools/submit_p6c_reseed.sh` (Phase 6c refresh runs). `submit_seed_robustness.sh` reruns the four
production picks with seed 43 on the ablation grid (`$STORE/ablation_seed43/`).

## submit_ablation.sh - ablation grid

```bash
bash scripts/submit_ablation.sh {phase1|phase1_unperturbed|phase2|phase2b|phase3|phase3b|all} [model]
```

4 inits (2023-05-15, 2023-08-15, 2024-02-15, 2024-11-15), 10 members, 240 h, models `aurora`,
`graphcast_operational`, `sfno`, `aifs`. Phase 2 sigmas are `0.01 * sqrt(N_total / N_group)` with the tensor
counts hardcoded from the checkpoint dumps. Output:
`$STORE/ablation/<phase>/<model_id>/<YYYYMMDD>/<run_tag>/forecast.zarr` (run-tag conventions in
[../docs/DATA.md](../docs/DATA.md#5-experiment-names-and-run-tags)), logs in `$STORE/ablation_logs/`.

## evaluate_baselines.sh - SwissClim on the production grid

```bash
bash scripts/evaluate_baselines.sh all <experiment>           # 8 sbatch, one per module, 800 GB / 12 h each
bash scripts/evaluate_baselines.sh gen-configs <experiment>   # only write the YAMLs
bash scripts/evaluate_baselines.sh intercompare               # cross-experiment intercomparison
```

Always name the experiment: `all` without one submits every module for every experiment in `MODELS`
(100+ jobs). Modules: `maps, wd_kde, energy_spectra, multivariate, probabilistic, deterministic, ssim, fss`.
Lead 0-360 h, 6-hourly stride, IFS-ENS restricted to the stratified members 0,5,...,45. Output:
`$STORE/baselines/<experiment>/eval/<module>/` with the config next to it (`eval_<module>_only_config.yaml`);
intercomparison in `$STORE/baselines/intercomparison/`.

## evaluate_ablation.sh - SwissClim on the ablation grid

```bash
bash scripts/evaluate_ablation.sh <phase> [model]              # one sbatch per (model, run)
bash scripts/evaluate_ablation.sh intercompare <phase> [model] # per-phase comparison
bash scripts/evaluate_ablation.sh allphases_intercompare [model]
```

Builds each run's YAML by globbing the forecasts at submission time: make sure all 4 inits exist first (see
HPC.md). Output: `$STORE/ablation/<phase>/<model_id>/eval/<run_tag>/<module>/` and `.../intercomparison/`;
cross-phase in `$STORE/ablation/allphases/<model_id>/intercomparison/`.

## Typical order

```bash
bash containers/submit_build.sh all                  # once
bash scripts/submit_ablation.sh phase1               # then evaluate, intercompare, pick, next phase
bash scripts/evaluate_ablation.sh phase1 && bash scripts/evaluate_ablation.sh intercompare phase1
bash scripts/submit_all_inference.sh                 # production grid with the picks
bash scripts/evaluate_baselines.sh all <experiment>  # per experiment
bash scripts/evaluate_baselines.sh intercompare
```
