# Project conventions - deep technical reference

On-demand detail behind the summary in `CLAUDE.md`. SIGSEGV history, intercomp structure, Phase 2/3
implementation, JAX/CDS specifics, eval-design lessons. Read when touching the relevant code path.

## SFNO / multi-GPU SIGSEGV (the per-init escape hatch)
- Root cause (diagnosed 2026-05-26): the multi-GPU SIGSEGV is NOT primarily cross-job concurrency - it
  is within-job state accumulation when one Python process runs many inits sequentially. Week-helpers run
  14 inits/process; CUDA context / NCCL handle / leaked semaphores accumulate and eventually corrupt, so
  round 2/3 of a later init SIGSEGVs in all 4 GPU workers at once.
- Smoking gun: 6 SFNO jobs each running ONE init (per-init 2392054-59) all completed clean in ~11 min; the
  14-init week-helper for the cursed 20230102 week crashed 5x. One-init-per-job = no accumulation = no crash.
- Mitigations in `submit_all_inference.sh`: default week-helper runs `find /dev/shm -name 'sem.mp-*' -delete`
  between python calls (clears leaked semaphores). Opt-in `PER_INIT=1` = one sbatch per (model, init) with
  2-min stagger (full isolation; 14x more jobs). Other env vars: `CHAIN=1` (sequential), `PER_INIT_TIME_LIMIT`,
  `WEEKS=YYYY-MM-DD,...`. `sfno_pi` ids = the per-init smoking-gun experiment, NOT production cost - exclude
  from any cost/energy accounting.
- Older parallel-inference SIGSEGV fixes (e2s_inference.py): detect GPUs via CUDA_VISIBLE_DEVICES/nvidia-smi
  (no CUDA init in parent); fresh `mp.get_context("spawn")` per round; per-worker `del model; synchronize();
  empty_cache(); ipc_collect(); gc.collect()`; one per-round retry of failed members after 30s backoff.
  Affects atlas/fcn3/aurora/sfno. GraphCast (JAX, sequential) unaffected. Per-round /dev/shm cleanup inside
  `_run_members_parallel` (commit d210b9d) fixed the round-3 SFNO/AIFS crash the per-init cleanup missed.

## Ablation intercomparison structure
- `run_intercompare` (evaluate_ablation.sh) submits TWO sbatch per (phase, model) into the same intercomp dir:
  (A) non-prob modules WITH `mag_0_layer_all` (the leftmost reference panel); (B) prob-only WITHOUT mag_0
  (mag_0 is N=1, no probabilistic/ subdir, so SwissClim `common_files()` intersection would zero out and drop
  CRPS Maps / PIT / Spaghetti). Baselines unaffected (all N>=2).
- Cross-phase reference pull: for phase2+, reaches into `phase1/<m>/eval/` for `mag_0_layer_all` +
  `mag_0.01_layer_all`. One unified `run_intercompare` handles all phases (the old `phase3_crosscompare` was
  deleted). `PHASE2_BEST_TAG` table picks the phase-2 winner pulled into phase-3 intercomp.
- evaluate_ablation gotcha: `submit_eval` builds the YAML by globbing forecast.zarr at SUBMISSION time. If
  K<4 inits exist then, output NPZ filenames embed the init range and break SwissClim `common_files()` at
  intercomp (prob count ~57 instead of ~100). After re-running failed inferences, delete the stale eval dir
  + its config.yaml and re-evaluate with all 4 forecasts present. Always count forecasts before submitting eval.
- Phase 6 forecast location (don't go hunting): the SFNO refresh-N weight-noise runs live in the PHASE3 tree,
  `$STORE/ablation/phase3/sfno/<init>/mag_<sigma>_modes10_refresh<N>_<sigma>/forecast.zarr` (the sweep reused
  the phase3 run-tag convention), NOT under `phase6_family_v2/sfno/`. `phase6_family_v2/sfno/` holds ONLY the
  SwissClim `intercomparison/` aggregates by design - the run-tag forecast dirs were never stored there.
  Canonical path also in logbook `phase6d_higher_anchor.md:156-158`. (A scout once mis-concluded these zarrs
  were deleted by looking only in the phase6_family_v2 tree - they were never gone.)

## Phase 2 / Phase 3 implementation
- Phase 2: per-group sigma = `PHASE2_SIGMA_FULL[model]=0.01 * sqrt(N_TOTAL / N_GROUP)`, hardcoded N tables in
  submit_ablation.sh from checkpoint dumps. SFNO groups = encoder(3)/processor(80)/decoder(3) (the old
  thirds-of-79 split was wrong, off by 8 complex tensors). Perturbs the same tensor categories as Phase 1.
- Phase 3 GraphCast JAX broadcasting bug (fixed 2026-05-21, graphcast_coarse_perturbation.py): build noise as
  `jax.random.normal(key, shape=latent_mesh_nodes[:n].shape)` and use `.at[:n].set(coarse_slice*(1+sigma*noise))`
  directly - the old `factor.at[:n].multiply` against a 3D `[42,1,512]` mesh failed (`[n,1]` noise won't
  broadcast over the trailing 512).
- AIFS perturb target (added 2026-05-29): `_perturb_torch` loads via `obj.load_state_dict(state, strict=False)`
  (state_dict() returns a copy); `_save_anemoi_ckpt` re-packs the anemoi torch-zip so `inference-last/anemoi-metadata/*`
  survives; `_PERTURB_SKIP_PREFIXES=("pre_processors.","post_processors.")` never perturbs normaliser stats.

## Baseline eval design + lessons
- 7 canonical intercomp "models": atlas, fcn3, aifsens (trained) + ifs_ens (classical ref) + aurora_encoder,
  graphcast_all, sfno_modes10 (post-hoc). `evaluate_baselines.sh all <model>` ALWAYS submits per-module (8
  sbatch/model, --mem=800G --time=12:00:00) since 12h is insufficient for the bundled all-modules eval at 112
  inits. bundled-main/bundled-etsfss kept as legacy escape hatches. ETS dropped 2026-05-27.
- IFS-ENS zarr subsampled to stratified 10 members `[0,5,10,...,45]` to match AI ensemble size.
- dask_profile `balanced` (32 workers) default, `safe` (16x24G) under memory pressure. 6h stride for line-plot
  modules; det/ssim are memory-heavy single-stream.
- 3 bivariate pairs: T x q (Clausius-Clapeyron 500hPa), u x v wind, geopotential-gradient x wind_speed (geostrophic 500hPa).

## CDS / aifsens constraints
- aifsens must run sequentially (`CHAIN=1`): earth2studio CDS client fetches one variable at a time (176
  sequential requests/init); parallel jobs throttle each other. ARCO can't replace CDS (missing tcw/tp06/stl1/stl2).
  Per-init prefetch makes ~36 sequential CDS requests; the `aifs_predownload_cache` work pre-fills the on-disk
  cache to dodge this. AIFS v2 dropped: needs 6 wave-period-band vars not in `reanalysis-era5-single-levels`.

## GraphCast / JAX specifics
- Sets `CUDA_VISIBLE_DEVICES='0'` (JAX needs CUDA to init); forced sequential (XLA recompiles per member,
  multi-GPU OOMs); `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `_prefetch_ic_data` writes a /tmp/_discover_vars.py
  helper (not inline -c). GraphCast deterministic by construction (no stochastic JAX ops).

## Data integrity + misc
- Before each large eval spot-check `xr.open_zarr(path).data_vars` - reshard is non-destructive and does NOT
  detect missing variables (e.g. aifsens 20230706_1200 had 7/9 vars, re-inferred 2026-05-15).
- swissclim_evaluations is editable-installed from local `SwissClim_Evaluations/` (research branch); `uv sync`
  can drop it - reinstall on ModuleNotFoundError.
- Bash gotcha: `((count++))` returns exit 1 when count=0; with `set -e` it kills the script. Use `count=$((count+1))`.
- SwissClim module output paths: ETS/FSS are standalone (`ets/`, `fss/` dirs); intercomp reads
  `ets/ets_metrics*.csv` + `fss/fss_metrics*.csv`.
