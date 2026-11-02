---
name: sfno-modes10-baseline
description: SFNO modes10 sigma=0.25 launched as a 5th probabilistic baseline (vs fcn3/atlas/aifsens/ifs_ens) over the full 112-init baseline grid
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

The SFNO Phase 3 modes10 sigma=0.25 winner (see [[calibration-winners]]) is being run as a full probabilistic baseline so we can compare CRPS/SSR head-to-head against trained probabilistic baselines on the SAME init grid (not the 4 ablation inits).

Why: paper claim is "post-hoc perturbation CRPS within 20-30% of trained probabilistic". To support that quantitatively we need apples-to-apples on the 112-init baseline grid, not on the ablation grid. SFNO modes10 was chosen because it's the cleanest "physics-motivated" winner -- coarse spectral modes are the most defensible structural targeting story.

How to apply: When summarizing baseline comparisons or paper-ready numbers, include sfno_modes10 alongside the 4 trained probabilistic baselines. Use the 5-way intercomp at `$STORE/baselines/intercomparison/` (the old 4-way will be replaced).

## Setup

- Init grid: 112 init_times (8 weeks x 7 days x 2/day) -- identical to fcn3/atlas/aifsens/ifs_ens
- Lead: 360h (matches baselines, longer than ablation's 240h)
- Vars: 9-var baseline subset (incl. 10m winds) -- identical to other baselines
- Levels: 500, 850
- Members: 10
- Seed: 42
- Perturbation: `--weight-magnitude 0.25 --coarse-mode-cut 10`
- Output: `$STORE/baselines/sfno_modes10/<YYYYMMDD_HHMM>/forecast.zarr`

## Script changes (committed for this run)

- `scripts/submit_all_inference.sh`: added `sfno_modes10` variant with `CONTAINER_BASE=sfno` and `EXTRA_FLAGS="--weight-magnitude 0.25 --coarse-mode-cut 10"`. Output path uses variant key, not model_id, so other baselines unchanged.
- `scripts/evaluate_baselines.sh`: added `sfno_modes10` to MODELS list + case branch. Added `AFTER_JOB` env-var support to `submit_main_eval` / `submit_etsfss_eval` (mirrors existing pattern in `run_intercompare`). Job-tag echo redirected to stderr so stdout has only parsable IDs (for chaining).

## Chain layout (final state, 2026-05-26 evening)

Inference required multiple rounds. Failure pattern was SFNO multi-GPU SIGSEGV in round 2/3 of a later init within the same week-helper process tree (root cause: multiprocessing state accumulation; see MEMORY.md "SFNO inference recurring..." section). Final job IDs:

- **Initial batch** (8 week-helpers, 2381461-68): 4 completed cleanly (2381463, 2381466, 2381467, 2381750 incl. retry), 4 partial-then-SIGSEGV.
- **Retry batches** (cursed weeks): 2381733, 2385273, 2385603, 2386373 all SIGSEGV'd at varying init# of week 20230102.
- **Stagger experiment** (3-min `--begin=now+Nmin`): 2385348-50, 2385603 -- still got 1 SIGSEGV (2385603). Confirmed contention isn't the dominant cause.
- **Per-init breakthrough** (6 jobs for 20230102 missing inits, 2392054-59): ALL succeeded in ~11 min each running simultaneously. Smoking gun for the within-process state-accumulation root cause.
- **Trampoline 2392060**: count guard verified 112/112, submitted eval + intercomp cleanly.

Lesson committed to script: see `submit_all_inference.sh` PER_INIT mode + IPC cleanup between python calls.

Chain trampoline body (`baseline_eval_logs/chain_sfno_modes10.sh`):
   - Calls `bash scripts/evaluate_baselines.sh all sfno_modes10` -- regenerates configs WITH all 112 forecasts now present, submits main + etsfss eval, captures both IDs from sbatch --parsable stdout.
   - Submits 5-way intercomp dependent on both eval IDs. Intercomp body: `rm -rf $STORE/baselines/intercomparison && python -m ai_models_ensembles.cli intercompare ...` (overwrites old 4-way).
3. **Eval (2 jobs)**: main (stride 6h, all modules except ets/fss) + etsfss (stride 24h, ets/fss only).
4. **5-way intercomp (1 job)**: atlas + fcn3 + aifsens + ifs_ens + sfno_modes10. Dropped `--module fss` from intercomp because 5-way FSS at 112 inits is brutal -- can add back later if needed.

## Why the trampoline matters

Critical gotcha: `evaluate_baselines.sh` generates the eval YAML config at submission time by globbing for forecast.zarr. If submitted before inference completes, only the K<112 forecasts present at submission go into `paths.ml`. Subsequent intercomp `common_files()` intersection then drops everything because NPZ filenames embed the init range. The trampoline defers config generation until inference is fully done. Same gotcha was documented at [[evaluate-ablation-stale-config]] (now [[evaluate-stale-configs]]) for the ablation case.

## Eval recovery 2026-05-27

First attempt (main eval 2392672 + etsfss 2392673) both TIMEOUT'd as expected per the universal 12h limit: main got through maps/wd_kde/energy_spectra in 11h, never reached deterministic/prob/ssim/multivariate. ETSFSS got 4/26 ETS files, 0/12 FSS files in 6h.

Recovery via `scripts/evaluate_baselines_remaining.sh` (the same pattern used for the other 4 baselines after their identical timeouts): one job per (model, heavy module). Submitted 6 per-module jobs (2398789-94): light bundle (det+ssim), multivariate, probabilistic, ets, fss, energy_spectra. 5-way intercomp 2398796 chained afterany.

This isn't a SFNO-specific problem -- the eval YAML, zarr layout, chunks, shards, and var list are all identical between sfno_modes10 and atlas/fcn3/aifsens. The 12h walltime is just universally too tight for the bundled main eval.

## Validation TODO once intercomp lands

- Compare 5-way CRPS line plots: where does sfno_modes10 sit relative to aifsens (36.0), atlas (36.9), fcn3 (38.0), ifs_ens?
- Check SSR: is the SSR=1.09 from the 4-init ablation grid stable at 112 inits?
- ETS/FSS at thresholds 75/95 percentile: does the perturbation preserve sharpness?
