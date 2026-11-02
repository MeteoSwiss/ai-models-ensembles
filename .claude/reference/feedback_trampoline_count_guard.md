---
name: feedback-trampoline-count-guard
description: "Always guard chain trampolines (inference->eval->intercomp) with an explicit forecast-count check before submitting eval, to avoid stale-config gotcha"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When writing a trampoline shell script that chains inference jobs to eval + intercomp (any `afterany` dep on inference), put an explicit forecast-count guard at the top: count expected forecasts on disk, refuse to submit eval if count < expected, exit with a clear error.

Why: SLURM `afterany` fires regardless of inference success or failure. SwissClim eval configs are generated at submit time from a glob over `forecast.zarr` dirs, so a trampoline that fires with N<N_expected forecasts produces an eval that runs on N inits. The NPZ output filenames embed the init range, and downstream intercomp `common_files()` intersection with other baselines (each with N_expected inits) silently drops entire plot categories (spaghetti, CRPS spatial, PIT, etc.). Hit this on 2026-05-26 with the SFNO modes10 baseline: trampoline fired with 72/112 forecasts because 4 inference jobs SIGSEGV'd, and the eval would have produced a broken intercomp. Cost: 1h of GPU time on cancelled eval. Without the guard, would have wasted ~12h.

How to apply:
- Any trampoline that chains inference -> eval should start with:
  ```bash
  EXPECTED=<count>   # full set you expect; e.g. 112 for the baseline grid
  ACTUAL=$(find "<output_dir>" -maxdepth 2 -type d -name forecast.zarr | wc -l)
  if [[ "$ACTUAL" -lt "$EXPECTED" ]]; then
      echo "ABORT: only $ACTUAL/$EXPECTED forecasts present; refusing to submit eval."
      exit 1
  fi
  ```
- The trampoline's stderr/stdout should make the abort obvious so it's easy to spot when checking `sacct` (which we should always do per [[feedback-check-fast-completed-jobs]]).
- Example committed pattern: `$STORE/baseline_eval_logs/chain_sfno_modes10.sh` (the trampoline that wraps `scripts/evaluate_baselines.sh`).
- Permanent fix candidate (not yet implemented): move the eval config generation from submit-time to job-runtime inside the eval job itself; that makes the count guard redundant. Until then, the trampoline guard is the practical safety net.
- Applies to both `evaluate_baselines.sh` (baseline grid, 112 inits) and `evaluate_ablation.sh` (ablation grid, 4 inits). Adjust EXPECTED accordingly.
