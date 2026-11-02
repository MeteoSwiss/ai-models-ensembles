---
name: feedback-cluster-max-resources
description: Always submit cluster jobs with --time=12:00:00 and --mem=800G -- the max on Clariden. System-wide rule across all projects.
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When submitting any sbatch on Clariden (or by extension any CSCS HPC project), default to `--time=12:00:00` and `--mem=800G`. These are the per-job maxima on the `normal` partition and there is no benefit to asking for less.

Why: there is no per-job billing penalty for requesting more than you use -- the scheduler bills based on actual elapsed time, not requested wall time, and memory is per-node so a single job already has the full node's RAM available. Asking for less just opens up failure modes:
- 6h walltime exhausted on a module that turned out heavy -> TIMEOUT -> partial output -> stale-config gotcha cascade (hit 2026-05-27 on ETS members eval; hit again on the bundled main eval).
- 444G memory exhausted on a multivariate / probabilistic eval -> OOM (hit 2026-05-27 on the light bundle: det+ssim killed with exit 0:125 / OOM at 52 min in).
Both failures cost wall-clock time and require chains to be re-wired. Setting max wall + max mem makes single-shot success the norm.

How to apply:
- Default for every sbatch in this project's `scripts/` directory: `--time=12:00:00 --mem=800G`.
- Applies to inference jobs (already 12h/444G; bump to 800G), eval jobs (current 12h/444G is the cap-and-OOM combo we just hit; bump to 800G), intercomp jobs (current 6h/128G is fine for now but no harm in bumping mem).
- Applies system-wide -- not specific to ai-models-ensembles. Take the same rule to any other CSCS HPC project.
- Exception: very short utility jobs (trampolines, dry checks) where 4G/10min is intentionally small to be cheap. Don't bump these.
- If a partition has stricter caps (e.g. `debug` 1.5h), use the partition's max.
