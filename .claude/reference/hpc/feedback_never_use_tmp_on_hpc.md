---
name: feedback-never-use-tmp-on-hpc
description: NEVER use /tmp on this CSCS HPC cluster (or any HPC cluster). Redirect TMPDIR + DASK_TEMPORARY_DIRECTORY + similar spill paths to /iopsstor/scratch.
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

On CSCS Clariden (and most HPC clusters) `/tmp` is a small tmpfs on each compute node. For this cluster `/tmp` typically has only ~94 GB available, which is fine for shell scratch but catastrophically too small for any dask spill, large NumPy intermediate, or downloaded model checkpoint. Anytime a Python / dask / training job spills to disk by default, it spills to `/tmp` and silently fills it -> `OSError: [Errno 28] No space left on device`.

Why: on 2026-05-28, four SwissClim eval jobs (deterministic + ssim for sfno_modes10 / aurora_encoder / graphcast_all at 112 inits) all hit this -- each ran 50-60 min and then failed with disk-full while dask spilled intermediate state to /tmp. Symptom: the job COMPLETED with exit code 0 (SwissClim caught the OSError and exited cleanly) but produced no output. The fix is trivial -- redirect spill -- but it must be done explicitly.

How to apply, for every sbatch wrap on this cluster:
```bash
sbatch ... --wrap="export TMPDIR=/iopsstor/scratch/cscs/sadamov/dask-tmp && \
                   export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/dask-tmp && \
                   <actual command>"
```

`/iopsstor/scratch` is the fast parallel filesystem (typical: 2.5 PB total, hundreds of TB free). Project-private subdir: `/iopsstor/scratch/cscs/sadamov/dask-tmp` -- pre-create with `mkdir -p`.

Applies system-wide, not just this project. Take the same TMPDIR + DASK_TEMPORARY_DIRECTORY redirect to any CSCS HPC project. Same rule as [[feedback-cluster-max-resources]] -- one of the "default for every sbatch" rules.

Other spill paths to know about (override the same way if used):
- `XDG_CACHE_HOME` for cached models
- `HF_HOME` / `TRANSFORMERS_CACHE` for HuggingFace
- `TORCH_HOME` for PyTorch hub
- `NVIDIA_NGC_CACHE` for NGC checkpoints
