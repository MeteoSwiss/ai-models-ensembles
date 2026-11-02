---
name: feedback-uv-cache-on-capstor
description: "ALWAYS keep the uv cache (`UV_CACHE_DIR`) on /capstor (or home) on CSCS, NEVER on /iopsstor/scratch. Scratch gets aggressively cleaned and breaks the cache mid-install. System-wide rule."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

On CSCS Clariden (and likely other CSCS systems) `/iopsstor/scratch/` is fast but gets purged on a cleanup schedule (files untouched >~30 days are removed). If the uv package cache (`UV_CACHE_DIR`) sits there, cleanup running mid-install or between installs leaves partial cache entries -- uv's index points at paths that no longer exist on disk. The next `uv pip install` then fails with errors like:

```
error: Failed to install: cftime-1.6.5-cp312-cp312-...
  Caused by: failed to open file `.../cftime-1.6.5.dist-info/METADATA`:
             No such file or directory (os error 2)
```

Why: hit on 2026-05-28 during rebuild of `/capstor/store/cscs/swissai/a122/IFS/tools/repo-download-ifs/.venv`. Three separate packages (cftime, cffi, ...) had broken METADATA / WHEEL files in the cache.

How to apply:

1. **Move the uv cache off /iopsstor/scratch.** Three lines in `~/.bashrc` (or shell rc):
   ```bash
   export UV_CACHE_DIR=/capstor/store/cscs/swissai/a122/sadamov/uv-cache
   export UV_LINK_MODE=copy
   mkdir -p "$UV_CACHE_DIR"
   ```
2. **When you hit a broken cache**, run `uv cache clean` (wipes everything; next install re-downloads). Or `uv cache prune` for unreferenced/broken entries only.
3. **One-off escape hatch** during a single install: `uv pip install --no-cache -r requirements.txt` -- bypasses the cache entirely. Used 2026-05-28 to recover.
4. `UV_LINK_MODE=copy` silences the "Failed to hardlink" warning when cache and target venv live on different filesystems. Slight extra disk use, no behavioural change.

Applies system-wide (same flavour as [[feedback-cluster-max-resources]] and [[feedback-never-use-tmp-on-hpc]]). Take the same three-line rc fix to any CSCS HPC project.
