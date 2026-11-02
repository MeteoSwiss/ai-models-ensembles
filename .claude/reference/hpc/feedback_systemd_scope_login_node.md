---
name: feedback-systemd-scope-login-node
description: "When running a long-lived Python job from a login node (nohup-style), ALWAYS wrap it in `systemd-run --user --scope` with `-p MemoryMax=NG -p MemorySwapMax=0` so the kernel kills only your process if it overshoots, not other login-node users. Also the right way to set TMPDIR/DASK/SSL via --setenv into the scope."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Why

Shared login nodes on HPC clusters have hundreds of GB of RAM, but it's contended. A Python process that allocates more than the OOM-killer's threshold gets killed by the kernel; under load the kernel may pick a different victim (other users' processes, login-node services). Either way the user pays:

- Their own process dies silently (no traceback in nohup.log, just sudden truncation).
- Other users see the node go unstable / crash.
- Saw it 2026-06-01 on the IFS perturbed-IC download: PID 263264 nohup'd from login node, ran 14h, died mid-write of an 18 GB merge with NO error in the log. Verified dead via `kill -0 263264 -> exit 1`. Cost ~5 days of in-flight progress (resumable, so 78/224 inits preserved, but the next 146 had to wait for a restart).

## Pattern

Wrap any long login-node nohup with a memory-capped systemd transient scope:

```bash
systemd-run --user --scope \
    -p MemoryMax=32G -p MemorySwapMax=0 \
    --setenv=TMPDIR=/iopsstor/scratch/cscs/$USER/dask-tmp \
    --setenv=DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/$USER/dask-tmp \
    --setenv=SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi; print(certifi.where())')" \
    --setenv=REQUESTS_CA_BUNDLE="$SSL_CERT_FILE" \
    --setenv=PYTHONUNBUFFERED=1 \
    nohup .venv/bin/python -u my_script.py [args] \
        >> path/to/nohup.log 2>&1 &
disown
```

`-p MemoryMax=32G -p MemorySwapMax=0` makes the cgroup hard-cap memory. If the process exceeds it, the kernel kills ONLY this cgroup; the login node and other users are unaffected.

Verify after launch:

```bash
PID=$!
cat /proc/$PID/cgroup    # look for the .scope cgroup name
SCOPE=$(cat /proc/$PID/cgroup | head -1 | grep -oE 'run-r[a-f0-9]+\.scope')
systemctl --user show $SCOPE -p MemoryMax,MemorySwapMax
# expect: MemoryMax=34359738368 (32 GiB), MemorySwapMax=0
```

## Picking the cap

- Profile or estimate the peak RSS of one work unit.
- Set MemoryMax to **2x the peak** for headroom. Tight caps OOM the process; generous caps still protect the node.
- The IFS download merge peaks at ~18 GB → 32 GiB cap chosen (~1.8x peak).

## When NOT to use this

Don't use this for sbatch jobs -- SLURM already provides cgroup isolation per job. Use this ONLY for login-node `nohup ... &` style work.

## When to use SLURM instead

If you have node-hours available, prefer sbatch with `--mem=800G --time=12:00:00` (the [[feedback-cluster-max-resources]] default). Login-node-with-cap is for when node-hours are tight and the process is genuinely long-running but light enough to fit in a 32-64 GiB cgroup.

Related: [[feedback-cluster-max-resources]] (sbatch defaults), [[feedback-never-use-tmp-on-hpc]] (TMPDIR redirect inside the scope), [[feedback-ssl-cert-on-hpc]] (SSL_CERT_FILE for compute nodes -- not strictly needed on login but defensive), [[feedback-verify-kill]] (verify the PID after kill, even with systemd scope).
