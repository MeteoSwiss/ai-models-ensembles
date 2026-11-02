---
name: ipc-cleanup-between-rounds
description: SFNO/AIFS round-3 SIGSEGV root-caused to leaked /dev/shm semaphores across mp.spawn rounds within a single python process. Fixed in e2s_inference.py via per-round IPC cleanup (commit d210b9d).
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# IPC cleanup between member rounds (e2s_inference.py)

SFNO and later AIFS hit round-3 SIGSEGV `GPU 0..3 member exited with code -11` even with the per-init week-helper cleanup. Root cause: `mp.spawn` leaks `/dev/shm/sem.{mp-*,pym-*,tmp.*}` files BETWEEN MEMBER ROUNDS inside the same python process (not just across inits). Round 1 -> round 2 inheritance of corrupted semaphore state causes SIGSEGV in round 2 or 3 workers.

**Why:** matches the SFNO state-accumulation pattern documented earlier ([[paper-phase-structure]] cross-ref) but at finer grain. The week-helper `find /dev/shm -delete` only fires between python calls (between inits) - leaked semaphores from rounds within one init still survive and accumulate.

**How to apply:** the IPC cleanup MUST happen between member rounds INSIDE the python process. The week-helper cleanup alone is insufficient for problematic models.

## Fix (commit d210b9d in `e2s_inference.py`)

Inside `_run_members_parallel`, after each round completes, unlink leaked semaphores before starting the next round:

```python
if round_idx < len(rounds) - 1:
    import glob, os
    for pat in ("sem.mp-*", "sem.pym-*", "sem.tmp.*"):
        for f in glob.glob(f"/dev/shm/{pat}"):
            try: os.unlink(f)
            except OSError: pass
    time.sleep(5)
```

Belt-and-suspenders: `scripts/submit_ablation.sh` sbatch wrap also runs the same cleanup before and after the python call (preserving STATUS via $?).

## work_dir cleanup-on-failure
Same commit: wrapped inference call in try/finally so `_e2s_work` is always removed on failure. Prevents the future inode leak that caused a122 to hit 100% quota.

## Related
- [[paper-phase-structure]] SFNO state-accumulation discussion
- [[store-relocation-2026-05-30]] (inode quota was the symptom)
