---
name: feedback-unbuffered-stdout
description: "System-wide rule: ALWAYS unbuffer Python stdout in long-running scripts. Use `python -u` in bash launcher / SLURM job, or `print(..., flush=True)` on every progress print, or `sys.stdout.reconfigure(line_buffering=True)` at script top. Without it, SLURM .out / nohup logs stay empty until the process exits or the buffer fills (4096 bytes by default when stdout is a file), and a stuck or slow job is indistinguishable from a working one until the wall expires."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Rule

For any Python script that runs longer than ~30 seconds and whose stdout is being redirected to a file (SLURM .out, `nohup`, systemd-scope, `>`), force unbuffered output. One of these MUST be present:

1. **`python -u script.py`** in the launcher (bash, sbatch, srun, systemd-run). Cleanest -- no source edits needed.
2. **`print(..., flush=True)`** on every progress line in the script. Safer when you don't control the launcher.
3. **`sys.stdout.reconfigure(line_buffering=True)`** as the first runtime line. Affects all subsequent prints in the script.
4. **`PYTHONUNBUFFERED=1`** in the environment. Equivalent to `-u`.

Pick at least one. For SLURM jobs that are long, expensive, or whose progress you'll want to monitor mid-run, **pick all of `-u` AND `flush=True`**, because someone (you, in three hours) will be tailing the log and asking "is this thing actually running."

## Why

Python's default stdout buffering is **line-buffered when stdout is a TTY** and **block-buffered (4096 bytes) when stdout is a file or pipe**. SLURM, nohup, systemd-run, and bash redirection all turn stdout into a file. So a script with `print("[i/N] done")` that runs fine when you test it interactively goes silent the moment you submit it -- the prints sit in the buffer until either:

- 4096 bytes accumulate (often many minutes for a low-rate progress log), or
- the process exits and the buffer flushes.

If the process exits cleanly the log lands; if it hits the wall and gets killed mid-buffer, you lose every line that was buffered. Either way you can't tell the difference between "stuck at init 50" and "working hard on init 50" in real time.

## How to apply

- **Always** prepend `-u` to the python invocation in sbatch scripts and bash launchers.
- For scripts in repos under your control, also add `print(..., flush=True)` on the heartbeat lines, since `-u` is easy to forget on someone else's machine.
- For systemd-run / nohup wrappers, `-u` lives in the wrapped command:
  ```bash
  systemd-run --user --scope ... bash -c 'python -u my_script.py >> my_script.log 2>&1'
  ```
- For interactive Python on Clariden / Cray nodes (rare), `sys.stdout.reconfigure(line_buffering=True)` works.

## Real-world precedent

- 2026-06-04: SLURM job 2467627 (`persistence_mae_112inits`) ran for 2 h on `compute_persistence_mae.py` and TIMEOUT'd. The log file had four lines of FutureWarnings from import time and nothing else, because every `print(f"Processing {key}...")` call sat in the 4096-byte buffer that never filled before the wall expired. Could have been a stalled job, a slow job, or genuine progress -- impossible to tell from the log. With `-u` or `flush=True` the log would have shown "Processing 2m_temperature... Processing mean_sea_level_pressure..." and we'd have known immediately whether to extend the wall or kill and restart.

## Anti-pattern

```bash
sbatch --wrap "python long_script.py"     # silent log, no way to monitor
```

```python
print(f"[{i}/N] done")    # buffered when stdout is a file
```

## Correct

```bash
sbatch --wrap "python -u long_script.py"
```

```python
print(f"[{i}/N] done", flush=True)
```

```python
import sys
sys.stdout.reconfigure(line_buffering=True)
print(f"[{i}/N] done")
```
