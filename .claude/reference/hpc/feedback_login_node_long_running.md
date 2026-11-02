---
name: feedback-login-node-long-running
description: "For long-running login-node jobs on Clariden, use `setsid nohup CMD < /dev/null > LOG 2>&1 & disown`. Trust ppid=1 + ps -p etime, not pgrep. Don't restart a working process just because pgrep returns 0 transiently."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Long-running login-node jobs (Clariden, sadamov, Linger=no)

**Rule:** When the user vetoes SLURM and wants a long-running job on the login node, the only launch pattern that survives Claude Code's bash session lifecycle is:

```bash
setsid nohup CMD < /dev/null > LOG 2>&1 &
disown
```

**Why:** setsid creates a new session (decouples from any terminal/parent process group). nohup makes the new process ignore SIGHUP. `< /dev/null` closes stdin so no upstream pipe death propagates. `&` + `disown` removes the job from the bash job table so the bash that launched it doesn't care if it survives.

After launch, the PID's `ppid` becomes `1` (reparented to init) -- that's the success indicator.

## What does NOT work

- `mv DIR1/ DIR2/` for a 20 TB cross-project Lustre directory inside a foreground `Bash` tool call. The tool moves it to a background command after ~2 min and reaps it on a longer timeout. Killed after ~2.7 TB / 67 min.
- `systemd-run --user --scope`. Despite being a "transient scope" it terminates with the spawning shell's process group. Dead after ~5.4 TB.
- `systemd-run --user --unit=NAME` (a transient SERVICE -- different from scope). Works in principle but if you re-launch a redundant one alongside an already-working setsid rsync, the redundant service rsync exits and the unit gets reaped from `systemctl --user list-units`. Not the launch pattern's fault, but watch out: a redundant service launch will appear to "go missing" later.

## False-negative trap: pgrep / `ps -ef | grep`

A working setsid'd login-node job had `ppid=1`, `ps -p <PID> -o etime` showing 2h+ alive, AND `pgrep -u $USER -af "rsync.*ifs_ens"` returning **0 matches**. The rsync was very much alive (du was decreasing on source, increasing on dest, log file mtime was advancing). My panic restart was wasted effort.

Plausible causes for the false negative: transient `/proc` visibility hiccup under heavy I/O, very long cmdline truncated past grep pattern, or my own grep ending up in the cmdline regex via shell quoting.

**Rule:** Before pronouncing a long-running process dead and relaunching it, cross-check with at least TWO of:
1. `ps -p <known-PID> -o etime,stat,comm` (specific PID -- the most reliable signal)
2. Log file mtime advancing over a 30-60 sec window
3. The work product (source `du` decreasing, dest `du` increasing, output file growing, etc.)

A redundant restart is not free: it can race against the survivor on shared resources (rsync with `--remove-source-files` survived this race OK, but the next thing might not). See [[ifs-zarr-reshard]] for the 2026-06-07 rsync-restart loop where this rule was learned.

## Related

- [[feedback-verify-kill.md]] -- the inverse rule: ALWAYS `ps -p $PID` AFTER `kill` to confirm before assuming the job died.
- [[feedback-systemd-scope-login-node]] -- the MemoryMax=NG safety wrapper rule (orthogonal: that one is for memory containment, this one is for survival).
- [[feedback-cluster-max-resources]] -- SLURM defaults (the alternative when login-node won't do).
