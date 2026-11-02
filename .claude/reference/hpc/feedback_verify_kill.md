---
name: feedback-verify-kill
description: "Always `ps -p $PID` AFTER kill to verify the right process actually died. PID files and `pgrep` can return stale or alternate IDs after setsid/nohup; an unverified `kill -TERM` against the wrong PID lets the real process keep running silently."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When stopping a long-running background process started with `setsid nohup ... &`:

1. The PID captured in `$!` from bash is often the parent (the setsid wrapper or the immediate child); the actual python may have a different PID after detachment.
2. `pgrep -f "python ..."` can return MULTIPLE matches if the same command line is reused across launches; the PID file might point at a stale or wrong one.
3. `kill -TERM <stale_pid>` exits 0 if the PID doesn't exist (no error), so the script silently does nothing.

ALWAYS verify after killing:

```bash
kill -TERM "$PID"
sleep 2
if ps -p "$PID" > /dev/null 2>&1; then
    kill -KILL "$PID"  # escalate
fi
# Final verification:
ps -p "$PID" -o pid,etime,cmd 2>&1 | head -3   # MUST show nothing or just header
pgrep -fa "<command pattern>"                  # cross-check for any matching process
```

Caught 2026-05-29 on the IFS download: the original 09:49 launch (PID 96119) was meant to be killed at 12:09 but `kill -TERM` was issued against the stored PID file value (which had been overwritten with a different launch's PID). 96119 stayed alive and burned MARS quota on 16 failed inits over 10 hours before being noticed at 20:03. After `kill -KILL 96119` + `ps -p 96119`-verifies-empty, that loop was finally closed.

Lesson: NEVER trust that a kill succeeded without `ps -p $PID` after. Treat the PID file as a hint, not authority -- cross-check with `pgrep -fa <pattern>` and verify each kill.
