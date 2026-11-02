---
name: feedback-check-fast-completed-jobs
description: "When SLURM jobs complete much faster than expected, always check sacct exit code + log tail before treating them as successful"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When a SLURM job finishes much faster than its expected runtime (e.g. expected 2-3h, completed in 6-15 min), do not assume it succeeded just because it left the queue. Always run `sacct -j <id> --format=JobID,JobName,State,ExitCode,Elapsed` and tail the .out/.err logs before reporting status to the user.

Why: On 2026-05-25 I reported "2 of 8 SFNO baseline inference jobs already completed" based on them leaving the queue, when in fact both had FAILED early (NGC FSTimeoutError at 6 min and SIGSEGV at 13 min). The user caught it. Treating "left the queue" as "successful" missed two known transient failure modes (NGC FSTimeout, multi-GPU SIGSEGV) that are documented elsewhere in memory and have known recovery paths. If the user hadn't noticed, the chain trampoline would have fired with 2 weeks of forecasts missing (28 inits) and the downstream eval would have run on an incomplete set, triggering the stale-config gotcha.

How to apply:
- Any time a long-running inference / training job's elapsed time is much smaller than expected, verify with sacct + log tail.
- Especially for SFNO inference jobs (NGC FSTimeoutError is recurring; SIGSEGV is transient but real).
- Especially when reporting progress in status updates to the user -- "X / N completed" must mean genuinely successful, not just "left the queue".
- For chains using `afterany`, fast failures are particularly dangerous because they satisfy deps but produce no outputs. Prefer `afterok` when the chain semantics actually depend on success, OR build the chain to re-verify and retry inside the trampoline body.
- **Also: do not trust the user's "looks good" / "completed" report without verifying with sacct yourself.** On 2026-05-26 the user reported "one just completed all good looking" for 2385273, but sacct showed FAILED with SIGSEGV at 18:41 -- the user had just seen it leave the queue. Same failure mode as before, doubled down by uncritically agreeing. Always sacct first, then respond.
