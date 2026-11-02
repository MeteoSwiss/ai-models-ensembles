---
name: feedback-no-bashx-for-submit-scripts
description: "Never use `bash -x` to \"preview\" or \"dry-run\" a script that contains sbatch / git push / curl / other side-effecting calls. -x traces execution, it does not prevent execution."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When asked to plan a submission, write the plan in conversation or commit script changes, but do not run the script under `bash -x` as a "preview". The `-x` flag prints commands AS THEY EXECUTE -- it does not skip them. Same applies to `set -x` inside an interactive bash. Same caveat for `set -v`.

Why: on 2026-05-27 I ran `WEEKS=2023-01-02 bash -x scripts/submit_all_inference.sh aurora_encoder graphcast_g2m` to "trace what would happen". It submitted 3 real sbatch jobs (bl_aurora_encoder + bl_graphcast_g2m for week 2023-01-02). I had to scancel and clean partial state. The user had explicitly said "don't launch anything yet".

How to apply:
- If a script has its own dry-mode (e.g. `bash scripts/foo.sh dry`), use that.
- If the script lacks a dry-mode, the safe trace options are:
  - Read the script and explain the planned actions in conversation.
  - Add a temporary `DRY_RUN=1` guard with `[[ -n "$DRY_RUN" ]] && { echo "[DRY] sbatch ..."; continue; }` before sbatch lines.
  - Use `bash -n` for syntax check (does not execute).
- Once you understand the script, point to the relevant lines/diff in conversation. Do not execute it.
- This is a stricter version of [[feedback-check-fast-completed-jobs]] -- there I fail on the side of trusting "left the queue" = success; here I fail on the side of "trace" = "no action".
