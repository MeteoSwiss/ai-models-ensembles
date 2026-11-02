---
name: feedback-slurm-cancel-does-not-cancel-mars
description: scancel on a SLURM job that holds a MARS (ECMWF) request does NOT cancel the remote MARS retrieval. The Python client dies locally but the request keeps running at ECMWF and blocks subsequent requests (MARS allows only 1 active task per user).
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When a SLURM job runs `earthkit.data.from_source("mars", ...)` (or any other ECMWF API call) and you `scancel` it, the LOCAL Python process is killed but the REMOTE MARS retrieval keeps running at ECMWF until it completes naturally. The MARS web API tracks requests by account, not by client connection, and there's no auto-cleanup on disconnect.

This matters because:
- **MARS allows only 1 active task per user.** A zombie request from a cancelled SLURM job will block all subsequent requests from queueing into the active slot.
- Your next paranoia / production submit will sit in MARS internal queue for the full retrieval time of the cancelled-but-still-running request before it even reaches the tape.

Hit on 2026-05-28: cancelled paranoia 2418737 (SSL error) and 2418959 (resubmit for unbuffered output) both left MARS requests running. The replacement 2418976 sat queued at MARS for ~17 minutes before becoming active -- exactly because it was waiting for the two zombies to finish.

How to apply:

1. **Before scancel on a job that hit MARS**: explicitly cancel the MARS request via the web UI at `https://api.ecmwf.int/web-api/list-requests`. Find the request ID (logged by ecmwfapi as `Request id: ...`), open it, click cancel.
2. **Or accept the queue wait**: do nothing, your next request will queue behind the zombies and complete later.
3. **For debugging-style restarts (kill mid-flight, edit code, resubmit)**: factor in the zombie cost. If a 20-min retrieval is mid-flight when you cancel, your "fast" resubmit will actually take 20+ min.
4. **One active task per user is a strict cap** -- not a soft suggestion. Plan production downloads as a single long-running process, not many fast ones.

The corollary: **for time-critical loops with MARS**, prefer fewer, larger requests over many small ones. Each request has the overhead of queue + mount + extract, and they all serialize through the 1-active limit.
