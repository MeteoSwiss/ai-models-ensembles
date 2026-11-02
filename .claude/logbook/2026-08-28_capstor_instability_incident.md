# 2026-08-28 capstor instability: three lost gap-fill passes (evidence for a CSCS ticket)

One-line: `/capstor/store/cscs/mch` has been cycling between healthy and unusable since ~2026-08-27
17:30, destroying three gap-fill passes; the review-response campaign is 36 forecasts short because of
it, not because of anything in the code.

## Symptom
A directory glob over 112 init dirs (`ls -d $STORE/baselines/sfno_ic_only/*/forecast.zarr`) oscillates
between 0 s and unusable:
  08-27 17:2x   >90 s (timeout)          08-27 18:25  0 s
  08-27 18:35   "Cannot send after transport endpoint shutdown" (mid-submission)
  08-27 19:0x   >45 s (timeout)          08-27 23:55  170 s
  08-28 00:20   0 s x10 consecutive      08-28 04:36  30 s
Overnight probe log: 57 slow probes out of ~480, best stable streak 36 probes (~18 min).
Also seen on the login node: `pwd: write error: Input/output error`, and `rm` failing with
"Cannot send after transport endpoint shutdown".

## Impact on jobs
GPU jobs hang in the "preparing data + model" phase (they read checkpoints + the IC zarr from capstor)
for 45-90 min against a normal ~10 min total runtime, then fail or hit the walltime:
  pass 1 (08-27 14:00): 24 jobs -> 12 FAILED, 10 TIMEOUT at 1h30m, 0 COMPLETED
  pass 2 (08-27 18:35): SFNO stage never submitted - `rm -rf _e2s_work` failed mid-script
  pass 3 (08-28 00:20): submitted during a 10-probe healthy window; failures resumed by 04:36
Jobs die before writing, so NO forecast.zarr was corrupted (counts unchanged across all three passes).

## A trap this creates for status checks
`timeout N ls ... | wc -l` returns a PLAUSIBLE WRONG NUMBER when the glob is killed mid-stream: wc
counts the partial list and exits 0. This produced a false "aifs_all_s010 25/112" (truth: 112/112,
confirmed by `find`, 12 s). Use `find <dir> -maxdepth 2 -name forecast.zarr -type d | wc -l` and treat
any count taken during a slow window as unverified.

## Code fixes that came out of the failures (keep regardless)
- `_prefetch_ic_data`: 3 retries, and after them the failure is NON-FATAL - one undecodable ARCO chunk
  used to kill a whole 14-init week-helper. Members then fetch their own inputs.
- `submit_all_inference.sh`: `rm -rf _e2s_work` is now `|| true`; under `set -e` a capstor hiccup during
  cleanup aborted an entire submission stage.
- Round retries 1 -> 2 (`E2S_ROUND_RETRIES`) for the GH200/UCX startup SIGSEGV.
- Deleted a ZERO-BYTE entry in the ARCO fsspec cache (`e2s_cache/arco/436a855c...`, written 08-26 19:03
  during the failing campaign). A truncated cache write is decompressed on every subsequent read ->
  deterministic "error during blosc decompression: 0" that no retry can win. This is the best
  explanation for graphcast_m2g/g2m being stuck on the same 2024-01-02 week across three passes;
  UNTESTED because the resubmitted job has sat PENDING on cluster priority for 9+ h.

## Where the campaign stands
Complete 112/112: aurora_ic_only, graphcast_ic_only, aifs_ic_only, aurora_enc_s044, aifs_all_s010.
Short: sfno_ic_only 105, sfno_enc_s054 101, sfno_enc_s035 108, graphcast_m2g 98, graphcast_g2m 98.
`ic_decomp` + `per_init_rivals` are queued behind the 24 outstanding jobs (dependency repointed after
the first set died). Decision pending with the user: wait for the filesystem, or relax
`--require-inits` and analyse on 98-105 paired inits (still >> the 4-init ablation basis).
