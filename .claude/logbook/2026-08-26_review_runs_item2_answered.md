# 2026-08-26 Review response: item 2 answered, IC-only arm validated, campaign launched

One-line: the fixed-climatological rescore changes NO configuration pick (0/24 winner-rival
comparisons flip), the IC-only arm runs clean and shows the spread contraction the paper only cited,
and the 263-job campaign is running after three failure modes were fixed.

## Item 2 (ES/VS/SIGK propriety) - ANSWERED, no paper claim moves
- Fixed scale = mean over 1990-2019 of the cos-lat-weighted spatial std, per (var, level):
  `tools/data/channel_scale_1990_2019.json` (2t 15.21 K, msl 1125 Pa, z500 2734, q850 0.00419).
- `tools/submit_esvs_sigk_fixedscale.sh` scores 4 winners + 6 rivals at 120/240 h BOTH ways
  (80 CSVs). `tools/compare_fixedscale_ranking.py` reports flips.
- RESULT: fixed scale inflates every value ~4-6% and flips NOTHING (0 of 24). The truth-std branch
  reproduces tab:calibration to the printed digits (aurora ES@120 0.710, graphcast 0.761,
  aifs SIGK -17.1), so the rescore is faithful rather than a re-derivation.
- Caveat for the reply: demanding the winner beat EVERY rival fails on SIGK for aurora/graphcast/sfno
  under BOTH scalings (pre-existing, visible in the published table). Against the rivals the paper
  NAMES (GraphCast m2g, SFNO Phase 2b encoder) the winner is ahead on SIGK at 240 h, so the
  manuscript sentence is accurate as written.

## Item 1 (IC-only arm) - RUNS CLEAN, early signal
`sfno_ic_only` 2t member spread: 0.376 K at lead 0 -> 0.250 K at 24 h -> 1.618 K at 240 h
(2023-07-02 00Z; the smoke init 2023-01-02 gave 0.355 -> 0.221 -> 1.711). Non-zero spread at t=0 is
what weight-only cannot produce, and the contraction over day 1 IS the spread-limited behaviour the
paper currently asserts from Bulte2024 alone. Verifier `_smoke_verify.py` fails a run whose members
are identical at lead 0, so a degenerate IC arm cannot pass unnoticed.

## Three failure modes fixed mid-campaign
1. Week-helper invisible in the container: helpers are written to `$LOG_DIR` and read from INSIDE the
   container, but only `$STORE` is mounted. Moving LOG_DIR to iopsstor (yesterday's workaround for the
   dead ln001 capstor mount) killed 15 jobs in 15 s each. `submit_all_inference.sh` now bind-mounts
   LOG_DIR whenever it sits outside `$STORE`. Per-init jobs were immune (they use `--wrap`).
2. `RuntimeError: error during blosc decompression: 0` in `_prefetch_ic_data`: truncated remote ARCO
   reads under ~58-way concurrency. The prefetch exists to avoid exactly this but had no retry of its
   own; now 3 attempts with 30/60 s backoff. IC-only arms are immune (they skip the prefetch and read
   the local IC zarr).
3. GH200/UCX `SIGSEGV (-11)` in all 4 GPU workers at startup: the built-in single retry rescued the
   smoke test but under load both attempts can land in the same bad window. Round retries 1 -> 2
   (`E2S_ROUND_RETRIES`).

## Self-inflicted regression (cost 6 jobs)
The first retry edit used `os.environ` inside `_run_members_parallel`, which already has a LATER
`import os` for the /dev/shm cleanup - a function-scoped import makes the name local for the whole
function, so the earlier use raised `UnboundLocalError`. Every job starting in the ~4 min window
19:04-19:08 died. Rule: code that 250 queued jobs will execute gets an import + statement-order check
BEFORE the edit is left in place, the same way the post-processing tools were tested.

## Cluster context
`maint_2026_08_26` 07:00-19:00 (1370 nodes) held the whole queue; work started when it lifted at 18:45.
Yesterday's capstor flakiness on ln001/ln003 was the run-up to it. All submit scripts moved a122 -> ab016.
