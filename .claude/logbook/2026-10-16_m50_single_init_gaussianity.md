At M=50 every ensemble (SPW, trained, IFS-ENS) is measurably non-Gaussian at single gridpoints of one init; the M=10 null result was test power. SPW T2m is more Gaussian than trained/IFS, SPW WS10 less.

# 50-member single-init Gaussianity case (2026-10-15/16, talk context, paper untouched)

Trigger: Slack thread with a colleague. They asked whether the distribution induced by weight perturbations at one
output gridpoint for one IC is non-Gaussian (pooling over ICs gives a Gaussian mixture, so it can't answer that). 10
members could not show it (see `2026-10-14_ens_anom_gaussianity.md`, `member_shapiro_rate`).

Runs: all 7 baselines, 50 members, 240 h, init 2023-10-05 00 UTC, production recipes and seed 42, jobs
3670515-21 (normal, ab016, PER_INIT). Wall: GraphCast 0:44, FCN3 1:44, the other SPW/AIFS-ENS ~1 h, Atlas 6:15
(13 rounds x ~25 min). SFNO hit the usual round-3 SIGSEGV, recovered by the built-in retry.
- `scripts/submit_all_inference.sh` env overrides `INITS LEAD_HOURS NUM_MEMBERS OUT_ROOT` (defaults unchanged).
  Forecasts written to iopsstor scratch (no capstor write stalls, no 1.4M transient member inodes on s83),
  then copied by a CPU job with per-file size check (3670535, 3.9 min, 31 files/model).
- Durable copy: `$STORE/baselines_m50/<model>/20231005_0000/forecast.zarr` (~6 node-h of GPU; do not delete).
- Extraction: `compute_point_ensemble.py --root ... --inits 20231005_0000 --tag m50` (3670536) ->
  `point_zurich_m50.npz`; `compute_ens_anomalies.py --baselines all ifs_ens --root ... --tag sw_m50 --p-only`
  8192 points (3670537) -> `<model>_sw_m50.npz`. Plots `submit_ens_anom_plots.sh --case` (3670538).
- Figures: `figures/ens_anom/member_shapiro_rate_m50_20231005_0000`, `point_member_qq_zurich_20231005_0000_240h_m50`,
  `point_members_zurich_20231005_0000_240h_m50`. Code+figures commit `9c38cce`.

Reproducibility check: members 0-9 bit-identical to production for SFNO and Aurora. AIFS SPW and AIFS-ENS differ:
6 h mean |dT2m| ~0.002 K vs spread 0.2-0.4 K, growing chaotically to 0.12 / 0.47 K mean at 240 h. Same noise
draws, GPU nondeterminism in AIFS (same class as the GraphCast/JAX note in CLAUDE.md), not a seeding bug.

Shapiro-Wilk share non-Gaussian at BH FDR 5% over 8192 points, 240 h (T2m / Q850 / Z500 / WS10, %):
- Aurora SPW 2.5 / 39 / 8.7 / 12.3; GraphCast SPW 6.4 / 24 / 3.2 / 14.1; SFNO SPW 5.8 / 45 / 8.5 / 9.4;
  AIFS SPW 0.8 / 20 / 2.4 / 11.7
- AIFS-ENS 17.7 / 38 / 10.6 / 6.2; Atlas 15.0 / 34 / 5.4 / 6.3; FCN3 8.0 / 39 / 6.9 / 5.9
- IFS-ENS 50 20.7 / 55 / 8.4 / 11.3; IFS-ENS 10 at the same init 0.0 / 2.8 / 0 / 0.2 (the power gap)
- Lead dependence: SPW start at ~0 at 24 h (linear-response regime) and grow; trained ensembles are already
  non-Gaussian at 24 h in T2m (AIFS-ENS 3%) and Q850 (~10%).

Zurich Q-Q at 240 h (40 panels, BH): WS10 rejects for every model except AIFS-ENS (right skew, bounded at 0;
Aurora strongest, p=5e-6). T2m rejects for GraphCast SPW, SFNO SPW, AIFS SPW, FCN3 and IFS-ENS 50; SFNO SPW shows
a cold-member cluster like IFS-ENS's. Z500 only IFS-ENS. ERA5 climatology (30 yr) Gaussian everywhere.

Caveats: one init; the 8192 points are spatially correlated, so the share is not 8192 independent samples.
