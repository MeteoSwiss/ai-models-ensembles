# 2026-08-25 Internal co-author review: triage + response runs launched (incl. duplicate-launcher incident)

One-line: assessed a co-author's 6 technical + 3 positioning review comments against paper and code
(4 valid and actionable, 1 wording bug, 1 already-caveated), launched the GPU/CPU jobs that answer them;
a 3x duplicate-launcher incident forced a full cancel + clean resubmission.

## Verdicts on the review comments
1. IC vs weight attribution: VALID. Paper has weight-only vs weight+IC (fig:milton-wt-vs-ic) but no
   IC-only arm; Limitations explicitly declines it citing Bulte2024. `e2s_inference.py` already supports
   `members=N, ic_perturb=yes, weight_perturb=no`, so the arm is a config change. -> `*_ic_only` runs.
2. Truth-dependent ES/VS/SIGK normalisation: VALID, and it does touch selection (the multivariate bundle
   is the tie-breaker at main.tex:443-449). Scale today = per-init truth spatial std
   (`energy_variogram_score.py:118`, `signature_kernel_score.py:241`). -> fixed 1990-2019 climatological
   scale, new `--scale-json` on both tools.
3. Runner-up configs never validated out-of-sample: VALID. The 112-init grid is a held-out check on the
   winner only. Runner-ups inside the +-0.02 CRPSS band (Tab. calibration): aurora Phase 2 enc s=0.044,
   graphcast m2g s=0.030 + g2m s=0.014, sfno enc s=0.054 + s=0.035, aifs Phase 1 all s=0.01.
4. IFS-ENS skipna numerator vs all-112 denominator: VALID and exactly fixable - the WB2 gaps are whole
   (init, lead) slabs (2026-06-22 audit), so restricting the denominator is a re-average. -> per-init
   climatology dump + `tools/crpss_common_sample.py`.
5. Spread-error / state-dependent uncertainty: VALID, genuine gap (SSR + rank/PIT histograms only).
   -> `tools/spread_error_binned.py` + `tools/plot_spread_error.py`.
6. "only low-l noise contains the overdispersion" (main.tex:903): wording bug, "contains" = holds in
   check, read by the reviewer as "includes". Reword, no science change.
7-9. Positioning (retrofit framing, architecture-specific tuning as a headline finding, 3-question
   story): fair, prose-only, deferred to the writing pass.

## Runs launched
- `scripts/submit_all_inference.sh` extended with 4 `*_ic_only` (IC zarr, no weight noise, 112 inits,
  360 h) and 6 runner-up cells (4 held-out weeks = 56 inits): registries, IC-zarr mount, `DRY_RUN`,
  overridable `LOG_DIR`.
- CPU: `crps_clim_eval` (per-init denominator dump), `per_init_crps`, `spread_error`, `channel_scale`.
- All submit scripts moved off the frozen `a122` account to `ab016` (35 files).
- `python3 -c "datetime.fromisoformat"` in the submit script replaced by `date -d`: the login-node
  system python3 is 3.6.15, so the date arithmetic silently aborted the script.

## Incident: 3 concurrent launchers
Each `ssh <node> 'setsid nohup ... &'` whose CLIENT timed out had still STARTED the launcher remotely.
Three copies ran, every job was submitted 3x, and triplicates of the same init ran concurrently against
the same `forecast.zarr` (2 already FAILED, 0 COMPLETED). Cancelled all 127 `bl_*` jobs plus a watchdog
that re-cancelled for 12 min. Cure: never relaunch a detached remote job after a client timeout without
first checking `pgrep -af` on the target node; the resubmission now runs as a single `flock`-protected
slurm job on a compute node instead of over ssh.

## Environment gotchas hit today
- `/capstor/store/cscs/mch` mount DEAD on clariden-ln001 (reads hang, writes I/O-error), healthy on
  ln002 (~4 s) and on every compute node (instant). Login-node health is per-node.
- CSCS ssh cert expired mid-session -> `Permission denied (publickey)` to all 4 login nodes, so remote
  processes could not be killed. Submitting work as slurm jobs is immune to this.
