Summary: ran the missing GraphCast weight-space refresh (graphcast_all, refresh-every-20, sigma=0.01, 112 inits) for a talk only: 240 h spatial-mean SSR 3.20 -> 2.97, same "barely moves" verdict as Aurora/AIFS; paper untouched, its "architecture excludes the weight-space refresh" line is overstated.

# GraphCast weight-space refresh (2026-10-07, attended, talk-only)

## Why it was missing
- 2026-06-11 scope decision ("NO GraphCast Phase 6c run", `phase6c_generalisation_results.md`), justified as
  "architecturally excluded". That only holds for a scale-restricted weight subset (GNN weight sharing across
  mesh levels, why Phase 3 went to activation space). graphcast_all is frozen noise on all 264 tensors and
  refreshes like the others. Paper main.tex:769 overstates it; logged in the author's private notes.
  User: do NOT change the paper for now; results go into a talk only.

## Implementation
- `ai_models_ensembles/graphcast_fresh_perturbation.py`, gated by `GRAPHCAST_FRESH=1`,
  `GRAPHCAST_FRESH_SIGMA`, `GRAPHCAST_FRESH_REFRESH_EVERY`. Wraps `model.run_forward` and overrides
  `params=` per call: earth2studio passes params into the jitted apply via `functools.partial`, so a same-shape
  pytree does NOT recompile (verified: step time flat 1.3-1.4 s/it across the step-21 resample).
  Noise `w*(1+sigma*N)` per leaf, seed `_mix_seed(seed+member, leaf, epoch)` like the other fresh hooks.
  264 float leaves / 36.0M params = same set `_perturb_npz` perturbs for `--layer all`.
- Launcher: `tools/submit_p6c_reseed.sh graphcast full` (MODEL_ID=graphcast_operational, arco, 240 h).
  SSR: `sbatch tools/submit_p6c_reseed_ssr.sh graphcast`. Plot: `tools/plot_refresh_ssr.py --with-graphcast --out ...`.

## Run
- Debug test 3604209 (4 mbr, 1 init) ok. Grid: 112/112 COMPLETED first try, 12.5-14 min/init, 26.2 node-hours.
  SSR eval 3605741 1:47 h. Output `$STORE/baselines/graphcast_p6c_reseed/` (+ `spatial_mean_ssr/spatial_ssr.csv`).
- False alarm worth remembering: the 4-member test showed 2x the frozen global-mean spread at early leads; with
  10 members it was 11.1 vs 7.8 (F-test p~0.15) and pointwise spread matched (32.2 vs 31.4). All frozen inits
  reuse the same 10 draws (seeds 42-51), so cross-init agreement of frozen stats is NOT independent evidence.

## Result (7-variable mean spatial-mean SSR, 112 inits)
| lead | frozen | refresh-20 |
|---|---|---|
| 24 h | 1.23 | 1.35 |
| 72 h | 2.45 | 1.87 |
| 120 h | 3.13 | 2.35 |
| 240 h | 3.20 | 2.97 |
- 240 h spread ratio refresh/frozen median 0.71 = 1/sqrt(2), exactly the two-epoch random-walk expectation
  (Aurora 0.80, AIFS 0.77). Spread falls as theory says; a 3x excess just cannot be fixed by it. Per-variable
  SSR is noisier than spread because the spatial-mean error of the ensemble mean depends on the draw set.
- IC augmentation (3.20 -> 1.90) remains the GraphCast spatial-mean fix. CRPSS: see the section below.

## Talk material
(talk-slide detail kept in the local notes)

## CRPSS frozen -> refresh (7-var mean, WB2 clim denominator, 112 inits; same recipe as the headline figure)
| model | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| Aurora | 0.785->0.775 (-0.010) | 0.633->0.607 (-0.025) | 0.471->0.439 (-0.032) | 0.134->0.104 (-0.030) |
| AIFS | 0.790->0.791 (+0.002) | 0.677->0.680 (+0.002) | 0.542->0.547 (+0.005) | 0.221->0.228 (+0.007) |
| SFNO | 0.726->0.740 (+0.015) | 0.556->0.575 (+0.019) | 0.388->0.409 (+0.021) | 0.125->0.131 (+0.006) |
| GraphCast | 0.749->0.745 (-0.004) | 0.628->0.618 (-0.010) | 0.489->0.472 (-0.016) | 0.167->0.150 (-0.017) |
- GraphCast per-variable CRPS @240h: z500 +6.6%, t500 +8.4%, msl +3.7%, z850 +3.3%, u/v +2-3%, q ~0, t850 -0.3%,
  2t -3.1%. Eval job 3609399 COMPLETED 05:04 (2026-10-08 01:02).
- Frozen values reproduce the paper's 240 h gaps to AIFS-ENS (0.036/0.089/0.122/0.131).
- GraphCast refresh CRPS: `bash scripts/evaluate_baselines.sh all graphcast_p6c_reseed probabilistic` -> job
  3609399, output `$STORE/baselines/graphcast_p6c_reseed/eval/probabilistic/crps_line_*_by_lead_ensprob.csv`.
- Found + fixed (`8f7add5`): ba8f855 single-quoted the WB2 truth paths in evaluate_{baselines,ablation}.sh, so
  any config generated after it had a literal `${AIENS_WB2_22:-...}` truth path. First submission 3609394
  cancelled while pending; older on-disk configs predate the bug.
