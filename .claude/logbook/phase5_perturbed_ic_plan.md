---
name: phase5-perturbed-ic-plan
description: "Phase 5 = real IFS-ENS perturbed analyses as per-member IC + ORIGINAL weight sigma, all 4 perturbed baselines, full 112 inits, no sweep. Decided 2026-06-08. Implementation scoped; --ic-source wiring does NOT exist yet."
metadata:
  node_type: memory
  type: project
  originSessionId: 2af74a68-4741-4543-85e8-2c21803bb2f7
---

# Phase 5: perturbed-IC (hybrid) baselines

## Decided design (2026-06-08, user)
- **All 4 perturbed baselines**: graphcast_all, aurora_encoder, sfno_modes10, aifs_perturbed.
- **Config = real IFS-ENS perturbed analysis as per-member IC + ORIGINAL weight sigma** (graphcast all@0.01, aurora encoder@0.025, sfno modes10@0.25, aifs decoder@0.0275 - see [[calibration-winners]]). Only ONE variable changes vs the production baseline: the IC source.
- **CRITICAL physics (user, 2026-06-08): IC sigma and weight sigma are NOT additive, they are lead-time complementary.** IC spread dominates days 0-3 then DECAYS (AI models are too dissipative to grow IC perturbations like real chaos). Weight spread is INSUFFICIENT for SSR=1 in the first few days (a small structural perturbation needs rollout steps to accumulate) but stays relevant / sufficient at long lead - which is exactly where the original sigma was tuned (240h). So IC + original-sigma is NOT an over-dispersion trap: IC adds ~nothing at 240h (decayed) and fills the short-lead underdispersion that weight-alone can't. The combination can give SSR(lead) ~ 1 across the WHOLE forecast, which neither source does alone. **This is the hypothesis to test; we keep both.**
- **Full 112 inits immediately** (inference is cheap; eval is the bottleneck).
- **Zero sigma sweeps.** If over-dispersed, ONE analytic correction via variance decomposition (below), not a sweep.

## Implementation: DONE 2026-06-08 (`--ic-zarr` flag, unit-tested on host)
The per-member real-IC wiring now exists end-to-end (was the "gotcha"; ~80% reused from the existing `_ifs_analysis_source` plumbing). Changes:
- **e2s_data.py**: new `ifs_ens_member_ic_source(zarr_path, member_id, cached_ds=None)` -> `from_swissclim(open_zarr(PATH)[.isel(lead_time=0)].isel(ensemble=member_id))`. Handles BOTH pure-analysis (no lead_time, the perturbed-IC zarr) and forecast-layout (lead_time=0, the ifs_ens stand-in) stores. `cached_ds` reuses the opened lazy store across members. Added to `__all__`.
- **e2s_inference.py**: `ic_zarr` param threaded through `run_inference` -> `_run_members_{sequential,parallel}` -> `_gpu_worker` (added to the positional `worker_args` tuple) -> `_data_for_member`. When set it overrides base arco/cds AND `ic_magnitude` for the IC; weight perturbation stays orthogonal in `_model_for_member`, so `--ic-zarr ... --weight-magnitude <orig>` = the hybrid. `_prefetch_ic_data` is skipped when `ic_zarr` is set (IC is local). The sequential `use_shared_model` opt (weight_magnitude<=0) is CORRECT for IC-only: same weights, different IC per member.
- **cli.py**: `--ic-zarr PATH` option on `infer`, passed to `run_inference`, echoed in the banner.
- **tests/test_repository_smoke.py**: `test_infer_help_has_ic_zarr` + `test_ifs_ens_member_ic_source` (synthetic SwissClim zarr; asserts member k - member 0 == k and multi-time serving). All 7 pass.
- Verified on the relocated `ifs_ens.zarr` stand-in: z500/t850/q500 finite, members differ (max|z500| 210.9), single + multi-time serving work. (msl all-NaN there = the known [[ifs-ens-msl-bug]], NOT the wiring.)

## MARS-shortNames blocker: AUTOMATED FIX queued 2026-06-08 (chained behind download)
The completed `ifs_analysis_perturbed_ic.zarr` stores vars as MARS shortNames (`z,t,u,v,q,w` x13 levels + `2t,10u,10v,100u,100v,msl,sp,tcwv`). Surface `2t,10u,10v,100u,100v` would NOT map to `t2m,u10m,v10m,u100m,v100m` via `from_swissclim`. Solved with a metadata-only rename to SwissClim long names (chosen over a MARS branch -- also makes the store match the format my unit test passed against).
- **`tools/rename_ic_to_swissclim.py`** (in repo, committed-worthy): renames all 14 MARS dirs -> SwissClim long names in place (LocalStore dir rename + rebuild root consolidated_metadata; NO data copy). Idempotent. zarr 3.1.6 `Group.move` is NotImplementedError, so it uses `os.rename` + drop stale `consolidated_metadata` + `zarr.consolidate_metadata`. Tested on synthetic v3 zarr (rename + idempotent re-run + from_swissclim tokens verified).
- **Supervisor `/iopsstor/scratch/cscs/sadamov/phase5_ic_postprocess.py`** chains the whole tail: waits for download (`[224/224]` in log + no `download_ic_perturbed` proc + z@500 finite x224) -> runs `fill_interp_levels.py` (150+600) -> runs the rename -> verifies `from_swissclim` serves z500/z600/t2m/u10m/t850/q500 finite & members differ -> writes sentinel `/iopsstor/scratch/cscs/sadamov/phase5_ic_postprocess.DONE`.
- **Launched 2026-06-08 ~21:05 on clariden-ln003**, scope `phase5_ic_post.scope` (32G), bash PID 34914, log `/iopsstor/scratch/cscs/sadamov/phase5_ic_postprocess.log`. **After-break**: check the `.DONE` sentinel + the log's "ALL DONE" line. If it failed, the log says which step (FILL/RENAME/VERIFY).

## Still TODO after the sentinel lands
1. **submit_all_inference.sh**: add a `perturbed_ic` run variant per model = original sigma/layer flags + `--ic-zarr $STORE/.../ifs_analysis_perturbed_ic.zarr`, output to `baselines/<model>_ic/`. Init grid: the zarr has 224 6-hourly inits; each 00/12Z baseline init is covered with its own + t-6h.
2. **Per-model var/level coverage check**: confirm each model's input vars/levels are a subset of the 14 vars / 13 levels.
3. **Smoke test** (1 init x 10 members per model) inside the container: members differ + fields physical + MSL finite.

## EXECUTED 2026-06-09: Phase 5 AIFS launched, 3 bug fixes during smoke

**AIFS chosen first** (user pick) -- IFS-ENS IC zarr matches AIFS's native preprocessing, AIFS uses all 13 standard ERA5 levels natively so no `fill_interp_levels` needed, fastest iteration. Sequence and bugs fixed:

1. **Supervisor crash from 2026-06-08 21:04 cleared manually**: phase5_ic_postprocess.py died with `IndexError: Index 500 is out of bounds` (used `isel(level=500)` instead of `sel`). Sentinel never written. Ran `tools/rename_ic_to_swissclim.py` directly -- all 14 MARS shortNames -> SwissClim long names. fill_interp_levels NOT NEEDED for AIFS (it uses native 13 levels including 150 & 600).
2. **Smoke 1 (2501720) failed**: `KeyError: "Variable 'd2m' not present in XarrayDataSource"` -- the MARS download by design only carried the 14 atmospheric vars EDA actually perturbs per member; surface/boundary fields (d2m, soil temp/moisture, snow, sea-ice, SST, land-sea mask) come from the deterministic background and are shared across members anyway.
3. **Fix A: `MergedDataSource(primary, fallback)` in e2s_data.py**. Tries the IC zarr first per variable, falls back to the base CDS source on `KeyError`. The base CDS cache at `/iopsstor/scratch/cscs/sadamov/e2s_cache/cds/` (44 GB) is already populated from the production aifs_perturbed runs so the fallback fetches are FREE (no new CDS calls). `_data_for_member` builds a `build_data_source(base_source)` and passes it as `fallback_source=` to `ifs_ens_member_ic_source`.
4. **Smoke 2 (2501760) failed**: `ValueError: Required dimension variable not found in the required index 3 in dim list ['batch', 'variable', 'lead_time', 'time', 'lat', 'lon']`. The actual data had `variable` at index 1 but AIFS expects it at index 3 (after `time`). Inspected `earth2studio.data.CDS.fetch_cds_dataarray` and `aifs.AIFS.output_coords` in the container: CDS returns `(time, variable, lat, lon)`, our `XarrayDataSource` returns `(variable, time, lat, lon)`, and AIFS's expected order is `(batch, time, lead_time, variable, lat, lon)`.
5. **Fix B: `MergedDataSource.__call__` transposes to canonical `(time, variable, lat, lon)`** matching CDS's order. Only applied in MergedDataSource (XarrayDataSource left alone since other models that use it don't fail).
6. **Smoke 3 (2501773) PASSED**: 1m29s wall, no tracebacks, output zarr written, 2m_t max|diff| 6.91 K and z@500 max|diff| 895.6 m^2/s^2 between members at lead 24h -- real IC-perturbation-driven spread.
7. **First full launch (8 jobs 2501778-86) used CHAIN=1 default + sequential ICs**: I'd copy-pasted the production aifs_perturbed defaults without thinking. CANCELLED.
8. **Fix C: stratified IFS-ENS subsample**. User flagged that IFS-ENS members are NOT generic random draws (EDA + singular-vector combinations differ per member, adjacent members can share EDA centroids), and the rest of the paper uses **`[0, 5, 10, ..., 45]`** stride-5 for the Milton case study + headline IFS-ENS subsample. Phase 5 should match for paper consistency. Code in `ifs_ens_member_ic_source`: `stride = max(1, n_ens // 10); actual_ensemble = (member_id * stride) % n_ens`. Smoke-tested: m=0..9 -> IFS-ENS [0,5,...,45], direct readout matches `from_swissclim`-served values bit-exactly.
9. **Re-launched 2026-06-09 05:25 as 2501792-99: CHAIN=0 + stratified ICs**. 8 independent week-helpers, all 800G, 4 GPU each. Expected ~2 h wall (vs ~16 h serial). Output: `$STORE/baselines/aifs_perturbed_ic/<init_tag>/forecast.zarr`.

After AIFS Phase 5 completes: evaluate via `evaluate_baselines.sh ... aifs_perturbed_ic`, then regenerate the headline + Tab. 4 (the table generator should pick up the new baseline path same as existing aifs_perturbed). Lead-resolved SSR is the key Phase 5 diagnostic; if SSR(lead) ~ 1 across all leads while CRPS is held, Phase 5 is a win. Then move to GraphCast (next baseline). Aurora + SFNO follow.

## Execution sequence
1. [prereq] IC download 224/224 (in flight, ~2h on ln004, see [[ifs-ens-ic-experiment]]) + run `fill_interp_levels.py` for 150 + 600 hPa.
2. [VERIFY - the one real risk] each model's required input vars/levels are a subset of the zarr's {T/U/V/Q/Z/W on 13 pressure levels} + {MSL, 2T, 10U, 10V, 100U, 100V, SP, TCWV}. GraphCast/Aurora use 13 levels (1000..50) -> matches. SFNO + AIFS channel sets need explicit checking. This is what could turn "cheap" into "more work".
3. [implement] the --ic-zarr per-member wiring.
4. [smoke] 1 init x 10 members per model: confirm members DIFFER (different IC member), fields physical (T~250K@500, MSL~101kPa).
5. [run] full 112 x 10 x 4 models inference.
6. [light eval FIRST] **LEAD-RESOLVED** SSR (not lead-averaged) + spatial-mean SSR + CRPS. The target signature is SSR(lead) ~ 1 from day 0 to day 15. Reference curves: weight-only (under early, ~1 late) and, optionally, IC-only (~1 early, under late) - the two brackets that prove complementarity.
7. [decide] SSR(lead) near 1 across leads + spatial-mean spread collapsed toward trained-prob -> run FULL eval suite ONCE. A residual short-lead gap (model damps BOTH sources early) is a FINDING to report, not a knob - bumping weight sigma to fix day 1 would over-disperse day 10.

## No analytic sigma correction (the additive formula was WRONG)
The earlier "sigma^2_IC = sigma^2_total - sigma^2_weight, then scale" idea assumed additivity. User corrected 2026-06-08: the two sources are NOT additive and occupy different lead bands, so there is no single-lead variance budget to solve. sigma stays at the original; the experiment IS the combination. The [[hybrid-ic-weight-plan]] additive predictions are superseded by this lead-complementary view.

## Paper framing
4 new rows (or a Phase-5 column in the ablation/headline). Expected result: CRPS roughly held, **spatial-mean spread collapses toward trained-prob** (the cure for the [[spatial-mean-vs-pointwise-ssr]] 9x/6x coherent-shift artifact in graphcast_all/aurora_encoder). Headline: "real IFS-ENS IC perturbation removes the coherent-whole-field-shift artifact of weight perturbation while holding pointwise CRPS." Limited paper space -> keep to one compact table block.
