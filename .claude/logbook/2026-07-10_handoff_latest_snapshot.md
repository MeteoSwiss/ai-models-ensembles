# Handoff - LATEST (2026-07-10, review COMPLETE + all decisions resolved) - 1 SLURM spot-check in flight

The full 9-dimension AIES review is done AND the user (attended, live) resolved all 6 open decisions.
Everything is applied UNCOMMITTED across three repos. One bounded SLURM spot-check is in flight; its
result gates whether the fig:phase6 numbers need updating.

Full detail: `.claude/logbook/2026-07-10_full_review_completed.md` (read the "ATTENDED FOLLOW-UP" section
first). Base commits unchanged: code `5f91d06`, paper `5c249e9`, SwissClim at its own HEAD.

## IN FLIGHT - SLURM spot-check (the one thing to follow up)

**Reseed grid DONE: 8/8 forecast.zarr present** in `$STORE/baselines/sfno_p6c_reseed/` (corrected
fresh-hook seeding). The 5 one-per-init resubmits (2731662/63/65/66/67) all COMPLETED ~8 min each after the
debug packing left 3/8 (see LESSON below).

**SSR comparison DONE (job 2731786, COMPLETED 18 min): the seeding fix MATERIALLY moves the SFNO refresh
SSR.** Paired 8-init spatial-mean SSR @240h (7-var x 2-level mean), identical 8 anchor inits, only the
seeding differs:
- orig (old seed+step): 0.8174
- reseed (fixed _mix_seed): 0.9172  -> Delta +0.10 (+12%), well outside rounding.
CSVs: `/iopsstor/scratch/cscs/sadamov/tmp/ssr_{reseed,orig}_8init.csv`; log `.../reseed_ssr_cmp_2731786.log`.
Real effect (paired control isolates seeding; far larger than any run-to-run noise). Mechanism: the old
code collapsed ~20 member/tensor draws to ~11 and reused noise across tensors, suppressing ensemble spread;
correct seeding restores independence -> more spread -> higher SSR. For the UNDERdispersed SFNO (target 1)
that is BETTER calibrated (full-112 0.76 -> ~0.86 estimated).

=> The "keep reported numbers" branch is RULED OUT. fig:phase6 + sec 4.3 refresh numbers need regeneration
with the corrected seeding. **USER DECISION (GPU + paper edit):**
1. SFNO: `MODE=full bash tools/submit_sfno_p6c_reseed_spotcheck.sh` (all 112, one-per-init, throttled 12),
   then rerun the SSR (tools/submit_sfno_p6c_ssr.sh pointed at sfno_p6c_reseed) -> new SFNO refresh SSR.
2. Aurora + AIFS p6c: MUST also be re-run with the fixed seeding (their refresh runs used the same buggy
   hooks). CAUTION - they are OVERdispersed (refresh 1.11 / 1.68); more spread from correct seeding pushes
   their SSR UP, i.e. potentially FURTHER from 1 (could weaken the "refresh pulls overdispersed down toward
   1" story). Needs measuring before trusting the current 1.56->1.11 / 2.54->1.68. No Aurora/AIFS reseed
   submitter exists yet - adapt submit_sfno_p6c_reseed_spotcheck.sh or use the AURORA_FRESH/AIFS_FRESH env
   vars (see tools/submit_refresh_probes.sh).
3. The refresh CRPS-cost numbers (sec 4.3, the -0.025/-0.017 CRPSS + 1-5% costs) also derive from the buggy
   runs - re-derive after the reseed grids land.
The frozen headline (8-way, Milton, uplift, per-variable) is UNAFFECTED (frozen arm seeds correctly). Only
the Phase-6 refresh excursion (fig:phase6 + sec 4.3) is in question. Everything else from tonight stands.

### 2026-07-11 ~06:50 - REDO RESULT IS A MAJOR FINDING (see logbook 2026-07-11_reseed_ssr_major_finding.md)
The corrected-seeding reseed grids show the paper's Phase-6 claim is a BUG ARTIFACT for the over-dispersed
backbones. Spatial-mean SSR @240h: Aurora frozen 1.56 -> refresh 1.11(buggy)/**1.43(fixed)**; AIFS 2.54 ->
1.68(buggy)/**2.50(fixed)**, and AIFS is HIGHER than buggy at EVERY lead (worse at short leads). SFNO
(under-disp) 0.66 -> 0.76(buggy)/**~0.92(fixed)**. So refresh reliably helps only the UNDER-dispersed SFNO;
the dramatic "pulls over-dispersed Aurora/AIFS down toward 1" was the old seeding suppressing spread. This
FALSIFIES the blanket fig:phase6/sec-4.3 claim -> USER DECISION on how to reframe (do NOT rewrite blind).
Frozen headline UNAFFECTED. Eval state: Aurora/AIFS SSR done, CRPS running (2733344/45, 112-init configs);
SFNO 109->112 (3 resubmits 2733398-2733400), SSR+CRPS auto-fire at 112 (watchers b2witahjn/b0yhblkq1).
NB caught + cancelled an accidental full eval matrix (ran evaluate_baselines.sh with no args = "all", 138
jobs); fixed a python3->date-d bug in evaluate_baselines.sh too. Full detail in the logbook entry.

### REDO IN PROGRESS (2026-07-10 ~22:25) - proper launcher + validation wave live

Launcher (NEW, tracked): `tools/submit_p6c_reseed.sh <sfno|aurora|aifs> [spot|full]` - one unified script,
per-model settings taken VERBATIM from the original bl_<model>_p6c logs (sfno: 444G/arco/sigma0.35/modes10;
aurora: 800G/arco/sigma0.025/encoder; aifs: 800G/**CDS**/sigma0.0275/decoder; all refresh-20, 10 mbr, 240h,
seed42). ONE SBATCH PER INIT (SFNO SIGSEGV-safe), global throttle keeps ALL p6reseed_* jobs <= MAX_CONCURRENT
(default 12) nodes. Writes to `<model>_p6c_reseed/` (originals untouched). Per-init runtime: SFNO ~8min,
Aurora ~2h10, AIFS ~2h (mem 800G each). (Supersedes the sfno-only submit_sfno_p6c_reseed_spotcheck.sh.)

AIFS LAUNCHER FIX (was failing fast, NOT freezing): the first AIFS spot attempt (2731840-2731852) FAILED
in ~1 min with "Missing/incomplete configuration file: /users/sadamov/.cdsapirc" - my submitter never
mounted the CDS creds. Fixed per scripts/submit_ablation.sh: mount `.cdsapirc`+`.ecmwfapirc` to BOTH `~`
and `/root`, AND point AIFS at the PERSISTENT `$STORE/e2s_cache_backup` (5376-file CDS cache, survives the
scratch purge) instead of the scratch e2s_cache. Also mem=800G for ALL models now (user: all can use 800G).
Re-launched AIFS validated: job 2731866 hit the cached CDS ICs and ran members (no auth error, no 2h fetch).

WARM-CACHE SPEED CHANGES EVERYTHING: with the caches warm, each init is ~8 MIN, not the cold-cache ~2h the
originals took (that was mostly ARCO/CDS fetching). So the full 112x3 redo is ~3.5h at 12 nodes, NOT 35h.
Aurora full-validated: 8/8 spot COMPLETED, forecast.zarr 100% finite, 2t@240h ensemble std 2.06 K (fresh
hook working). SFNO reseed 8/8 done earlier.

FULL SEND (user: "full send") - launched, hit a DATE BUG, fixed, relaunched:
- BUG: MODE=full built init dates with `python3 -c ...` which is NOT on PATH in a detached background
  shell, so dates came out EMPTY -> malformed inits `T00:00`/`T12:00` -> ~10 aurora jobs crashed fast with
  "Invalid isoformat string: 'T12:00'" (job names `p6reseed_aurora__1200`). FIXED in submit_p6c_reseed.sh:
  use GNU `date -d "${ws} + ${off} days"` (coreutil, always on PATH) + a generation guard (exit if date
  not YYYY-MM-DD) + a submission-time guard (skip any init not matching the full ISO pattern). Cleaned up:
  cancelled the 10 malformed jobs, killed the lingering feeders (their in-memory INITS were malformed), and
  rmdir'd the 2 empty garbage dirs (aurora_p6c_reseed/_0000,_1200 - no forecast.zarr, safe). The 8 valid
  spot inits per model were untouched.
- RELAUNCHED (verified well-formed, 0 malformed): aurora feeder bg6wlr8c3, aifs feeder bsgg0fyhi, sfno
  feeder bn3ocbjrm - each `submit_p6c_reseed.sh <m> full`, 112 inits, skip-exists keeps the 8 spot inits,
  shared global throttle 12 nodes. Completion watcher b2118o3rl -> reports all three at 112/112 + failures.
- Cache split (user asked): sfno/aurora -> scratch arco cache (arco is public + re-fetchable, no creds);
  ONLY aifs -> persistent $STORE/e2s_cache_backup (CDS is creds-gated + slow). Do NOT point sfno/aurora at
  the $STORE backup - it has no arco ICs.
To re-launch a single grid: `bash tools/submit_p6c_reseed.sh <sfno|aurora|aifs> full`.

DETACHED for session-independence (user wanted to close VSCode/SSH): the three session-tied feeders were
killed and replaced by a SINGLE `setsid nohup` driver (PID 80500, PPID=1, own session) running
`for m in aurora aifs sfno; do bash tools/submit_p6c_reseed.sh $m full; done`, log
`/iopsstor/scratch/cscs/sadamov/tmp/p6c_reseed_driver.log` (scratch may purge the LOG but the setsid
process survives on the login node - verify with `ps -p 80500` / `du` on the reseed dirs, not just pgrep).
A duplicate-guard (skip if a p6reseed_<model>_<tag> job is already queued) was added to the submitter so
the driver doesn't double-submit in-flight inits. This makes the INFERENCE session-independent; it will
finish ~3.5h regardless of the SSH/VSCode session. Session watchers (b2118o3rl etc.) die on close - ignore.

EVAL AUTO-QUEUED (user kept the session; two session watchers submit per grid the moment it hits 112/112):
1. Spatial-mean SSR (fig:phase6): watcher b2witahjn -> `sbatch tools/submit_p6c_reseed_ssr.sh <model>` (NEW
   script, CPU sbatch; parameterised clone of submit_sfno_p6c_ssr.sh pointing at <model>_p6c_reseed).
   Output `<grid>/spatial_mean_ssr/spatial_ssr.csv`.
2. CRPS/CRPSS (sec 4.3 costs): watcher b0yhblkq1 -> `bash scripts/evaluate_baselines.sh <model>_p6c_reseed
   probabilistic`. Registered the 3 reseed grids in evaluate_baselines.sh (MODELS + MODEL_KIND=ai + case);
   config-gen runs at 112 so it captures all inits. Output `<grid>/eval/probabilistic/`.
Config-gen deliberately deferred to 112 (NOT run now) so it does not cache an 8-init config.
AFTER both land (check `<grid>/spatial_mean_ssr/spatial_ssr.csv` + `<grid>/eval/probabilistic/`):
  paired compare reseed-112 vs ORIGINAL <model>_p6c (SSR@240h; and refresh-vs-frozen CRPS costs), then
  regen fig:phase6 (tools/plot_refresh_ssr.py - repoint at *_reseed), update sec 4.3 numbers, sync, recompile.
Spot-check already showed SFNO SSR moves +0.10 (0.82->0.92); the KEY unknown is whether Aurora/AIFS (over-
dispersed) move the WRONG way (SSR up, away from 1) - the full-grid SSR answers that.
NEW/edited files this session (uncommitted): tools/submit_p6c_reseed.sh, tools/submit_p6c_reseed_ssr.sh (new);
scripts/evaluate_baselines.sh (+3 reseed models); tools/submit_sfno_p6c_reseed_spotcheck.sh (superseded, can rm).

EVAL after each grid lands (CPU sbatch, one node each, NOT the full 8-module suite): only
(1) spatial-mean SSR (tools/spatial_mean_ssr.py / submit_sfno_p6c_ssr.sh pattern -> fig:phase6) and
(2) CRPS/CRPSS (eval/probabilistic -> sec 4.3 costs). Per-pixel SSR + LSD/FSS/SSIM/ES/VS/SIGK/W1/maps/
spectra/bivariate are NOT needed (those are frozen-baseline metrics, unaffected by the seeding bug).

RERUN SCOPE = Aurora + AIFS + SFNO p6c grids ONLY. GraphCast is correctly EXCLUDED (verified 2026-07-10):
(1) it is not in fig:phase6 (plot_refresh_ssr.py reads aurora/aifs/sfno only); (2) its refresh mechanism is
JAX activation perturbation in graphcast_coarse_perturbation.py, a different file, NOT the three torch
fresh-hook modules that had the bug; (3) its seeding is collision-free (fresh = hk.next_rng_key() Threefry
chain; frozen = one jax.random.PRNGKey(seed+member_id) per member, Threefry-initialised so 42/43/... are
independent, not additive-colliding) and it perturbs a SINGLE activation slice so the cross-tensor reuse
(Major 2) cannot occur. The GraphCast activation control 0.27/0.15 (sec 4.3) stands as-is.

LESSON (debug attempt, job 2731600, COMPLETED but 3/8): packing all 8 inits SEQUENTIALLY into ONE job
re-triggered the known SFNO multi-GPU multiprocessing SIGSEGV (signal 11 on the parallel member spawn;
init 1 ran clean in a fresh process, later inits segfaulted as multiprocessing/semaphore state
accumulated, retries failed). The SFNO SIGSEGV escape hatch is ONE SBATCH PER INIT (process isolation) -
same reason the production `submit_sfno_p6c_production.sh` and `PER_INIT=1` exist. Do NOT pack multiple
SFNO inits into one job, on debug or anywhere. The tracked `tools/submit_sfno_p6c_reseed_spotcheck.sh`
(one-per-init, normal) is the reliable path; the ephemeral debug script
`/iopsstor/scratch/.../submit_sfno_reseed_debug.sh` is a dead end for SFNO (throwaway, not tracked).

WHEN THEY LAND: compute the 8-init spatial-mean SSR@240h (7-var mean) of `sfno_p6c_reseed` (CPU sbatch;
adapt `tools/submit_sfno_p6c_ssr.sh` to the reseed path, or `tools/spatial_mean_ssr.py`) and compare to
the ORIGINAL `sfno_p6c` restricted to the SAME 8 inits (20230102_0000, 20230402_0000, 20230702_0000,
20231002_0000, 20240102_0000, 20240402_0000, 20240702_0000, 20241002_0000).
- MATCHES within rounding (both ~0.76 region) -> seeding fix does not move the reported number; keep the
  fig:phase6 numbers, done.
- MOVES materially -> run `MODE=full bash tools/submit_sfno_p6c_reseed_spotcheck.sh` (all 112, throttled
  at 12 concurrent), recompute the full SSR, and update fig:phase6 (1.56->1.11, 2.54->1.68, 0.66->0.76) +
  the sec 4.3 refresh CRPS-cost numbers. Frozen counterparts unchanged, no re-run needed.
Check `sacct -j 2731579-2731586` / `squeue --me | grep p6reseed`.

## Decisions resolved this session (all applied uncommitted)

1. Phase-6 majors -> FIX + spot-check: `_mix_seed` (splitmix64) in the 3 fresh-hook modules replaces
   additive `seed + step`; verified collision-free (24000 draws -> 24000 distinct). Frozen arm untouched.
   Spot-check submitted (above).
2. (paper-process detail kept in the local notes)
3. Refresh orchestration -> added `tools/submit_refresh_probes.sh` (released reproducibility artifact for
   the SFNO cadence sweep + Aurora/AIFS refresh-20 probes; syntax-checked, not run).
4. W1 seed -> `zlib.crc32` at the 5 wd_kde.py sites in SwissClim (process-reproducible; immaterial).
5. Release tags -> LEAVE AS-IS (user owns tagging before submission; App E wording unchanged).
6. Bold-tie caption -> added "decided before rounding" to tab:production-metrics + tab:calibration
   captions (both repos) + mirrored in assemble_production_metrics_table.py.

Plus the 18 review fix_now findings applied earlier tonight (see the logbook report's "Applied tonight").

## Uncommitted diffs (nothing committed; user reviews + commits)

CODE repo (MeteoSwiss):
- ai_models_ensembles/{aurora,aifs,sfno}_fresh_perturbation.py (seeding fix), e2s_perturbation.py
  (IC docstring), README.md (7-model registry + production picks), figures/{calibration_basis_table,
  experiments_tables,milton_summary_table,table_c1_production_metrics}.tex (synced captions),
  tools/assemble_production_metrics_table.py (caption mirror).
- NEW (untracked): tools/submit_refresh_probes.sh, tools/submit_sfno_p6c_reseed_spotcheck.sh.
PAPER repo (sadamov): main.tex, paper_refs.bib, figures/{calibration_basis_table,
  experiments_tables,milton_summary_table,table_c1_production_metrics}.tex.
SwissClim repo (separate, inside the tree): src/swissclim_evaluations/plots/wd_kde.py.

tectonic PASS (0 errors), main.pdf 10.09 MiB. (paper-process detail kept in the local notes)

## Next up

1. USER: review + commit the three repos' diffs. Tag both repos at the reported-results revision before submission (decision 5).
2. Follow up on jobs 2731579-2731586 (the SSR comparison above) - the last thing gating fig:phase6.
3. Then the presubmission checklist has no remaining nightly-safe items; the rest is user-owned
   (tags, Zenodo-on-acceptance) or attended (figure regen class).

(Nightly review-run budget section: paper-process detail kept in the local notes)

## Standing constraints

No auto-commit/push in nightlies; never delete forecast.zarr/data/memory/`sfno_p6c` (the reseed writes to
a NEW path); sbatch only after checking not-already-done (done: squeue was empty, reseed path was absent);
<=12 nodes (spot-check used 8; MODE=full throttles to 12). Do NOT re-add Rasp2020/aifs_perturbed_ic to the
headline; do NOT revert the settled items (1.93 refresh clause, three-backbones headline, 07-06 pooled bivariate, 07-07 geostrophic 35N, significance wording). If any Agent/Workflow call
errors with a session-limit message: STOP, no relaunches (global rule in ~/.claude/CLAUDE.md).

## Notes for the next agent

- Reconcile against `git log` + `git status` in ALL THREE repos first (code, paper, SwissClim).
- The seeding fix is bind-mounted into the SFNO container at runtime (submit scripts mount
  $SRC_DIR/ai_models_ensembles), so 2731579+ already use the corrected code.
- tectonic recipe: `source /capstor/store/cscs/mch/s83/sadamov/miniforge3/etc/profile.d/conda.sh &&
  conda activate base && export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())') &&
  cd ai-models-ensembles-paper && tectonic -X compile main.tex`.
- Read tool reaches /capstor absolute paths; Bash `ls` on $STORE subdirs is sandbox-blocked but stat via
  Read works. Workflow full finding evidence:
  `/iopsstor/scratch/.../43729201-.../tasks/wgymak6sr.output` (scratch purges; the logbook report carries
  everything load-bearing).
