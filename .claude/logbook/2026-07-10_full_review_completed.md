# 2026-07-10 - resumed 9-dimension pre-submission review COMPLETED; 18 fixes applied, 6 user decisions

One-line summary: the review the user stopped on 07-09 is now complete end-to-end (8 dimensions + 12
referee concerns, all adversarially verified at xhigh); 18 confirmed fix_now findings applied uncommitted
in both repos (paper compiles, ~7,476/7,500 conservative), 6 confirmed findings left as user decisions,
headed by TWO MAJOR Phase-6 refresh-hook seeding defects that affect the noise-model claims around
fig:phase6 (not the frozen headline results).

(paper-process detail kept in the local notes)

## Verdict

Minor revisions territory, converging. No finding invalidates the headline (frozen-configuration)
results: the frozen-arm perturbation code draws independent noise per tensor per member (one
`np.random.default_rng(seed)` stream continuing across the checkpoint walk), and R7's independent
recomputation reproduced every 240h headline gap exactly with an 8/8-block sign consistency. The two
majors are confined to the Phase-6 refresh arm's noise fine-structure.

## MAJOR - user decisions (Phase-6 fresh/refresh hooks, core-code dimension)

Both verified by independent re-trace of `sfno_fresh_perturbation.py`, `aurora_fresh_perturbation.py`,
`aifs_fresh_perturbation.py`, `e2s_inference.py:181-238,488`, and `tools/submit_sfno_p6c_production.sh`.

1. **Cross-member seed collisions.** Generator seed is `int(seed + step)` with seed = CLI seed +
   member_id, so member m at refresh-epoch e+1 and member m+1 at epoch e draw bit-identical noise
   (weights are restored to the shared original each step, so the perturbed model is literally identical,
   applied at a 120h lag). Production refresh-20 (SEED=42, 10 members, 240h): 20 (member,epoch) slots
   served by only 11 distinct draws. SFNO N=1 cadence point: 400 slots by 49 draws. Contradicts main.tex
   L337-339 ("drawn independently ... for every member") and appendix "T/N independent draws".
2. **Identical noise across tensors within a step.** `_seeded_noise_like` re-creates a generator with
   the SAME seed for every tensor, so all 33 Aurora encoder tensors / 32 AIFS decoder tensors / 8 SFNO
   spectral filters get identically-seeded noise per step - whereas the frozen arm draws one continuing
   stream across tensors (independent). The frozen-vs-refresh comparison in fig:phase6 therefore differs
   in cross-tensor noise covariance in addition to the temporal anchor, confounding the pure
   temporal-decorrelation attribution.

   Options (both): (a) fix seeding (single hashed derivation from (seed, param_idx, noise_step); exact
   snippets in the workflow output) + GPU re-run of the three p6c grids and the SFNO cadence sweep -
   re-run cost is the blocker; (b) disclose as an implementation limitation (adds words, needs a named
   trim). A code-only fix without re-run desyncs released code from the reported fig:phase6 numbers
   (1.562->1.111, 2.544->1.682, 0.663->0.764). NOTE: the 06-23 GPU check verified hook fire COUNTS only,
   not seeding - this is new, real, and was found by code reading, not by any statistical anomaly.

## Applied tonight (18 confirmed fix_now findings -> 15 distinct edits + 3 from the earlier partial run)

(paper-process detail kept in the local notes)

CODE repo:
- ai_models_ensembles/e2s_perturbation.py module docstring: IC perturbation correctly described as
  multiplicative `var * (1 + N(0, magnitude))` (was "additive ... scaled by spatial std" - wrong on both
  counts; flagged independently by two dimensions).
- README.md: 5-model -> 7-model registry; stale per-model winners table replaced with the production
  picks (the old table showed GraphCast g2m sigma=0.014 - contradicting the published graphcast_all
  sigma=0.01 pick); superseded 2026-05-27 interim 5-way table dropped; "All five" -> "All"; named-groups
  table gained the `aifs` row.

(paper-process detail kept in the local notes)

## Confirmed but left for the user (6, prioritised)

1. The two Phase-6 MAJORS above (code+rerun vs disclose).
2. (paper-process detail kept in the local notes)
3. No released orchestration for the refresh experiments (sec 3.2.2): the hook modules ARE released and
   env-driven, but no script sets AURORA_FRESH/AIFS_FRESH, and the SFNO ablation cadence sweep has no
   submitter (only tools/submit_sfno_p6c_production.sh). Preferred zero-word fix: add
   tools/submit_refresh_probes.sh (exact env-var recipes in the workflow output) or a README subsection.
   App E's "All ... orchestration ... released" is otherwise slightly overclaiming.
4. W1 subsample seed uses Python's salted built-in hash (wd_kde.py:158,175,197,552,593; PYTHONHASHSEED
   set nowhere) - not process-reproducible, materially nil (pairing preserved within run). Paper side
   fixed tonight; optional SwissClim fix = zlib.crc32 at the five sites.
5. Release tags still missing on both repos (App E promise; text fallback exists but tagging is cleaner).
6. Optional polish: "decided before rounding" caption clause for the bold-tie cells in
   tab:production-metrics / tab:calibration (bolding verified correct at full precision, FCN3 0.8717 vs
   IFS-ENS 0.8746; purely presentational) - mirror in the two assemble_*_table.py generators if wanted.

(paper-process detail kept in the local notes)

## Reviewer-defence ammunition banked (not in the paper, word-margin)

- 8/8-block sign consistency for all four headline gaps (exact p=0.0078 each); smallest gap
  (aifs_perturbed +0.036) per-block range +0.025..+0.048, never near zero.
- Atlas NA-box z500 CRPS ratio 1.031 @240h (mid-pack of 8); Atlas LSD growth broad-spectrum and
  lead-monotone, consistent with SDE-sampler small-scale accumulation, NOT the untraced NA anomaly.
- W1 subsample noise: 2e5-point paired subsample, seed differences immaterial to published digits.

## ATTENDED FOLLOW-UP (2026-07-10, user live on opus) - all 6 open decisions resolved

The user reviewed the report and decided each open item via AskUserQuestion. Actions taken (all
uncommitted):

- **Phase-6 majors -> "fix code + spot-check re-run".** Applied a unified `_mix_seed` (splitmix64 with
  distinct odd multipliers per axis) to all three fresh-hook modules
  (aurora/aifs/sfno_fresh_perturbation.py), replacing the additive `int(seed + step)`. Now the per-draw
  seed is a well-mixed function of (base_seed=42+member, per-tensor/per-module unit_idx, noise_epoch).
  Verified pure-Python over the production ranges (members 0-9, up to 40 units, up to 60 epochs): 24000
  draws -> 24000 distinct seeds, 0 collisions; the old cross-member (m,e)vs(m+1,e-1) and cross-tensor
  collisions are both gone. The three modules py_compile clean. The frozen arm (e2s_perturbation.py) is
  untouched, so no frozen/headline number can move. Bind-mount means the container picks up the edit at
  runtime.
  Spot-check submitted: `tools/submit_sfno_p6c_reseed_spotcheck.sh` (new, based on
  submit_sfno_p6c_production.sh) runs SFNO refresh-20 sigma_N=0.35 with the FIXED seeding on 8
  season-spanning anchor inits (first day of each production week) to a NEW path
  `$STORE/baselines/sfno_p6c_reseed/` (original sfno_p6c untouched). **SLURM jobs 2731579-2731586**
  (8x, PENDING at submit, 240h, one node/4 GPU each, 2h limit). MODE=full in the same script re-runs all
  112 throttled at 12 concurrent if the spot-check moves the number.
- (paper-process detail kept in the local notes)
- **Refresh orchestration -> "add submit_refresh_probes.sh".** New tools/submit_refresh_probes.sh
  (released reproducibility artifact) covering the SFNO ablation cadence sweep N in {1,10,20,40} and the
  Aurora/AIFS refresh-20 two-point probes, with the exact env-var invocations, output under
  $STORE/ablation/phase6_probes/. Syntax-checked (bash -n), not run (documents the probes for a reader).
- **W1 seed -> "fix in SwissClim".** SwissClim_Evaluations/src/swissclim_evaluations/plots/wd_kde.py:
  added `import zlib`; replaced Python's salted `hash(...)` with `zlib.crc32((...).encode())` at all 5
  sites (158/176/198/553/594). py_compile clean under the venv python. Makes the W1 subsample seed
  process-reproducible; immaterial to any published number. (NB: SwissClim is a separate checked-out repo
  inside the tree - its diff is uncommitted too, see git status.)
- **Release tags -> "leave as-is".** No change; user owns the tagging before submission.
- **Bold-tie caption -> "add 'decided before rounding'".** Added the clause to tab:production-metrics and
  tab:calibration captions in BOTH repos, and mirrored the production-metrics string in
  tools/assemble_production_metrics_table.py:509 (the calibration generator emits only the tabular body,
  so its caption lives solely in the hand-maintained .tex - already edited). Caption text word-exempt.

Post-follow-up compile: tectonic PASS, 0 errors, main.pdf 10.09 MiB.

### SPOT-CHECK RESULT (2026-07-10 ~22:05) - the seeding fix MATERIALLY moves the SFNO refresh SSR
Debug packing (job 2731600) hit the SFNO SIGSEGV (3/8); resubmitted one-per-init on normal (2731662-67,
all COMPLETED); 8/8 reseed zarrs. Paired SSR (job 2731786): @240h 7-var mean, same 8 inits,
orig(old seeding) 0.8174 vs reseed(fixed) 0.9172 = +0.10 (+12%), well outside rounding. Real (paired
control isolates seeding). Mechanism: old code gave ~11 distinct draws for 20 member/epoch slots + reused
noise across tensors -> suppressed spread; fix restores independence -> more spread -> higher SSR; for the
underdispersed SFNO (target 1) that is BETTER (full-112 0.76 -> ~0.86 est). So the reported fig:phase6 +
sec 4.3 refresh numbers derive from buggy runs and need regeneration with the fix. Aurora/AIFS (OVERdispersed,
1.11/1.68) must also be re-run - more spread pushes their SSR UP, possibly further from 1, which could
weaken the "refresh pulls overdispersed down toward 1" claim; measure before trusting. Frozen headline
(8-way/Milton/uplift/per-variable) UNAFFECTED (frozen arm seeds correctly). Full decision + invocations in
the handoff. This is a genuine "the majors matter" outcome - the earlier same-day report's optimistic
"probably robust" read was WRONG for SFNO; the fix changes the number (favourably for SFNO, TBD for
Aurora/AIFS).

### (superseded) FOLLOW-UP as originally written when jobs 2731579-2731586 were pending
Compute the 8-init spatial-mean SSR of `sfno_p6c_reseed` (CPU sbatch, adapt tools/submit_sfno_p6c_ssr.sh
to the reseed path / tools/spatial_mean_ssr.py) and compare @240h 7-var mean against the ORIGINAL
`sfno_p6c` restricted to the SAME 8 anchor inits (20230102_0000, 20230402_0000, 20230702_0000,
20231002_0000, 20240102_0000, 20240402_0000, 20240702_0000, 20241002_0000). Decision rule:
- if the reseed 8-init SSR@240h matches the original 8-init SSR@240h within rounding (both ~0.76 region):
  the seeding fix does not move the reported fig:phase6 number -> keep all reported numbers, note the code
  was corrected; done.
- if it moves materially: run MODE=full (all 112, reseeded), recompute the full spatial-mean SSR, and
  update the fig:phase6 numbers (1.56->1.11, 2.54->1.68, 0.66->0.76) + the sec 4.3 refresh CRPSS costs.
  The frozen counterparts (sfno_modes10 etc.) are unchanged and need no re-run.

## Sources verified tonight (per verify-claims rule, before each number entered the paper)

$STORE/baselines/{aifs_perturbed,aifs_p6c}/eval/probabilistic/crps_line_geopotential_500_by_lead_ensprob.csv
(240h rows read directly); tools/submit_sfno_p6c_production.sh (SIGMA=0.35, REFRESH=20, SEED=42);
.claude/logbook/phase6c_generalisation_results.md:32 (unchanged-sigma probe selection); e2s_perturbation.py
:32-50 (multiplicative IC math) and :156-172 (_MODEL_LAYER_GROUPS unet_bottom 494:590); README registry
table (7 models); climatechange.ai/papers/neurips2022/87 (Graubner authors); doi.org/10.21957/e529074162
(ECMWF CY48R1 Part V); milton CSV via the workflow's verified read (Atlas 281.05 vs 283.29).
