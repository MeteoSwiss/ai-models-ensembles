# Nightly handoff - LATEST (STAGED - move to .claude/handoff/LATEST.md)

Updated 2026-06-14 ~06:35 (unattended nightly, Opus 4.8). The 2026-06-14 01:00 run had died instantly
(claude binary purged by an iopsstor scratch purge; fixed during the day - see logbook
2026-06-14_nightly_scratch_purge_claude_reinstall.md). THIS run is the recovered one and DID complete
the full multi-agent paper review + fixes. Work committed directly to `main` in BOTH repos, NOT pushed.

## .claude WRITES WERE BLOCKED THIS SESSION
Writing under .claude/ hit the sensitive-file permission gate and auto-denied (the settings.json
allow-rule from 2026-06-13 did NOT take effect - both new-file Write AND existing-file overwrite of
.claude/handoff/LATEST.md were blocked). So:
- This handoff is staged at `handoff_staging/LATEST.md` - MOVE IT to `.claude/handoff/LATEST.md`.
- The full review report is staged at `handoff_staging/2026-06-14_paper_e2e_review.md` - MOVE IT to
  `.claude/logbook/2026-06-14_paper_e2e_review.md`.
- Worth checking why the .claude/** allow-rule is not applying to the unattended runner.

## Done this session (committed to main, both repos, verified by recompile)
Ran a 7-dimension multi-agent Workflow review of the whole manuscript (main.tex + 5 figures/*.tex +
paper_refs.bib): 68 agents, 61 findings, 49 confirmed after adversarial verification. Applied 21
high-confidence, reversible fixes. FULL detail + every deferred item is in the staged review report.
Headline highlights:
- FIXED 4 undefined \cite keys that were rendering `[?]` on p.23 (Ullrich2017/2021 -> the v2 keys;
  Knapp2010 -> Knapp2010ibtracs; added @misc{Knapp2018} IBTrACS v4, doi 10.25921/82ty-9e16). The paper
  did NOT compile clean before (LaTeX "There were undefined citations"); it does now.
- FIXED a FABRICATED bib entry: Chen2026atlas claimed "J. Chen / NVIDIA Atlas-ERA5..." but
  arXiv:2601.18111 is Kossaifi et al. (2026) "Demystifying Data-Driven Probabilistic Medium-Range
  Weather Forecasting" (verified via WebFetch). Corrected author+title; kept the key.
- FIXED Milton "medians 1.6-2.7x smaller" -> "1.6-2.0" (stale; true span 1.64-1.97).
- FIXED stale Variable-sets prose (headline Tab+Fig now use all 7 vars incl MSL w/ skipna, not "six,
  MSL excluded"); 4 orphan (never-\ref'd) figures now referenced; VS "strictly proper" -> "proper" in
  body to match App.B; en-dash/spelling/stray-word style fixes; cost-caption 25-40x -> 20-40x.
Tectonic recompile: PASS, no undefined cites/refs, no errors, PDF 7.5 MB. 29 cosmetic box warnings only.

## In flight
- NOTHING submitted to SLURM this session. Queue empty for this project. No background procs.

## Next up (priority order)
1. ZOTERO sync (papers-to-zotero rule): add Knapp2018 (IBTrACS v4, doi 10.25921/82ty-9e16) to
   collection C8CXQJFN, and update the Atlas entry to Kossaifi et al. 2026 (rename key to Kossaifi2026
   on next Better-BibTeX export; ~6 \citep{Chen2026atlas} sites would update automatically).
2. USER FRAMING DECISIONS (deferred, all entangled - see review report sections 1-3): GraphCast
   "coarsest mesh wins" vs all-layers production pick; 3-vs-4 post-hoc baseline scoping; 360h "within
   0.06" qualifier. The user reserved the GraphCast framing on 2026-06-12; agents drafted rewordings.
3. ATLAS COST NORMALISATION (review report #4): s/mbr=2280 is a 61-step/15.25-day rollout while the
   other s/mbr are 10-day; confirm what was actually run, then renormalise s/mbr + "38 min" + slowdown
   factor + GH200/A100 speedup together. TRAINING-COST "20-90K" conflation + FCN3 80K-vs-90K (#5).
4. Submission blockers needing the user: [FUNDING-TODO] (main.tex:1639); abstract ~263 words > AMS 250
   (cut the refresh-every-N sentence 71-77); PNG->PDF vector regen (temporalCRPS x3, spectrogram,
   Milton F1-F5+F8); Zenodo DOIs (after acceptance, phrasing already correct).
5. Code verifications needing GPU/container (carried over): fresh-hook per-AR-step count
   (aurora/aifs/sfno); fresh-GraphCast bit-identical reproducibility.

## Do NOT
- Re-litigate the 12 rejected findings (review report bottom): headline/uplift/Milton/ES-VS cells were
  re-derived and are CORRECT; AIFS 255M vs 229M is two different models, not an error; the 1130 "three
  deterministic backbones" is intentional.
- Do not revert main.tex:1469 "truth-std-normalised CRPS" silently - it was a deliberate 2026-06-12
  edit; confirm intent with the user (flagged in review report #6).
- Push / open PRs (commit straight to main; user decides pushes). >32GB on login node or >12 SLURM
  nodes. AI trailer on paper-repo commits. Delete forecast.zarr or the *_pre_fortinfix archives.

## Notes for the next agent
- This session's tool limits: NO arbitrary python/numpy exec (only `git`, bare-PATH `tectonic`,
  `--version`, grep/glob in working dir). Read works on any path incl /capstor; Grep works on a single
  $STORE file; ls/head/glob on /capstor are gated. Verify numbers by Read+Grep on the CSVs + hand
  arithmetic, or submit an sbatch for heavy recompute.
- tectonic resolves on PATH (0.16.9, capstor conda). Compile: `tectonic <paper>/main.tex --chatter
  minimal --keep-logs`; check undefined cites via `grep "There were undefined" main.log`.
- The review workflow script is at /tmp/paper_review_workflow.js (re-runnable via Workflow scriptPath);
  raw findings JSON was in /tmp (ephemeral) - the staged review report captures all actionable content.
