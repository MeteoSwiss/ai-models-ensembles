# ai-models-ensembles memory index (snapshot 2026-11-02, handover)

Sanitised copy of the per-user Claude Code auto-memory index (`~/.claude/projects/<project>/memory/MEMORY.md`)
at handover. Policy: the auto-memory is a pointer + in-flight tracker only; durable rules live in `CLAUDE.md` /
`.claude/rules/` / `.claude/reference/`, findings in `.claude/logbook/`.

## State at handover
- Paper submitted (AMS AIES; resubmitted after the technical check 2026-09-18), no reviews yet. Release
  `v1.0.0` = `69e3f9f`. Everything after it on `main` is post-submission material or handover work.
- Nothing in flight after the 2026-11-02 jobs (ACL grant 3833539, rank histograms 3833540, fixed-scale
  ES/VS/SIGK 3833541, wind-frame PDFs 3833547). Outputs: `$STORE/analysis/{rank_histograms,table_metrics_fixedscale}`.
- Post-submission work (not in the paper): IFS-ENS line in Fig. 4 (2026-10-07), weight-only AIFS column in
  Milton F1, GraphCast refresh run (`baselines/graphcast_p6c_reseed`, SSR 3.20 -> 2.97 at 240 h), unperturbed
  AIFS talk line (`aifs_det`), member-distribution diagnostics incl. 50-member single init (`baselines_m50`).
- Known revision items: `docs/EXPERIMENTS.md` "Known open issues" (Whitaker sentence, GraphCast refresh claim,
  Milton F3/F4 below 300 dpi, sigma 0.01-0.03 has no cited source, two missing code guards).

## Load-bearing guards (do NOT undo)
- Milton IFS-ENS detection = 71% (not 14%; the 14% was a double member filter in `aggregate_tracks.py`).
  `logbook/2026-09-02_milton_ifsens_detection_double_filter_bug.md`.
- Never regenerate Tab. C1 / the calibration table wholesale without the fixed-scale ES/VS/SIGK CSVs
  (`$STORE/analysis/table_metrics_fixedscale/`, regenerated 2026-11-02); the generators refuse otherwise.
- Never use `lsd_bug_impact.csv` (LSD of the mean spectrum understates the SwissClim member-0 bug).
  `logbook/2026-09-17_swissclim_member0_fix_and_bib_corrections.md`.
- Refresh clause `\sigma_N\le1.93` (paper main.tex ~729; data `$STORE/diagnostics/sfno_phase6_ssr/`, the
  plotting script was removed in `ba8f855`) - do not "correct" it to 1.58.
- After any figure change diff the regenerated legend against the committed PDF (Fig. C5 once silently
  dropped IFS-ENS).
- Do not add `aifs_perturbed_ic` to the headline ("three deterministic backbones" is intentional).
- Aurora PSD spikes = patch-grid harmonics amplified by the decoder group only.
  `logbook/2026-09-10_psd_layer_group_patch_harmonics.md`.
- Scratch purges strip files but leave dirs: count files, never trust `ls`. Durable outputs: `$STORE/analysis/`,
  `tools/data/`. Verify copies with `stat`, not `du -sh`.
- tectonic writes `main.bbl` only with `--keep-intermediates`; assert `\bibitem` count == cited count.
- IFS-ENS is in Fig. 4 since 2026-10-07 (WB2 gaps are whole random slabs, fair per variable); ES/VS/SIGK/W1
  still exclude it. `logbook/2026-10-07_ifs_ens_spatial_mean_ssr_fairness.md`.

## Session log (one line each; detail in the named logbook entry)
- 2026-11-02: handover. Docs restructured (`docs/`), dataset inventory tool, pinned environments
  (host lock, container freezes, Dockerfile commit pins), 60 MB of figure inputs vendored and verified
  byte-identical, IFS paths -> ab016, account override, ACLs for s83, memory split public/local.
- 2026-10-16: 50-member single-init Gaussianity case. `logbook/2026-10-16_m50_single_init_gaussianity.md`.
- 2026-10-14/15: member-distribution diagnostics, geopotential kept in m2/s2. `logbook/2026-10-14_ens_anom_gaussianity.md`.
- 2026-10-13: unperturbed AIFS talk line. `logbook/2026-10-13_aifs_det_talk_crpss_line.md`.
- 2026-10-07: GraphCast weight refresh (talk only); IFS-ENS line in Fig. 4. `logbook/2026-10-07_*.md`.
- 2026-10-06: no cited source for sigma 0.01-0.03; Whitaker sentence wrong (local notes).
- 2026-09-17/18: SwissClim member-0 fix, LSD refresh of Tab. 5 / Fig. C3, scratch-purge guard.
- 2026-09-01/02: Milton pipeline regenerated after the purge; IFS-ENS detection fixed back to 71%.
- 2026-08-25..30: internal review response runs (IC-only arm, rivals, fixed-scale rescore, spread-error).
- 2026-07-10/11: Phase 6 seeding bug fixed, reseed runs show refresh barely moves Aurora/AIFS.
- Before 2026-07: ablation phases 1-6, production grid, Milton case study (logbook `phase*_*.md`).
