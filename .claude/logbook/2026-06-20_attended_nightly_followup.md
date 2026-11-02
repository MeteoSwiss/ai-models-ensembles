# 2026-06-20 attended session (Opus 4.8): cleared nightly-flagged items + AIFS architecture verify

One-line: cleared every item the 2026-06-20 nightly e2e review flagged (Atlas cost, AIFS-ENS footnote, FSS
notation, Bodnar2025 bib, calibration caption, preamble TODO); verified the AIFS / AIFS-ENS same-architecture
claim against the source paper and added the matched-pair rationale to the case study. Committed: main
`3f0eaad`, paper `e567c2d`. tectonic clean (main.pdf 18.2 MB, 0 undefined). Both local
only, NOT pushed (push question pending at session end).

## Overnight context (what happened, the overlap question)
- Nightly 2026-06-20 (01:00-01:53, exit=0, sentinel written) ran a 13-agent adversarial e2e review (~1.44M
  tokens), re-confirmed the numerical core clean, committed 5 small fixes (main `e08073a`, paper `51ab037`),
  flagged the Atlas cost as the one open numerical error, and submitted job 2572726 (w1_wdkde).
- OVERLAP / REDUNDANCY: the evening eval job 2570234 (w1_wdkde_baselines) ran 19:58->01:13 and completed all
  6 models (wrote all 6 W1 CSVs). The nightly, at 01:18 (5 min after 2570234 finished), submitted 2572726
  (same job) because it saw `(pending)` cells in the committed Table C1 - it recomputed and overwrote the
  same 6 CSVs (02:02-05:41). Fully redundant, ~4.4h duplicate single-node CPU, deterministic so the CSVs are
  valid, no data risk. The sigk_refresh probe job 2563873 (the "other task") had finished 2 nights earlier
  (Jun 18 21:47), so it never overlapped.
- (paper-process detail kept in the local notes)

## Table C1 W1 column (item already done by a concurrent session)
- HEAD `a9072ab` "Table C1: fill the W1 column (wd_kde re-eval 2570234 complete)" already filled it from the
  evening job. My `tools/assemble_production_metrics_table.py` regen reproduced it byte-identical (CRPSS
  reproduction check worst |diff| 0.0005). No action needed.

## Atlas cost (the one confirmed-real numerical error) - RESOLVED
- The old 43.5 NH was NOT a legacy heuristic (as the nightly guessed): it is the exact sacct sum of the 8
  cleanly-completed `bl_atlas_<date>` jobs (43.54 NH), which cover only ~27 of the 112 production inits.
  Atlas actually has all 112 init forecast.zarr dirs (same as every baseline). So 43.5 undercounts the full
  run ~4x. Measured per-init rate 43.54/27 = 1.61 NH ~= pure-inference floor (CS h 1.58) => Atlas overhead
  negligible (the 2280 s/mbr diffusion sampler dominates).
- FIX (user chose the floor): NH 43.5->177.3 (= 112 x CS h 1.58), CHF 125.3->510.7 (x2.88), kWh
  208.8->851.2 (x4.8). Footnote added: "Atlas NH is its pure-inference estimate (112 x CS h); ... overhead
  negligible." figures/inference_cost_table.tex:60. No prose quoted the old numbers (only s/mbr 38min/18-40x,
  unchanged).

## AIFS / AIFS-ENS same-architecture verification (user-requested)
- Verified vs arXiv:2412.15832 (AIFS-CRPS / AIFS-ENS, Lang et al): "architecture largely follows the
  deterministic AIFS v0.2.1 with the encoder-processor-decoder design"; "16 processor layers, embedding
  dimension 1024, 8 attention heads ... 229 million parameters total"; 6h step; members differ only by
  Gaussian noise via conditional layer-norms in the processor.
- So AIFS and AIFS-ENS share the SAME forward graph (hence identical 75 s/member inference). The 255M (our
  aifs-single-mse-1.1 v1 checkpoint audit) vs 229M (AIFS-CRPS on v0.2.1) gap is a VERSION difference, not
  architectural. My nightly-handoff-driven "different models" footnote was wrong and clashed with the body
  (main.tex:320-326 "shares this architecture, matched pair") -> rewrote the footnote to "shares the AIFS
  backbone ... matched post-hoc-vs-trained pair" + version caveat.
- Added version caveat to the body model-list (main.tex AIFS item: "AIFS-ENS (AIFS-CRPS) is built on the
  earlier AIFS v0.2.1 (229M), the same forward graph at a slightly smaller version than the v1.1 checkpoint").
- Added the matched-pair RATIONALE to the Milton case study ("What IC augmentation buys (AIFS)" paragraph):
  AIFS is the only backbone whose trained-prob sibling runs the same forward graph as the post-hoc ensemble,
  so the contrast isolates probabilistic-generation strategy on a fixed architecture (both 75 s/member).

## Other fixes committed
- FSS notation (main.tex appendix B prose + summary-table formula): $P_{\bar X}$ (ensemble mean) ->
  per-member $P_{X_m}$ averaged over members. Verified vs SwissClim core metrics/fss.py ("per-member ensemble
  mode") + the per_member CSV the assemble script reads.
- SIGK "lead 0 to 240 h": VERIFIED no change needed - forecast.zarr lead_time = [0, 6h, ..., 360h] (n=61),
  starts at 0h.
- Bodnar2025 paper_refs.bib: added Nature vol 641, pp 1180-1187 (Crossref-verified via
  api.crossref.org/works/10.1038/s41586-025-09005-y). Zotero: the same vol/pages should sync on the
  next Better-BibTeX regen (entry already exists; not done here).
- calibration_basis_table.tex caption: dropped "also minimises both multivariate scores" ("both" but there
  are 3 ES/VS/SIGK, and mvar cells are populated for winner rows only so the claim can't be read off the table).
- preamble TODO (main.tex:32-36): trimmed the done table-rebuild + C1-plasma/C2-Milton items; kept Zenodo
  DOIs + release tags + the unconfirmed figure re-render list.

## Environment note
- `$STORE` resolves to the login-node default `/capstor/store/cscs/mch/s83` (WRONG) in a fresh shell; use the
  full path `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles`. Bash on $STORE works fine attended.
