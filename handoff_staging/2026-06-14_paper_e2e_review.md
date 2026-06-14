# 2026-06-14 full end-to-end paper review (7-dimension multi-agent workflow) + 21 high-confidence fixes applied

NOTE: this belongs in .claude/logbook/ but .claude writes were permission-blocked this unattended
session (sensitive-file gate). Move it to .claude/logbook/2026-06-14_paper_e2e_review.md.

Nightly unattended run (Opus 4.8). Ran a Workflow-orchestrated review of the whole manuscript
(main.tex + 5 figures/*.tex tables + paper_refs.bib): 7 parallel dimension reviewers (numerical
accuracy, internal consistency, citations, metric/formula correctness, style, narrative red-line,
figures/tables) each fanned out to adversarial verifiers that re-opened the source before confirming.
68 agents, ~2.6M subagent tokens, 462s. 61 findings raised, 49 confirmed after adversarial
verification, 12 rejected (mostly correctly-verified table cells + false-positive "inconsistencies").

Verification harness note: this session was locked down (no python/numpy exec, no ls/head/glob on
$STORE). Agents verified numbers by Read + single-file Grep on the $STORE CSVs and the in-repo
climatology JSONs, doing the arithmetic by hand. The committed headline + calibration + Milton tables
were re-confirmed against source; the body-text claims derived from them were the main yield.

## FIXED this session (high-confidence, reversible, committed to main)

Citations (were SUBMISSION-BLOCKING: 4 undefined \cite keys rendered `[?]` on p.23; LaTeX log
"There were undefined citations"):
- main.tex:919 `\citep{Ullrich2017,Ullrich2021}` -> `{UllrichZarzycki2017,Ullrich2021tempestv2}`.
- main.tex:928 `\citep{Ullrich2021}` -> `{Ullrich2021tempestv2}`.
- main.tex:930 `\citep{Knapp2010,Knapp2018}` -> `{Knapp2010ibtracs,Knapp2018}`.
- paper_refs.bib: added `@misc{Knapp2018}` = IBTrACS v4 dataset (Knapp, Diamond, Kossin, Kruk, Schreck
  2018, NOAA NCEI, doi 10.25921/82ty-9e16). VERIFIED via NCEI/web. ZOTERO PENDING (papers-to-zotero
  rule: add to collection C8CXQJFN by hand; API not used unattended).
- paper_refs.bib `Chen2026atlas` metadata was FABRICATED: it claimed "J. Chen / NVIDIA Atlas-ERA5:
  Stochastic-interpolant probabilistic weather forecasting" but arXiv:2601.18111 is actually
  Kossaifi et al. (2026) "Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting"
  (VERIFIED via WebFetch of arxiv.org/abs/2601.18111; that paper benchmarks ATLAS-SI/ATLAS-CRPS, so
  the arXiv id is correct). Fixed author->`J. Kossaifi and others`, title->the real one. Kept the KEY
  `Chen2026atlas` (renaming would touch ~6 \citep sites); it now renders "(Kossaifi et al., 2026)".
  Prose "NVIDIA Atlas-ERA5 \citep{Chen2026atlas}" still correct (model = Atlas, paper = Kossaifi).
  ZOTERO PENDING + key-rename-to-Kossaifi2026 left for the user's next Better-BibTeX export.

Numerical:
- main.tex:967 Milton "Medians are 1.6-2.7 times smaller than means" -> "1.6-2.0". Stale weight-only-era
  number; the current Phase-5 milton_summary_table mean/median ratios span 1.64 (gc) to 1.97 (aifs),
  max nowhere near 2.7. Re-derived all 8 rows.

Consistency / style:
- main.tex:488-491 Variable-sets paragraph said headline Tab+Fig "use six variables, MSL excluded ...
  re-aggregation left for the next revision" - STALE. Both generators (headline_8way_table.py,
  plot_headline_crpss_vs_lead_8way.py) now include MSL (VARS_2D has mean_sea_level_pressure) and both
  captions say 7 vars + skipna. Rewrote to "use all seven, with IFS-ENS surface fields (T2m, MSL)
  handled by skipna averaging because of the WB2 archive NaN gaps (T2m ~20%, MSL ~30%)".
- inference_cost_table.tex caption "$\sim$25-40$\times$" -> "$\sim$20-40$\times$" to match body
  (892) + conclusions (1317). True span is ~18-39x (2280 s/mbr vs 59-125); "20-40x" is the rough
  band already used in 2 of 3 places. (Deeper Atlas-cost normalisation issue deferred, see below.)
- headline_8way_table.tex:41 empty cell `--` (en-dash) -> `-`; line 10 comment `--` -> `-`. ALSO fixed
  the GENERATOR tools/headline_8way_table.py (fmt() returned "--"; comment line) so re-running it stays
  style-correct.
- inference_cost_table.tex:80 IFS-ENS Params `n/a` -> `-` (style rule: empty cells single hyphen).
- main.tex:806 colors->colours, :842 modeled->modelled, :854 normalized->normalised (British, to match
  the doc majority + user voice).
- main.tex:737 stray trailing word "(0.66 above)" -> "(0.66)".
- main.tex:33-34 TODO comment PNG list "Milton F1-F5" -> "F1-F5, F8" (F8 is also PNG).

Scientific correctness:
- main.tex:842-845 body called ES+VS "two strictly proper multivariate rules" -> "two multivariate
  scoring rules, one strictly proper (ES) and one proper (VS)". VS is only proper (Scheuerer-Hamill);
  this now agrees with the App.B summary line 1469 ("VS is proper") which the 2026-06-12 sweep already
  fixed but didn't propagate to the body.

Figures (orphans: 4 floats were defined but never \ref'd - AMS requires every float referenced):
- Added `Fig.~\ref{fig:spectrogram-delta}` in the spatial-mean mechanism paragraph (~718).
- Added `Fig.~\ref{fig:bivariate-Tq}` in the multivariate section (~883).
- Added `Fig.~\ref{fig:milton-spaghetti}` (F1) in the Milton detection paragraph (~941).
- Added `Fig.~\ref{fig:milton-intensity-vs-lead}` (F2) in the Milton intensity paragraph (~984).
  All four added sentences paraphrase the figures' own captions (no new unverified claims).

Tectonic recompile after edits: PASS, PDF 7.5 MB, NO undefined citations/references, no LaTeX errors,
no undefined control sequences. 29 over/underfull box warnings remain (pre-existing, cosmetic) +
the usual TeXGyreTermesX font-substitution warnings (cosmetic).

## DEFERRED for the user (real issues, but framing / numbers I can't pin down unattended)

NARRATIVE / FRAMING (the user reserved the GraphCast framing on 2026-06-12; these all touch it):
1. GraphCast "coarsest mesh nodes wins" interpretability claim (abstract:81, contrib-3:180, results:562,
   discussion:1146, conclusions:1339, App.C:1515) contradicts its production pick = ALL layers (Phase 1,
   CRPSS@240h 0.190); the coarse-mesh probe n_coarse_42 is 0.012 (near-worst). The "winner" wording calls
   a non-winner the winner for GraphCast. Two agents proposed rewrites that recast GraphCast's coarse-mesh
   result as the interpretability PROBE (activation space) while the production baseline perturbs all
   weights. NOT applied (user-owned framing). Also a real phase-label slip inside that same sentence
   cluster: App.C:1515 "Phase 3b winner is the first 42 nodes" - n_coarse_42 is Phase 3, not 3b; and
   results:563 "g2m + first 42" conflates Phase-2b g2m with Phase-3 n_coarse_42. Fix label + framing
   together when the user decides.
2. Post-hoc baseline count 3-vs-4 scoping (abstract:65, contrib-1, conclusions:1314): the headline
   quantitative claims ("within 0.09 to 0.13" @240h, "two of the three" SSR) silently scope to the 3
   non-AIFS post-hoc baselines, but the tables list 4 post-hoc rows. aifs_perturbed actually trails
   AIFS-ENS by only 0.035 @240h (much better than 0.09-0.13), and its frozen 240h spatial-mean SSR is
   2.54 (also above band), so "two of the three" is really "three of four". Suggest gating with "three"
   or naming the backbones, AND consider promoting the strong AIFS result. Interacts with framing -> user.
3. 360h "within 0.06" overclaim (abstract:67, contrib-1:156, conclusions:1315): unqualified, but
   aurora_encoder's 360h gap is 0.095 (0.084-(-0.011)); only SFNO (0.055) and GraphCast (0.061) are
   ~within 0.06. The body (669) + fig caption (647) already qualify "for SFNO and GraphCast". The
   abstract/intro/conclusion need the same qualifier - but the exact wording is entangled with #2
   (which baselines count), so left for the user. Minimal fix: append "for SFNO and GraphCast".

NUMBERS NEEDING USER INPUT ON WHAT WAS ACTUALLY RUN:
4. Atlas inference-cost normalisation cluster (inference_cost_table.tex). The Atlas s/mbr=2280 is a 61-step
   (=15.25-day) rollout (footnote: "61 rollout steps x 37.5 s/step"), but every other s/mbr is the 10-day
   (40-step) workload the table caption defines. So the "38 min" and the 20-40x slowdown mix rollout
   lengths. 10-day-normalised Atlas would be ~1500 s (25 min), ratio ~12-25x. Also the GH200-vs-A100
   "~1.5x faster" footnote compares 38 min (GH200, 15.25-day) to 63 min (A100, 10-day); per-step it is
   94/37.5 = 2.5x. NEEDS the user to confirm whether Atlas was run at 10-day or 15.25-day on the
   production grid, then renormalise s/mbr, CS h, "38 min", the slowdown factor, and the GH200 speedup
   together. I only harmonised the 25-40 vs 20-40 caption inconsistency (above).
5. Training-cost "20-90 thousand GPU-hours" (abstract:60, contrib:120, body:897-902). The body sentence
   lists Aurora 13K / GraphCast 22K / SFNO 4K / AIFS 11K (these are the DETERMINISTIC-backbone single-pass
   training costs, the "Train" column) inside a sentence claiming the TRAINED-PROBABILISTIC baselines
   "require 20-90 thousand additional GPU-hours". SFNO 4K and AIFS 11K are below the 20K floor and are
   not trained-prob baselines. The disclosed trained-prob additional cost is really AIFS-ENS 22K + FCN3
   90K (Atlas undisclosed) = 22-90K. Agent proposed splitting the two cost categories. Also FCN3 "90K"
   vs the footnote's own derivation 1024 H100 x 78h = ~80K (Phase 1 only); pick 80K or state which stages
   90K covers. Left for the user (touches the headline "20-90K" claim).

OTHER (smaller, judgment calls):
6. main.tex:1469 App.B propriety line says "ES and truth-std-normalised CRPS are strictly proper". There
   is no truth-std-normalised CRPS metric in the pipeline (CRPSS uses plain fair CRPS / climatology
   denominator; truth-std normalisation appears only in ES/VS). BUT this exact phrase was a DELIBERATE
   2026-06-12 edit, so I did not touch it - confirm intent. (The related body overstatement at 842 IS
   fixed above.) Secondary: ES as applied is truth-std-normalised (energy_variogram_score.py:123, the
   std is from the verifying truth); one agent wants a propriety caveat. My read: a forecast-independent
   per-layer rescale (same scale on forecast and obs) preserves strict propriety, so the caveat is
   probably unnecessary - user's call.
7. aifs sigma display 0.028 (3 tables) vs 0.0275 (cost footnote, run tag mag_0.027500). Display-precision
   call the user already deferred 2026-06-12.
8. main.tex:302 SFNO "573M params" is not in either cited source (Bonev2023 architecture paper;
   Mahesh2024 lists 48M/218M/1.1B). 573M is the e2s sfno_73ch_small config - likely correct but
   effectively uncited. Consider a config/checkpoint citation or a footnote.
9. 6 unused bib entries (Hamill2001, MarksHouze1987, Rogers2013, Knaff2019, Bengtsson1995, Hersbach2000)
   - harmless (BibTeX drops uncited), likely leftovers from the earlier Milton draft. Leave or prune.
10. spatial-SSR band 0.62-1.42 (body) vs 0.6-1.4 (abstract/concl) precision; F5 caption omits the
    classical (IFS-ENS) line style; ablation-temporalcrps caption (533-535) keeps a "will be combined
    into a single PDF" pre-submission note (tied to the PNG->PDF blocker). All nit/low; left as-is.

KNOWN SUBMISSION-BLOCKERS NEEDING THE USER (unchanged, re-confirmed present):
- [FUNDING-TODO] placeholder in acknowledgments (main.tex:1639).
- Zenodo data DOIs (App.D:1582, data statement) - correct pre-acceptance "will be inserted" phrasing,
  not a defect; fill after acceptance.
- Abstract ~263 words > AMS 250 limit (~13 over); refresh-every-N sentence (71-77, ~60 words) is the cut.
- PNG->PDF vector regen for raster figures (temporalCRPS x3, spectrogram, Milton F1-F5+F8) - AMS wants
  vector. SwissClim regen needed.

## REJECTED findings (do not re-litigate)
- Headline table cells (aurora/sfno @240h) re-derived end-to-end from temporal_metrics_combined.csv +
  crps_clim_eval_1990_2019.json: MATCH the committed .tex bit-for-bit. Uplift table, Milton bands,
  ES/VS 14% gap, all re-verified correct.
- "three deterministic backbones" at 1130 is intentional (that paragraph is about the 3 deterministic;
  AIFS discussed separately). AIFS 255M (text) vs 229M (cost table) is NOT an inconsistency - 255M is the
  deterministic AIFS, 229M is AIFS-ENS, different models. Zenodo placeholder + TODO comment block are
  intentional pre-submission state. Phase-6c-adopted-but-headline-frozen is already disclaimed in text.
