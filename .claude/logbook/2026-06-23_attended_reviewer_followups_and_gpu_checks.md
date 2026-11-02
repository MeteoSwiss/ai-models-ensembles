# 2026-06-23 attended (Opus 4.8): reviewer-followup edits, AMS overage fix, GPU checklist-C checks

One-line: cleared the 5 carried cosmetic flags + 4 presubmission-checklist items the user picked. Two
substantive findings: (1) the paper was OVER the AMS 7,500 word limit (~7,614, not the hand-estimated
~7,448), trimmed back to 7,466; (2) GraphCast is NOT bit-reproducible run-to-run (JAX/XLA GPU
nondeterminism), correcting the "deterministic by construction" note. Committed + PUSHED both repos.

## Committed + PUSHED (user authorised push)
- main `7a037db`: figures/inference_cost_table.tex caption param provenance + git rm of 2
  orphan table tex (headline_7way_table.tex, experiments_tables_preview.tex).
- paper `900889d`: Chen2026atlas->Kossaifi2026atlas (4 cites + bib), SwissClim-Eval->
  SwissClim_Evaluations, AMS length trim (5 cuts).
- Also pushed the two prior nightly commits the user OK'd: main e40dcb2, paper de7b0b7.

## Cosmetic flags (all 5 done)
- Atlas citekey: first author VERIFIED = Kossaifi (arXiv:2601.18111 + HF card both say Kossaifi). Renamed
  Chen2026atlas->Kossaifi2026atlas in tex+bib (bibliography-manager side: paper-process detail kept in the local notes).
- Atlas params: VERIFIED via arXiv:2601.18111 Sec 3.1 - SI/EDM predictive net = 2.4B, decoder = 1.8B,
  CRPS = 3.3B; NO single combined total stated. So the cost-table "4.2 B" (= 2.4+1.8, the SI variant we
  use) is correct and STAYS. The HF model card's "2.5B encoder" was looser; the peer-reviewed Sec 3.1 is
  authoritative. User said NO FOOTNOTES -> folded the provenance into the cost-table \caption{} (SFNO/FCN3
  = checkpoint state-dict counts, no source total; Atlas = 2.4B+1.8B sum) instead of footnote markers.
- FCN3 711M / SFNO 573M: checkpoint-derived (06-22 logbook verified neither source states a total); same
  caption sentence.
- Orphan tex: headline_7way_table.tex + experiments_tables_preview.tex confirmed unreferenced (only
  self-references), git rm'd.
- SwissClim-Eval -> SwissClim_Evaluations everywhere (user chose repo-name casing).

## GitHub URL
(paper-process detail kept in the local notes)
- Both GitHub repos verified PUBLIC via `gh repo view`: MeteoSwiss/ai-models-ensembles (default branch
  main) + swiss-ai/SwissClim_Evaluations. Casing matches the paper. Zenodo correctly deferred to
  post-acceptance.

## FINDING 1: AMS word-count overage
(paper-process detail kept in the local notes)

## FINDING 2: GraphCast is NOT bit-reproducible run-to-run (GPU checklist-C)
- Two single-member graphcast_operational runs, IDENTICAL seed(42)/config/init(2024-02-15)/GPU, lead 24h
  -> DIFFER from step 1: 2t mean 0.018 K / max 0.43 K; MSL mean 1.9 / max 44 Pa; z500/850 mean 1.7 / max
  34 m^2/s^2. (job 2600463 COMPLETED; compared on host venv via gc_compare.py.)
- CAUSE: GraphCast is the only JAX model; `_seed_rngs` sets torch/numpy + cudnn.deterministic +
  CUBLAS_WORKSPACE_CONFIG, all PyTorch-only - they do NOT control JAX/XLA, whose GPU reductions are
  nondeterministic by default (would need XLA_FLAGS=--xla_gpu_deterministic_ops or similar). So "GraphCast
  deterministic by construction" (CLAUDE.md/MEMORY) means NO STOCHASTIC TERM, not bit-reproducible.
- IMPACT: none on the paper. grep of main.tex: every "deterministic" = models have no noise term; App E
  claims code availability, not exact numerical reproducibility. Run-to-run floor (~0.02 K mean) << the
  weight-perturbation member spread (graphcast_all sigma=0.01), so it is a negligible confound. Documented
  caveat only. SUGGEST tightening the CLAUDE.md reproducibility line to "GraphCast carries no stochastic
  term; JAX/XLA GPU reductions are not bit-reproducible run-to-run (the PyTorch cudnn/CUBLAS determinism
  flags do not cover it)".

## FINDING 3: fresh-per-step hook fire count confirmed for all 3 weight-space backbones
- The 1-fire-per-AR-step law (guards the old sfno silent-identical-members bug, see sfno-phase6-hook-bug):
  - aurora: 8 fires / 8 steps (Jun-15 fresh_verify log aurorab_2536055, every-step print).
  - sfno: 8 spectral modules x 8 steps = 64, all modules in sync (Jun-15 sfnob_2536112).
  - aifs: job 2600469 COMPLETED today - decoder hook installs on 29 params, fires (step=1, |dW|mean=0.022
    at sigma=0.0275), 24h rollout completes (lead 0/6/12/18/24). Decoder forward_pre_hook is called once
    per AR step by construction -> 1 fire/step. (The Jun-15 aifs run timed out on the cold ECMWF open-data
    fetch; the first today retry failed on a missing ~/.cdsapirc MOUNT - fixed by binding it.)
- Current hook code gates the per-fire print to `if step == 1` (3 fresh_perturbation.py files), so the log
  shows install + step-1 only; the fire COUNT is inferred from (hook installed)+(step-1 fired)+(output zarr
  lead-step count). To directly count again, un-gate the print or add an end-of-rollout counter.

## How to launch a fresh-per-step / single-model debug run (no committed launcher exists)
Scratch harness: /iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/fresh_verify/ (submit_checks.sh,
gc_compare.py - throwaway). Key mounts (mirror submit_all_inference.sh:145-148):
  SRC:/workspace/ai-models-ensembles , SRC/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles ,
  STORE:STORE , /iopsstor/scratch/cscs/sadamov/e2s_cache:/workspace/.cache/earth2studio ,
  <scratch-out>:<scratch-out>  AND (for cds/aifs) ~/.cdsapirc + ~/.ecmwfapirc to both their own path and /root/.
GOTCHA: only e2s_cache is mounted from iopsstor scratch by default - if --output points at any OTHER scratch
path you MUST add it to --container-mounts, else the zarr writes to the container's ephemeral overlay and
vanishes on exit (this is why the Jun-15 fresh_verify/{aurora,sfno,aifs}/ output dirs are empty - only the
logs survived). Fire-per-step env: {SFNO,AURORA,AIFS}_FRESH=1 + _SIGMA + (_MODE_CUT for sfno) + _REFRESH_EVERY,
with --weight-magnitude 0.0 (else the static perturbation path is taken). model ids: graphcast_operational,
aurora, sfno, aifs; data-source arco (aurora/gc/sfno) / cds (aifs).
