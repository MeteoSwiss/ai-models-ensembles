# 2026-06-18 Signature Kernel Score (SIGK) added to the appendix + refresh/IC probe (in flight)

One-line: implemented the signature kernel score (temporal-path proper scoring rule) as a new tool,
scored all 7 production baselines, merged it as a 3rd column into the ES/Variogram appendix table (C1),
committed+pushed both repos; a refresh/fresh/IC variant probe is running (job 2563873) for the user to
inspect in chat - NOT in the paper.

## Origin
User flagged a new preprint: Dodson & Dutta, "Signature Kernel Scoring Rule: A Spatio-Temporal Diagnostic
for Probabilistic Weather Forecasting", TMLR 05/2026, OpenReview LOLXpt4E5D. Asked whether to add SEEPS /
ROC / the signature score. Verdict given: SEEPS N/A (no precip var), ROC optional (Milton only), the
signature kernel score is the one genuinely additive metric (temporal-path structure - the temporal
analogue of our spatial Variogram Score). User asked to implement it.

## What the score is (their Eq. 1, lower is better)
phi(F,o) = E_{X,X'~F}[Ksig(X,X')] - 2 E_{X~F}[Ksig(X,o)]. Each pixel's forecast is reframed as a PATH over
lead time (the 12-layer standardised trajectory), basepoint+time augmented; Ksig is the signature kernel
(inner product of path signatures) computed as a Goursat PDE under a static RBF kernel (sigma=1, dyadic
order 1). Energy-score family with a sign switch (kernel=similarity vs norm=distance). Strictly proper:
minimised when the ensemble path distribution = truth path distribution. Term 1 penalises under-dispersion
(members too self-similar), term 2 rewards closeness of member paths to the observed path.

## Implementation (main repo, committed 5e5e683, pushed to MeteoSwiss)
- `tools/signature_kernel_score.py` - numpy Goursat-PDE signature kernel (no torch/sigkernel dep),
  basepoint+time augmentation, Eq.1 estimator on a cos(lat)-weighted random pixel sample per init.
  `--self-test` validates the PDE vs from-scratch truncated signatures (rel-err ~1e-5, symmetric, RBF Gram
  PSD). Mirrors `energy_variogram_score.py` data loading + per-layer truth-std normalisation.
- `tools/submit_signature_kernel_score.sh` (SLURM, 7 baselines parallel), `tools/aggregate_signature_kernel_score.py`,
  `tools/data/sigk_production.csv` (provenance).
- Settings: lead 240, lead-stride 2 (12-hourly, L=21), n-pixels 128, sigma 1.0, dyadic 1, seed 42.

## Production results -> Table C1 (paper df8b07a, pushed to sadamov)
SIGK merged as the 3rd column of the ES/VS table (`tab:esvs`); the standalone `tab:sigk` table I first
added was removed at user request ("add it ... in the appendix table ev vario"). Bib: `Dodson2026sigkernel`
added to paper_refs.bib (Zotero side PENDING, handled separately).
Methods scoring-rule table got a SIGK row. Values (lower=better), 112-init grid, 128 px/init:
  AIFS-ENS -37.1 (best) | Atlas -33.1 | FCN3 -29.4 | aifs_perturbed -12.2 | aurora -9.9 | sfno +1.0 | graphcast_all +18.1
FINDING (additive, not a restatement of ES): SIGK agrees with ES/CRPSS at the top (3 trained-prob lead,
perturbed-AIFS best of the post-hoc group) but ranks graphcast_all LAST by a wide margin - mid-pack under
single-lead ES. Reads as temporally over-coherent (under-dispersed) member trajectories from frozen
all-layer perturbation, the failure mode the Phase-6 fresh-per-step refresh targets. Caption notes the
score is on the kernel's native scale (only ranking/gaps meaningful, not magnitude/sign).

## Refresh/fresh/IC probe - IN FLIGHT, NOT in the paper (user inspects in chat first)
Job 2563873 (debug partition, RUNNING since 21:20 on 2026-06-18; exploratory script at
`/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/sigk_refresh/submit_sigk_refresh.sh`).
Same SIGK settings/seed so paired. Configs + CSVs land in scratch `sigk_refresh/<label>.csv`:
- AIFS 112-init production (paired, anchor = aifs_perturbed frozen -12.2): aifs_perturbed, aifs_p6c
  (fresh-per-step / refresh), aifs_perturbed_ic (perturbed IC). Q: does refresh or IC improve SIGK + ranking?
- SFNO 3 ablation inits (20230515, 20240215, 20241115 - the ones with a matched fresh0.60): sfno_frozen025,
  sfno_frozen060, sfno_fresh060 (matched-sigma frozen-vs-fresh, the clean test), sfno_refresh20_035. Noisier (3 inits).
NEXT SESSION: read scratch/sigk_refresh/*.csv (or rerun the aggregator block), report deltas vs frozen +
re-ranking; do NOT put in paper until user approves.
GraphCast refresh = NO DATA anywhere on store/scratch (never produced). A graphcast refresh would need
fresh GPU inference over all inits (hours), offered to the user, not yet submitted.

## Lessons / environment
- VENV MIGRATED to capstor: `.venv` now symlinks to `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles`
  (persistent; the old `/iopsstor/scratch/cscs/sadamov/venvs/ai-models-ensembles` was purged - numpy __init__
  gone, `numpy has no __version__`). The dead scratch path (still hardcoded in my first submit + the template)
  broke the first variant job (2563696, exit in 9s). FIX: all `submit_*.sh` PY= updated to the capstor path
  (this session's venv-migration commit). Login node masked it (the .venv symlink resolved fine); only the
  hardcoded scratch path failed, and only on the compute node.
- For host analysis use `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python` (or `.venv/bin/python`).
