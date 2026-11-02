# 2026-07-06 ATTENDED: bivariate physical-consistency figures -> pooled members (+ subsampling), prose corrected against data

One-line: reviewer point "physical-consistency checks pool ensemble MEANS, not members" fixed end-to-end - added 1M paired subsampling to the SwissClim bivariate module (mean-flatten was tripping a Dask P2P shuffle), reran all 8 baselines pooled, regenerated Figs C1/C2, and CORRECTED the prose against the actual pooled NPZ data (the old "post-hoc broader low-q tail" claim was false).

## Why (reviewer M / verify-claims)
- Figs C1 (T-q Clausius-Clapeyron) + C2 (geostrophic) were computed on ensemble MEANS (`multivariate: mean` in SwissClim `_DEFAULT_ENSEMBLE_MODES`). The mean is not a physical state (q_sat convex -> a mean of physical members can sit above saturation; averaging masks member-level violations). Switched to `multivariate: pooled`.

## Root cause of the pooled crash (decisive)
- Uncapped pooled `.flatten()` on the ~7e10-point pooled array (112 inits x 10 members x 61 leads x full global grid) triggers a Dask distributed P2P shuffle that fails deep in the histogram2d unpack. aifsens uncapped ran 2h54m then died with `P2PConsistencyError`. Flaky (fcn3/atlas ran past that point), so not usable for paper figures.
- Fix: 1M paired subsample. The bivariate module was the ONLY `pooled` artifact with NO cap; wd_kde (W1) caps at 200k via `subsample_values`, histograms at auto too. Added `subsample_values(arr, max_samples, seed, lazy=True)` routing to `plots/bivariate_histograms.py` (`_flat` closure, all 8 flatten sites) + `bivariate_max_samples` resolve in `metrics/multivariate.py` (auto->1e6, null->full) + `bivariate_max_samples: auto` in `scripts/evaluate_baselines.sh`. Paired-ness UNIT-TESTED (same seed -> var_x/var_y sampled at identical grid cells; y=3x+7 joint survives; different seed does not pair; k>=size returns full field so ensemble-mean path untouched).
- Result: 8 jobs (2692624-2692634) COMPLETED clean in ~1h10-1h30 each, 48/48 enspooled NPZs. vs prior 3-4h crashes.

## Number choice: 1M (KDE-consistent alt 200k)
- 2D joint hist = 100x100 = 1e4 bins; 1M -> ~100/bin, smooth tail. 200k (the kde value) would thin the tail vs the current published (full-pool mean) look. Cost identical (both ms). Seed-invariance DEMONSTRATED synthetically: 3 seeds at 1M -> histogram correlation 0.9955, max bin diff 9e-5, tail mass identical.

## Prose CORRECTED against pooled NPZ data (was written for mean figures)
Quantified from the enspooled NPZs (T-q 500hPa; supersaturation = mass above Bolton q_sat):
- Supersaturation %: aurora 0.03, graphcast 0.30, sfno 0.67, aifs_pert 1.21, AIFS-ENS 0.14, FCN3 1.26, Atlas 0.22, IFS-ENS 0.02; TRUTH 0.11. -> "every baseline respects envelope" too absolute; small tail (0.02-1.3%) comparable to ERA5's 0.1%, NOT post-hoc-specific (FCN3 trained is highest).
- std(q) e-3: post-hoc 1.04-1.14, trained 1.16-1.19, truth 1.27. -> OLD "post-hoc show broader cold low-q tail" is FALSE (post-hoc are NARROWER in q). Corrected to: all forecasts marginally narrower in q than ERA5 (deterministic smoothing).
- Geostrophic off-line scatter (mass >15 m/s from U_g=(g/f)|gradH|): post-hoc 3.4-5.8%, Atlas 5.7%, vs ERA5 2.0 / AIFS-ENS 2.9 / FCN3 2.2 / IFS-ENS 2.1. -> mean HID this; caption now: core follows line, post-hoc + Atlas carry more member ageostrophic scatter (ties to existing Atlas geopotential caveat).
- Also fixed: captions "ensemble means"->pooled members; the WRONG "IFS-ENS excluded" -> all 8 panels incl IFS-ENS (raw-field hists tolerate the NaN gaps that kill ES/VS/SIGK); 1M subsample disclosed in-caption.

## Also this session: existing subsampling motivated in paper (appendix B)
- Verified vs code + added motivation: VS 5000 random pixel pairs (`tools/energy_variogram_score.py:167,186`, avoids O(N^2) full sum), SIGK 128 cos-lat pixels/init (`tools/signature_kernel_score.py:193`, kernel cost, mirrors VS), W1 200k paired subsample (wd_kde `kde_max_samples: auto`, was UNDISCLOSED - "pooled over members and pixels" had no cap note). ES confirmed FULL-grid (no subsample). All draw under fixed seed 42.

## Commit state (concurrent session committed most as it went)
- SwissClim `7483b98` "introduce subsampling for bivariate" (multivariate.py + bivariate_histograms.py) - COMMITTED.
- code `4c1fa8c` "introduce subsampling for bivariate" (evaluate_baselines.sh + template pooled) + `19a1dcc` - COMMITTED.
- paper `paper_refs.bib` Bulte2024 fix (eprint 2410.04483->2403.13458, authors C.Bulte/Horat/Quinting/Lerch; verified NOT a weight-perturbation paper) - COMMITTED.
- THIS umtrac commits: code-repo pooled `figures/bivariate_*.pdf/png`; paper `main.tex` (caption/prose corrections + appendix B subsampling motivation) + vendored `figures/bivariate_*`.
- LEFT UNSTAGED: code-repo deletion of `PAPER_REVIEW_ams_aies_blind_2026-07-06.md` (was committed `bbdebde`, now removed - user's review draft, their call).

## After-break checks
- Word count 7396 conservative (104 under 7500). Compiles clean (tectonic, no undefined refs).
- If ANY bivariate figure is regenerated again: it now defaults to pooled+1M; `tools/plot_bivariate.py` prefers `enspooled` over `ensmean` NPZ.
- Milton `eval_milton/multivariate/` still has OLD ensmean NPZs (not regenerated - Milton bivariate not in the paper headline figs; only the 112-init production ones matter).
