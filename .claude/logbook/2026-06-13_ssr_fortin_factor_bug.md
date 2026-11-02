# 2026-06-13: spatial-mean SSR Fortin-factor bug - all SSR^sp values were 5.4% too high

One-line: `tools/spatial_mean_ssr.py` applied (M+1)/(M-1) to the ddof=1 sample variance instead of
(M+1)/M; every spatial-mean SSR in the paper was inflated by sqrt(10/9) ~ 1.0541 at M=10. Fixed the
tool, rescaled all 41 affected CSVs exactly post hoc, regenerated both SSR figures, updated all paper
numbers and both summary-table formulas.

## The bug

Fortin et al. 2014 (doi:10.1175/JHM-D-14-0008.1): for a reliable M-member ensemble,
E[MSE of ensemble mean] = (M+1)/M * E[sample variance (ddof=1)]. The correct spread^2 estimator with
ddof=1 variance is therefore var * (M+1)/M. The tool computed var(ddof=1) * (M+1)/(M-1), i.e. it
double-counted the M-1 adjustment (the (M+1)/(M-1) form pairs with the ddof=0 variance). Constant
multiplicative bias: spread and SSR high by sqrt(M/(M-1)) = sqrt(10/9).

Cross-check: the per-pixel SSR (SwissClim / WeatherBenchX `UnbiasedSpreadSkillRatio`) is NOT affected -
it uses the other equivalent correction (subtract s^2/M from the squared error of the ensemble mean,
verified in WBX source).

## Remediation (exact, no recompute)

Because M=10 uniformly across every aggregated cell, the fix is an exact scalar rescale:
spread and ssr columns x sqrt(9/10) = 0.948683..., error unchanged. No zarr re-reads.

- Tool fix: `tools/spatial_mean_ssr.py` now uses `var_M * (M + 1) / M` (comment documents the bug).
- One-shot script: `tools/rescale_ssr_fortinfix.py` (stdlib-only; login-node safe). Targets
  `baselines/*/spatial_mean_ssr/spatial_ssr.csv`, `ablation/*/*/eval/*/spatial_mean_ssr/spatial_ssr.csv`,
  `diagnostics/sfno_phase6_ssr/*.csv`. Aborts on n_members != 10. Idempotent (skips when archive exists).
- Archives: `<stem>_pre_fortinfix_2026-06-13.csv` alongside each CSV, EXCEPT diagnostics where
  `plot_sfno_phase6_ssr.py` globs `*.csv`, so archives went to `pre_fortinfix_2026-06-13/` subdir.
- 41 CSVs rescaled: 14 baselines (incl. _ic variants, aurora_p6c, aifs_p6c), 12 phase3/sfno eval tags,
  8 phase3/graphcast_operational gcsigma tags, 7 diagnostics CSVs.
- Spot check exact: 0.7509401880 * sqrt(0.9) = 0.7124044142 (matched to full precision); error column
  byte-identical. Re-run: 0 rescaled / 41 skipped.
- Gotcha hit: pandas writes NaN ssr as '' (msl/u-wind rows in GC activation CSVs where truth was NaN);
  first run crashed AFTER archiving gcsigma_0.01, leaving an archive identical to the original which the
  idempotency check would wrongly skip-and-keep. Detected via filecmp, removed, re-ran with ''/'nan'
  cells passed through.

## Aggregation convention (reverse-engineered, now documented)

Every quoted paper SSR^sp number is var7_mean: level-mean per 3D variable first (2D variables deduped -
the tool writes them twice at level 500/850 with identical values), then mean over the 7 paper
variables. Same convention as the paper's CRPSS aggregation (main.tex Sec. methods-metrics). The
fig:tier1b caption now states this.

## Corrected numbers (OLD -> NEW, all verified from rescaled CSVs + regen log)

Production grid @240h: graphcast_all 3.38->3.20, aurora_encoder 1.65->1.56, sfno_modes10 0.70->0.66,
trained band 0.65-1.49 -> 0.62-1.42 (aifsens 0.618, fcn3 0.747, atlas 1.415). aurora_p6c 1.17->1.11
(@24h 1.38->1.31; aurora_encoder @24h 1.35->1.28). graphcast_all_ic 1.96->1.86. AIFS production
2.54 (frozen) -> 1.68 (p6c refresh); per-var temperature 3.72->0.94.
Design grid (phase3 sfno): frozen mag_0.25 0.87->0.83, p6c refresh 0.99->0.94, fresh060 0.46->0.44,
skip1 0.93->0.88. GC activation control (T2m-only @240h): frozen 0.29->0.27, fresh 0.16->0.15
(0.273/0.147 exact).

## Paper changes (nightly/2026-06-13, ai-models-ensembles-paper)

- 17 edits in main.tex: all stale SSR^sp numbers swapped (abstract, contributions, Sec 4.3 incl. both
  captions, discussion, conclusions); the three unverifiable "4-10x more spread" claims replaced with
  SSR-band statements (3.2 and 1.6 against 0.6-1.4); fig:tier1b caption corrected (six baselines not
  seven - no IFS-ENS in the plot; seven variable panels not five; level-mean convention stated); AIFS
  generalisation sentence upgraded from lost design-grid CSVs to 112-init production numbers; GC control
  made precise as 2-m temperature at 240h.
- Summary table: SSR^pix formula now subtracts s^2/M in the denominator (WBX unbiased form); SSR^sp
  formula now carries (M+1)/(M(M-1)) on the member-variance sum; prose note after the Phi definition
  explains both finite-M adjustments, citing Fortin2014.

## Figures

Regenerated via sbatch job 2525600 (COMPLETED, 15s): figures/tier1b_7way_spatial_mean_ssr.{pdf,png},
figures/phase6_sfno_spatial_mean_ssr.{pdf,png}. Log
($STORE/ablation_logs/regen_ssr_figs_2525600.log) prints per-baseline means that independently confirm
every paper number above. Note: the log's phase6 variant-mean uses a different aggregation
(12-field mean) than var7_mean, so its frozen_0.25=0.820 vs paper 0.83 is not a discrepancy.

## Committed (2026-06-13)

main repo nightly/2026-06-13: fd017d7 (tool fix + rescale script + 2 regen figures), 78744e6 (K<4-init
guard), f170534 (table tex provenance). Paper repo nightly/2026-06-13: 3733d8a (the 17 main.tex edits).
Nothing pushed; main untouched.
