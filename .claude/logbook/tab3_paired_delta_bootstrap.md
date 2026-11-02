---
name: tab3-paired-delta-bootstrap
description: "2026-06-04: side-analysis method for testing whether Tab. 3 within-(model, lead) winner-vs-runner-up margins survive the 4-init noise floor. Paired-delta bootstrap with B=10000 and a 90% CI. Kept as reviewer-defence ammunition only -- user explicitly does NOT want this added to the paper text, captions, or figure annotations."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Policy

Per user instruction 2026-06-04: this analysis exists for our own confidence and as ammunition if a reviewer pushes on the noise-floor question, but is NOT to be added to the paper text, the Tab. 3 caption, or any figure annotation. Do not introduce CI columns, `+/- 0.0XX` annotations, daggers, or footnote text about bootstrap intervals into `ai-models-ensembles-paper/figures/calibration_basis_table.tex` or `main.tex`.

If a reviewer specifically raises Reviewer 1's "winner margin smaller than within-phase noise" critique, this analysis is the answer to point at. Until that happens, the published Tab. 3 stays with point-estimate bolds only.

## What it does

For each (model, lead) row in Tab. 3 ablation:
1. Pick the within-model point-estimate winner phase X and runner-up phase Y (by mean CRPSS across the 4 mid-season inits).
2. Form the 4-init paired-delta vector `d_i = CRPSS_{X, i} - CRPSS_{Y, i}` for `i in 1..4`.
3. Resample those 4 deltas with replacement, B=10000 times, compute `mean(d_resampled)` per draw.
4. 90% CI for the inter-phase gap = `[percentile(m, 5), percentile(m, 95)]`.
5. The bold "survives" the test if the CI excludes 0 (and mean(d) > 0).

Resampling code (the whole bootstrap is 5 lines):
```python
rng = np.random.default_rng(42)
idx = rng.integers(low=0, high=4, size=(10000, 4))   # (B, N)
m   = deltas[idx].mean(axis=1)
ci_lo, ci_hi = np.percentile(m, [5, 95])
```

## Why paired and not per-cell at N=4

Per-cell bootstrap at N=4 has three known issues:
- Only `C(N + N - 1, N - 1) = 35` unique multinomial weight vectors -- the bootstrap distribution is coarse.
- Variance estimator has a `(N - 1)/N` factor -> ~25% downward width bias at N=4.
- A CI on the cell value does not directly answer the inter-phase question; you'd have to compare two overlapping CIs that share most of their noise.

Pairing cancels the shared atmospheric-state effect across phases at a given init -- hard inits drag all phases down together, easy inits push them up together -- leaving only the phase-specific component in `d_i`. Empirically `std(d)` is roughly 3-10x smaller than `std(CRPSS)` per cell, so the paired test has real statistical power even with N=4. This is the standard fix for the reviewer's noise-floor concern.

The 35-outcome / (N-1)/N issues still technically apply but to a noise level one third the size, so the resulting interval on the inter-phase gap is much tighter than any cell-level CI.

## Implementation

Script: [/iopsstor/scratch/cscs/sadamov/tab3_bootstrap_ci.py](/iopsstor/scratch/cscs/sadamov/tab3_bootstrap_ci.py)

- Recomputes fair CRPS (M=10) per (model, phase, init, lead, variable, level) directly from `<STORE>/ablation/<phase>/<model>/<init>/<config>/forecast.zarr` against the WB2 ERA5 truth zarrs (consolidated=True), cos-lat-weighted global mean.
- Sigma_clim denominator: `/iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json` (the current 4-year paper recipe). If/when the 30-year sigma_clim_1990_2019.json is integrated, re-run with `--recompute`.
- Variable-mean over the 7 paper-headline vars (2m_t, MSL, geo, temp, u, v, q); 3D vars averaged over 500 + 850 hPa first; matches `tools/headline_7way_table.py`.
- Parallelised over 60 (model, phase, init) tasks with 8 workers (ProcessPoolExecutor).
- **Per-task JSONL checkpoint** at `tab3_bootstrap_task_checkpoint.jsonl` so kills don't lose work. The checkpoint can be seeded from a previous run's log via `tab3_seed_checkpoint.py`.
- Final cube cached at `tab3_bootstrap_per_cell_crpss.json`; rerun without `--recompute` skips collection.

Run as:
```bash
systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 --unit=tab3_bootstrap.scope \
  bash -c 'export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp; \
           export SSL_CERT_FILE=$(... certifi ...); \
           /iopsstor/scratch/cscs/sadamov/venvs/ai-models-ensembles/bin/python \
             tab3_bootstrap_ci.py >> tab3_bootstrap_ci.log 2>&1'
```

Outputs (all in scratch, never in the paper repo):
- `tab3_bootstrap_ci.csv` -- one row per (model, lead) with mean_winner, mean_runnerup, mean_delta, ci_lo, ci_hi, ci_excludes_zero, sigma_init_winner, sigma_init_runnerup, n_inits.
- `tab3_bootstrap_ci.md` -- human-readable summary table + per-cell list of which bolds survive.

## Scope and limitations

- N=4 means even paired tests don't have power at 95%. CI is 90% by design.
- Bootstrap is non-parametric; no Gaussian assumption on d_i, no symmetry assumption on the CI.
- Pairing only answers winner-vs-runner-up. The winner-vs-rest extension (winner against EVERY other phase, with Bonferroni or Holm correction over P-1 paired tests per row) is queued as **#33b in [[paper-revision-round-todos]]**. It reuses this script's per-cell cube and adds the correction step on top. Trigger if a reviewer escalates the noise-floor question, or if we want a sharper claim before final submission.
- Coverage = nominal 90% under the bootstrap principle assumptions (delta i.i.d. across inits, smooth functional of the empirical distribution). At N=4 the true coverage is somewhat lower; the standard caveat for any small-N bootstrap.

## Related

- [[paper-revision-round-todos]] item "#33 Bootstrap CIs for Tab. 3 ablation winners" -- the reviewer-flagged motivation. Status now: implemented as side-analysis, NOT paper-integrated, per user decision.
- [[crpss-migration]] -- the Tab. 3 CRPSS denominator recipe this consumes.
- [[metric-audit]] -- the broader CRPSS provenance audit.
