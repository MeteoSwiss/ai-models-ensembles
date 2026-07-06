# Referee report (simulated): AMS AIES - open points only

Manuscript: "Probabilistic Forecasting from Deterministic Machine-Learning Weather Models via Multi-Scale Post-hoc Weight Perturbation"
Re-audited against the live working-tree `main.tex` + figures, attended session. Only items still open remain; everything fixed in the tree has been dropped.

## Resolved this session

- **M1. Sampling uncertainty now quantified.** The paired block bootstrap (`tools/block_bootstrap_crpss.py`, 10,000 resamples over the 8 weekly init-blocks) was run: it reproduces the headline table exactly, then confirms every 240 h CRPSS gap to AIFS-ENS is significant (95% CIs exclude zero, P(gap>0)=1.000 for all), from 0.036 [0.031, 0.041] for perturbed AIFS to 0.131 [0.121, 0.141] for sfno_modes10 (Atlas the tightest at 0.019 [0.009, 0.030]). The "Sampling uncertainty" limitation was upgraded from "indicative rather than significant" to state this result; the sub-0.02 ablation-cell separations are correctly kept as within-noise. Residual (optional): the abstract and headline table still print bare point estimates; CIs could be added there too.
- **Seed robustness (was M5).** All 16 seed=43 runs completed (`scripts/submit_seed_robustness.sh`, four production picks x four ablation inits, separate `ablation_seed43/` tree). `tools/compare_seed_robustness.py`: seed42 reproduces Tab. 4 exactly, and seed-to-seed CRPSS@240 h shifts by at most 0.006 (aurora +0.003, graphcast +0.004, sfno -0.006, aifs -0.001), all inside the +-0.02 band. Wired into the grid-sparsity limitation as one clause.
- **Fig. 6 and Fig. D1 panel titles (minors 1 & 14).** Milton F1 (spaghetti) and F2 (intensity) relabelled to journal-style names ("AIFS (weight+IC)", "AIFS-ENS", "IFS-ENS", ...) via a display map in `tools/milton/figures_milton.py`, regenerated, and synced into the paper repo (23 vendored figures). Verified visually.
- **M6. Spatial-mean rank histogram.** The PIT-only figure was replaced by a two-column verification-rank figure (`figures/rank_histograms_240h.pdf`, `tools/compute_rank_histograms.py` + `plot_rank_histograms.py`): left column per-pixel (near-flat, reproducing the PIT story), right column spatial-mean. The spatial-mean column shows exactly the predicted pattern, domes for graphcast_all / aifs_perturbed / aurora_encoder (over-dispersed, SSR>1), a mild U for sfno_modes10 (under-dispersed, SSR 0.66), TP baselines flattest, so the same members are near-calibrated pointwise yet mis-calibrated in the domain mean. Caption + paragraph + one cross-ref updated, synced, compiles (10.11 MB, page 41). The frozen-vs-refresh overlay the reviewer also floated is not added (would need the refresh runs re-scored for rank); the frozen dome alone already confirms the mechanism.

## Still open

**Physical-consistency checks: pooled-member rerun.** `[in flight, concurrent session]`
Figs. C1/C2 pool ensemble means; the mean is not a physical state (q_sat convex, so a mean of physical members can sit above saturation) and averaging masks member-level violations. Pooled-member eval running (`multivariate: pooled`); the five stalled >4 h jobs (2690118/120/121/122/126, Dask deadlock) were cancelled this session, three trained-prob resubmits (2691246-248) still running. Figures + the two captions ("ensemble means" -> pooled members) regenerate once the NPZs complete. Concurrent session's work.

**M3. IC-only control declined, not run.** `[soft / author, closed]`
Explicitly declined with a cited rationale (IC-alone is spread-limited; Sonderby, Bulte). Defensible; treated as closed at the author's discretion. A one-row-per-backbone decomposition on the ablation grid (perturbed analyses, unperturbed weights) would still turn the "lead-time-complementary" claim into a quantified split. Optional strengthening only.

**Tab. 2 GraphCast Phase 2 cell packs bold + dagger (+ star elsewhere).** `[text, lowest]`
Three marker types in one slashed cell block; a caption note now explains the slash pairing, but decoding is still dense. One row per group would fix it. Lowest priority.

## Questions for the authors (reply only)

1. Did you check the Phase 2 ranking's sensitivity to a parameter-count rescaling (vs the tensor/mode/node count) for at least one model?
2. Does the noise refresh at segment boundaries induce visible member-trajectory discontinuities - e.g. what does SIGK (reported only for frozen configs) do on the refresh runs?

## Editorial note

Compiled PDF is fully identifying - fine for AMS single-anonymous; anonymise if the double-anonymous option is intended. Word count: a full compression pass (this session) brought the manuscript from ~8,160 to 7,456 (conservative count incl. headers + data statement), 44 under the 7,500 AMS limit, with all reviewer additions retained and no citations dropped (Leutbecher2008 relocated from the trimmed §2.4 metric preview into appendix B). Cuts were redundancy/verbosity only: consolidated the triple metric coverage (Related §2.4 was a preview of Methods §3.5 + appendix B), compressed the Conclusions restatement, and tightened prose across Methods, Results, Discussion, Limitations, and appendices A-C. Compiles clean, no undefined refs/cites.
