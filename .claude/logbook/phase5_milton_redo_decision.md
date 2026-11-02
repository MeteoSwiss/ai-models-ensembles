---
name: phase5-milton-redo-decision
description: Decision gate for whether to re-run the Hurricane Milton case study (Sec 4.6) with Phase 5 (IFS-ENS-IC + weight) baselines once aurora/graphcast/sfno Phase 5 evals land. User explicitly wants to remember this for after the 3 evals.
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Should Milton (Sec 4.6) be re-run with Phase 5 baselines?

## Decision (2026-06-09, user)

**Yes in principle, but decision-gated**: do NOT pre-launch the Milton Phase 5 grid. Wait for aurora/graphcast/sfno Phase 5 evals to complete (forecasts in flight 2026-06-09 evening, evals to follow). Re-evaluate then.

The user explicitly said "no, we wait and remember it" -- this memory captures the plan so it isn't forgotten when the 3 evals finish.

## Why

**For re-running**:
- Hurricane tracking quality lives on early-lead ensemble spread. AIFS Phase 5 lifted SSR from 0.49 -> 0.67 at 24h and 0.71 -> 0.86 at 72h -- exactly the leads where Milton landfall prediction matters (Milton lifecycle 2024-10-04 to 2024-10-11).
- The current weight-only Milton table has a coherent-shift artefact ([[spatial-mean-vs-pointwise-ssr]]) that Phase 5 demonstrably reduces.
- Narrative consistency: if Phase 5 is the recommended deployment, Milton should demonstrate Phase 5's value on a real high-impact event.

**For waiting + caveats**:
1. Currently only AIFS Phase 5 is evaluated. Aurora/GraphCast/SFNO Phase 5 forecasts are in flight; evals come after. Need to confirm Phase 5 helps each model before committing.
2. Phase 5 may not change DETECTION RATE (depends on the deterministic forecast, not ensemble spread). Phase 5's gain is likely in MEAN POSITION ERROR (more diverse track variants better-bracket the truth) and intensity-bias variance.
3. Paper-space cost: Milton already has a 9-row table. If you swap weight-only -> Phase 5 rows, comparison is invisible; if you add Phase 5 rows alongside, table becomes 13-row and overflows. Cleanest middle ground: replace the 4 post-hoc weight-only rows with Phase 5 versions; one-line footnote noting weight-only Milton numbers are available on request.

## Compute cost (when the time comes)

- 14 inits x 10 members x 4 baselines = 560 Phase 5 forecasts. Roughly 4-8h total wall (mostly parallel via the existing per-week-helper pattern). Mirrors the existing Milton tracker submission `track_one_init.py` flow -- just feed the new forecasts in.
- Tracker (DetectNodes + StitchNodes) + summary table regen another 1-2h.
- Total: a half-day of compute on the GPU partition.

## Action sequence to follow when re-evaluating

1. Once all 3 (aurora/graphcast/sfno) Phase 5 production grids + per-model evals complete, re-run the SSR-vs-lead diagnostic for each model (`plot_ssr_phase5_aifs.py` already covers AIFS; extend to all 4 perturbed baselines as a 4-panel figure).
2. Verify each Phase 5 model shows the expected early-lead SSR lift WITHOUT inflating long-lead SSR (the AIFS signature: 24h 0.49 -> 0.67, 240h 0.96 -> 1.00).
3. If 4/4 show the lift: re-run Milton with Phase 5 baselines (replace weight-only rows). Update `milton_summary_table.tex` + F1-F3 figures + Sec 4.6 prose.
4. If 1+ baselines don't show the lift: heterogeneous Milton table is honest -- mix Phase 5 (where helped) + weight-only (where not). Explicit footnote per row.
5. If NONE show the lift (unlikely given AIFS result): keep Milton as-is; Phase 5 still works on the AIFS sub-narrative.

## Related

- [[milton-case-study]] -- existing case study setup + tracker pipeline
- [[phase5-perturbed-ic-plan]] -- the Phase 5 design + AIFS EXECUTED section
- [[spatial-mean-vs-pointwise-ssr]] -- the coherent-shift artefact Phase 5 fixes
- Paper Sec 4.6 (commit `00f3da8`) -- current Milton 8-way state
