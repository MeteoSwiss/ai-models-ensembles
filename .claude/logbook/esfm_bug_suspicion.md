---
name: esfm-bug-suspicion
description: "2026-06-04: ESFM (step-10000) baseline results look buggy. The Milton case-study tracker detected the storm in <5% of (init, member) pairs at lead 240 h, far below the 52-88% range of the other 6 baselines, and the 8-way headline CRPSS @ 240 h is -0.127 -- below climatology. User may want to remove ESFM from the paper. KEEP it for now; revisit before next major paper iteration."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Suspicion

Several signals point to ESFM-step-10000 not being a usable baseline at lead 240 h:

1. **Milton case-study tracker:** under 5% of the 140 (init, member) pairs produced a track touching Milton's lifecycle window, vs 52-88% for the other 6 baselines. The released checkpoint at step 10 k does not appear to produce a TC-relevant MSL field consistent with the standard TempestExtremes thresholds.
2. **8-way headline CRPSS @ 240 h = -0.127.** Negative means worse than climatology. All other baselines are between +0.30 and +0.41 at the same lead. See [[esfm-baseline-integration]] for the integration trace.
3. **Single-checkpoint comparison:** ESFM is the only baseline shown at a checkpoint that is still mid-training (the paper authors note step 10 k is not yet converged, with their reported numbers at later steps).

## Why keep it for now

User decision 2026-06-04: keep ESFM in Tab. baselines + headline figure + Sec 4.2 + Sec 4.3 prose UNTIL we have a concrete reason to drop it. The reasons might be:
- ESFM authors release a converged checkpoint and the result still under-performs.
- We confirm the step-10 k decision was the right call but a reviewer pushes back.
- We need page count and decide ESFM doesn't earn its space.

Also flagged in [[milton-case-study]] -- the Milton case study Sec 4.6 explicitly omits ESFM with a footnote citing "under 5% detection at lead 240 h, omitted".

## What to do if we drop it later

The integration touches: abstract, Sec 2, Sec 3.1, Sec 4.2, Sec 4.3, Tab. 2, Tab. 4 (currently the 8-way table), Tab. 6, App D, headline_crpss_vs_lead_8way figure. See [[esfm-baseline-integration]] for the full thread. Removal would be a clean revert of the threading commits plus a Tab. naming swap back from "8-way" to "7-way" in body text + caption.

## How to apply

- For Sec 4.6 Hurricane Milton case study draft (2026-06-04): keep the existing footnote ("ESFM (step 10 k) omitted because the released checkpoint does not produce a TC-relevant MSL field..."). Do NOT remove ESFM from any other figure or table yet.
- For tracker / detection analyses in any future case study: same footnote pattern; no need to re-derive.
- Before final submission: revisit this memory and decide whether ESFM is dropped or kept.

## Related

- [[esfm-baseline-integration]] -- the original 2026-06-03 integration trace, threading ESFM through 8 paper locations.
- [[milton-case-study]] -- where the bug suspicion first surfaced via tracker stats.
- [[paper-state-2026-06-03-evening]] -- the EOD paper snapshot that locked in ESFM-as-8th-baseline.
