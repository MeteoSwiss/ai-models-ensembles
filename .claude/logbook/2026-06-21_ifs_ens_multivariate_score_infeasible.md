# IFS-ENS multivariate ES/VS/SIGK are NOT fairly computable - keep excluded (do not re-derive)

Date: 2026-06-21 (attended, Opus 4.8). Context: user asked to add ES/VS/SIGK for IFS-ENS to
Tab. production-metrics (4). Initially authorised ("add NaN handling, compute it") on my wrong framing
of "~20-30% spatial NaN gaps". Investigated the actual data and OVERRODE: IFS-ENS stays "-".

## The finding (verified, reusable - don't redo this dig)
`ifs_ens.zarr` = `/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr`
(dims: init_time=112, ensemble=50, lead_time=61 @6h to 360h, level=[500,700,850], lat=721, lon=1440).

The NaN gaps are NOT spatial - they are **whole missing (variable, lead, init) slabs**, all-or-nothing:
- For a given (init, lead) a variable is either fully present for ALL 50 members or fully NaN for all.
  (init 0 -> 0/50 complete at 240h; init 50 -> 50/50.)
- Each variable is individually present in ~83-91 of 112 inits at 240h, but the missing sets differ
  per variable, so **jointly-complete inits (all 7 paper vars present at one lead) are scarce**:
  - 120h: 21/112 inits complete.
  - 240h: 23/112 inits complete.
  - SIGK path (all 7 vars at ALL ~21 path-leads 0,12,..240): essentially 0 complete inits.

## Why this kills the multivariate scores (but not the per-variable ones)
ES / VS / SIGK stack all 7 vars x 2 levels into one vector per pixel, so an init contributes only if
EVERY layer is present. That leaves ~20% of inits (non-random subset) for ES/VS and ~none for SIGK -
not comparable to the 112-init scores of every other baseline. So a number would be misleading.
Per-variable metrics (CRPSS etc.) are fine via skipna because each variable is scored at the leads it
has - that is why IFS-ENS appears in Tab. headline8way / per-variable-crpss (skipna) but is "-" in the
multivariate columns of Tab. calibration (2) and Tab. production-metrics (4).

## Consequence / what was done
- Kept IFS-ENS excluded from ES/VS/SIGK (and W1) in both tables. The paper's existing rationale already
  states it correctly (main.tex appendix C: "the multivariate stack requires consistent variable
  presence across all members and inits") - no caption change needed.
- Did NOT patch the scorers for NaN (ablation/production data is clean; a patch would only have served
  IFS-ENS and risked the published numbers). `energy_variogram_score.py` / `signature_kernel_score.py`
  remain NaN-naive by design.
- If anyone revisits: computing a reduced-n, footnoted IFS-ENS ES/VS on the ~21-23 complete inits is
  possible but explicitly rejected here as not directly comparable; SIGK is not feasible at all.

## Also this session (committed, not pushed)
main `0d8083f` (13 figure regens + Tab. 2 non-winner fill + Tab. 4 ES/VS/SIGK@120+240 with bolding +
submit/generator edits); SwissClim research `4a4975f` (bivariate grey contours + legend trim). Paper
main.tex was already committed by the concurrent 2026-06-21 nightly (`01406c3`) - the nightly did the
same figure-fix main.tex work overnight plus an SSR^pix formula refinement; my stale 1-line copy was
discarded. ES/VS/SIGK from job 2579121 (`scratch/table_metrics/`); the 4 ablation winners reproduced
the published calibration-table cells exactly; aifs_perturbed production ES@240 corrected 1.050 -> 1.054.
Finding: the temporal-path SIGK within-model optimum often falls on an all-weight/full-layer-group phase,
not the coarse-scale production pick (frozen coarse perturbation is temporally over-coherent).
