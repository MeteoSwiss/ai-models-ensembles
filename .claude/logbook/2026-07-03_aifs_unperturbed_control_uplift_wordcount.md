# 2026-07-03 ATTENDED: unperturbed-AIFS control generated, model-soups uplift now 4-backbone (monotone trend), AMS word-count confirmed + memory fixed

One-line: the unperturbed (sigma=0) AIFS deterministic control had NEVER been run; generated it, verified
AIFS is the strongest deterministic backbone, added the 4th row to `tab:uplift`, recast the model-soups
gain claim as a monotone 4-point trend, and confirmed + corrected the AMS word-count rule/method.

## The gap (why AIFS was missing from tab:uplift)
- `tab:uplift` (model-soups deterministic gain) had only 3 rows (aurora/graphcast/sfno). Not an oversight
  of the table: the Delta columns are relative to the unperturbed control `phase1/<model>/<init>/mag_0_layer_all`,
  and NO unperturbed AIFS control existed anywhere on disk.
- Confirmed: aurora/graphcast/sfno each have 5 `mag_0_layer_all` control init-dirs; AIFS had 0. AIFS's
  phase1 sweep starts at sigma=0.003 (`PHASE1_MAGNITUDES_aifs="0.003 0.01 0.03"`), never sigma=0. AIFS was
  added late (2026-05-29) and `run_phase1_unperturbed` (submit_ablation.sh:302) had already been run for the
  original three before AIFS joined. No control in baselines/ or scratch either. Not recoverable from any
  existing run (all AIFS runs carry sigma>0).

## Fix: generated the control
- `bash scripts/submit_ablation.sh phase1_unperturbed aifs` -> job 2671802, COMPLETED in 31 min (exit 0).
- Output: `$STORE/ablation/phase1/aifs/<init>/mag_0_layer_all/forecast.zarr` for all 4 inits (4.2G each,
  ensemble=1, 41 lead steps 0-240h, levels 500/850, finite ICs, sane 2t). These are the missing
  deterministic anchor - do NOT delete.
- CDS note: the ablation path does NOT mount the persistent CDS cache, and the 4 ablation inits (mid-month
  15th) are NOT in the production cache (early-month days 2-8), so the control did a fresh CDS fetch
  (variable ~26min-3h, dominated by the fetch, occasional transient retry). Keep such runs on `normal`
  (12h), not `debug` (1.5h could time out on a slow fetch).

## Verified result (AIFS is the strongest deterministic backbone)
- Deterministic control (mag_0) CRPSS @240h, var-mean (MAE as fair-CRPS at M=1, higher = stronger parent):
  AIFS -0.321 > GraphCast -0.362 > Aurora -0.521 > SFNO -0.588. aurora/gc/sfno reproduce the allphases CSV
  values exactly (cross-validation of the zarr-direct calc).
- Ensemble-mean gain @240h (Delta-mean, more negative = bigger gain): SFNO -22.4% (most) > Aurora -19.9% >
  GraphCast -15.8% > AIFS -14.8% (least). Perfectly MONOTONE against parent strength: weaker parent -> larger
  smoothing gain, at every step. So "gain largest where the prior is weakest" is now a 4-point trend, not a
  3-point observation, and the strongest-backbone endpoint is AIFS (was wrongly GraphCast).

## tab:uplift 4th row (main.tex, inline table)
- `aifs_perturbed` (pick = phase2 `mag_0.027500_layer_decoder`): CRPSS 0.793/0.537/0.204; Delta-mbr
  +11.5/+25.0/+12.7%; Delta-mean +0.8/-1.1/-14.8% at 24/120/240h. Existing 3 rows reproduced EXACTLY
  (validates the regen). Sentence recast: "falls monotonically from SFNO (weakest, gains most) to AIFS
  (strongest, gains least)."

## Tools changed (code repo, committed b2ee5a4 by a concurrent process, PUSHED)
- `tools/compute_perturbation_uplift.py`: added aifs PICK; optional positional lead arg
  (`python tools/compute_perturbation_uplift.py 24|120|240`, default 240); guard when mag_0 CRPSS is absent
  (delta -> None, prints n/a) so it runs even before a control eval populates the deterministic CSV.
- `tools/compute_member_vs_mean_rmse.py`: added aifs PICK.
- Regen path: `sbatch tools/submit_uplift_multilead.sh` (now includes aifs) OR the login-node loop (small,
  I/O-bound ~2-3 min/lead, no OOM - it reads one lead-slice at a time, NOT whole zarrs). Multilead CSVs land
  in `/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/uplift_multilead/uplift_L{24,120,240}.csv`.

## Paper commit
- Paper `2f92fb4` (sadamov): tab:uplift AIFS row + monotone sentence + abstract spread-skill
  rephrase + IC-augmentation scope fill + bundled the user's in-flight prose/math-notation polish (user chose
  "commit everything"). tectonic PASS. PUSHED. Code already at `b2ee5a4` (MeteoSwiss), PUSHED.

## AMS word count
(paper-process detail kept in the local notes)

## Not done / follow-ups
- If a deterministic-AIFS number is ever wanted on the 112-init PRODUCTION grid (headline), no production
  control exists either - would need a separate `mag_0` production run.
- Optional: run the deterministic eval/intercomparison on the new AIFS control to add `mag_0_layer_all` to
  the allphases aifs deterministic CSV (not needed for tab:uplift, which is zarr-direct + CRPSS-pert).
