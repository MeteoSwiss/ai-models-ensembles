---
name: esfm-baseline-integration
description: "2026-06-03 evening: ESFM added as the 8th production baseline (Ozdemir2026esfm, arXiv:2605.00850). Step-10000 checkpoint of ESFM_s_nm_10ens. Data layout differs from other baselines (single consolidated zarr with init_time dim vs per-init forecast.zarr directories). CRPSS@240h = -0.127 (intermediate-checkpoint, not fully trained). Paper threading: abstract + Sec 2 + Sec 3.1 + Sec 4.2 + Tab. 2 + Tab. 4 + Tab. 6 + App D + headline figure all 7-way -> 8-way."
metadata:
  node_type: memory
  type: project
  originSessionId: edbdb4e6-977a-4cd4-939c-3a6968e745a1
---

## What ESFM is

Earth System Foundation Model of \citet{Ozdemir2026esfm} = Aurora 3D Swin U-Net backbone with three extensions:
1. Per-variable tokenisation
2. Axial attention across variable tokens
3. AdaLN-Zero per-member modulation (`z' = z + gamma(c_iens) LN(z) + beta(c_iens)` with gamma, beta initialised to 0)

Trained jointly with MAE + almost-fair-CRPS (afCRPS_alpha=0.95, weight=1.0). 110M parameters. Four-stage training schedule:
1. 92k steps CMIP6 pretrain
2. 40k steps knowledge distillation
3. 100k steps ERA5 finetune
4. 10k steps ensemble finetune (loaded from a step-10000 pretrained checkpoint at `esfm_small_nm_1aa1et_esfmi_scposemb_llr`)

**The repo author is an ESFM co-author**, so internal swiss-ai SDSC training logs are accessible for the camera-ready cost numbers.

## What we use

Public eval CSVs + raw forecasts at:

- Forecast zarr: `/capstor/store/cscs/swissai/a122/ESFM_related/trained_models/ESFM_s_nm_10ens/predictions_rollout_subset_step10000.zarr`
- SwissClim eval (already done): `/capstor/store/cscs/swissai/a122/ESFM_Results/ESFM_s_nm_10ens/Evaluations_rollout/2023-2024/step10000/{deterministic,probabilistic}/`

The checkpoint is **ESFM_s_nm_10ens at training step 10,000** = an intermediate ensemble-finetune snapshot (10k of ~50k total ensemble-finetune steps), not a fully trained production model. CRPSS values are correspondingly worse than the other 7 baselines.

## Data layout (different from our other baselines)

| | ESFM | Other baselines |
|---|---|---|
| Layout | Single consolidated Zarr v2 archive | Per-init `<YYYYMMDD_HHMM>/forecast.zarr` |
| init_time | non-singleton coordinate (112 inits in one file) | singleton (one file per init) |
| Latitude | 720 (drops -90 pole, like Aurora) | 721 for most, 720 for aurora |
| Vars | 2m_t, 10m u/v, MSL, gpot, t, u/v, q at 500+850 | Same |
| Inits | 112, exact match to our production grid | Same |
| Members | 10 | Same |
| Lead | 6 h to 360 h (60 steps × 6 h) | 240 h (40 × 6 h) for most |

The mismatch required extending two of our standalone tools to handle the consolidated layout:
- `tools/spatial_mean_ssr.py` -- per-init loop now iterates over all init_time slices when the zarr has them
- `tools/energy_variogram_score.py` -- per-init loop refactored into a generator yielding per-init xarray views

Both also try `consolidated=True` first (ESFM's Zarr-v2 + .zmetadata), falling back to `consolidated=False`.

## Headline numbers (CRPSS, 6 paper vars, MSL excluded for ifs_ens bug)

| Lead | ESFM | Best (AIFS-ENS) | Worst non-ESFM (sfno_modes10) |
|---|---|---|---|
| 72 h | 0.572 | 0.771 | 0.654 |
| 120 h | 0.320 | 0.665 | 0.514 |
| 240 h | **-0.127** | 0.402 | 0.300 |
| 360 h | **-0.311** | 0.270 | 0.231 |

ESFM crosses below zero (worse than climatology) around lead 160 h and saturates near -0.3 at 360 h. Expected for a step-10000 intermediate checkpoint.

## Paper threading (what changed where)

| Location | Change |
|---|---|
| Abstract | added ESFM to the production-grid baseline list (was: "AIFS-CRPS, FourCastNet 3, Atlas, and the operational IFS-ENS"; now adds ESFM) |
| Sec 2.2 Related Work | new "Foundation-model conditioning" paragraph describing ESFM architecture and probabilistic head; "Three approaches" -> "Four approaches" |
| Sec 3.1 Methods/Models | dedicated ESFM intro paragraph (110M params, four-stage schedule, step-10000 snapshot, consolidated-zarr layout); "Three trained-prob / classical" -> "Five (foundation-model / classical)" |
| Sec 4.2 Headline | subsection title and body: "seven-way" -> "eight-way" |
| Sec 4.X (uplift / 7-way) text | "seven-way" -> "eight-way" + "seven baselines" -> "eight baselines" everywhere |
| App D Reproducibility | ESFM consolidated-zarr exception added to the directory layout description |
| Tab. 2 baselines | ESFM row in trained-prob block + new `esfmC` teal color in palette |
| Tab. 4 (was Tab. 4 headline 7-way) | renamed to `headline_8way_table.tex` with ESFM row; label `tab:headline7way` -> `tab:headline8way` (8 references swept) |
| Tab. 6 inference cost | ESFM row with parameter-scaled estimates ($\sim$10 K GPU-h, $\sim$25 s/mbr, $\sim$6.5 NH) and new $^{\ddag\ddag}$ footnote |
| `paper_refs.bib` | `Ozdemir2026esfm` BibTeX entry |
| `headline_crpss_vs_lead_8way.{pdf,png}` | new figure; main.tex includegraphics swapped |

The tier1b spatial-mean SSR figure (`tier1b_7way_spatial_mean_ssr.pdf`) is intentionally left as 7-way because the standalone spatial-mean SSR tool needs to land on ESFM first; job 2465814 is queued and the figure will be regenerated once that lands.

## Open items

1. **Exact cost numbers for Tab. 6** -- currently parameter-scaled estimates. Swiss-ai SDSC training logs (accessible to the repo author as co-author) will give the real wall-clock for camera-ready.
2. **Spatial-mean SSR + ES + VS on ESFM** -- jobs 2465814 + 2465815 queued. Will produce:
   - `$STORE/baselines/esfm/spatial_mean_ssr/spatial_ssr.csv`
   - `$STORE/baselines/esfm/energy_variogram_score/score.csv`
3. **Tier1b spatial-mean SSR figure regenerate to 8-way** once 2465814 lands.
4. **ES + VS supplement table update to 8-way** once 2465815 lands.

## Commits

| Commit | Repo | What |
|---|---|---|
| `9d5748c` | paper | abstract + Sec 2 + Sec 3.1 + Sec 4.2 + App D thread |
| `858334e` | code | Tab. 2, Tab. 4 (new), Tab. 6 placeholder |
| `74d7a35` | code | Tab. 6 placeholders -> parameter-scaled estimates |
| `fc0f3c5` | paper | headline-CRPSS figure include 7-way -> 8-way |
| `cf3a891` | code | 8-way headline figure + consolidated-zarr support in spatial_mean_ssr + energy_variogram_score |
| `9a1efa6` | code | venv pre-commit fix + ruff-format pass |

## Related

- [[feedback-papers-to-zotero]] -- the standing rule to add new bib entries to Zotero
- [[paper-folder-inventory]] -- paper repo structure, two-remote split
- [[paper-state-2026-06-03-evening]] -- the snapshot this integration extends (now superseded by this entry on the baseline count)
- [[metric-audit]] -- canonical metric definitions; ESFM uses the same conventions
