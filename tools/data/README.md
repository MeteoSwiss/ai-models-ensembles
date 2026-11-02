# tools/data/

Small inputs committed to git so the figures and tables can be rebuilt without access to CSCS.
How to use them: [docs/REPRODUCING.md](../../docs/REPRODUCING.md).

## Cached results of long jobs

| File | Content | Written by | Read by |
|---|---|---|---|
| `crps_clim_eval_1990_2019.json` | CRPS of the 1990-2019 ERA5 climatology ensemble on the 112 production inits: `{channel: {lead_h: CRPS}}`, 12 channels (2t, msl, z/t/u/v/q at 500/850 when present), leads 0-360 h | `tools/compute_climatology_crps_vs_eval.py` (`tools/submit_compute_climatology_vs_eval.sh`, 13-16 h over HTTPS) | headline figure/tables, Tab. C1, block bootstrap |
| `crps_clim_eval_1990_2019_per_init.json` | same, per init: `{channel: {lead_h: {init_iso: CRPS}}}` | same job | rival-validation table, common-sample CRPSS |
| `crps_clim_eval_ablation_1990_2019.json` | same for the 4 ablation inits, leads 0-240 h | same job (ablation tag) | calibration table, uplift, seed robustness |
| `channel_scale_1990_2019.json` | fixed per-channel standardisation scale (1990-2019 climatological std) for ES/VS/SIGK: `{channel: scale}` in physical units | `tools/compute_channel_scale.py` (`tools/submit_channel_scale.sh`) | `energy_variogram_score.py`, `signature_kernel_score.py`, all fixed-scale jobs, z500 plume |
| `persistence_mae_112inits.json` | MAE of persistence forecasts on the production grid: `{channel: {lead_h: MAE}}`, leads 6-360 h | `tools/compute_persistence_mae.py` (`tools/submit_compute_persistence.sh`) | headline figure (persistence reference) |
| `ic_decomp_4bb.csv` | spread / spread_rms / CRPS per (backbone, arm in {weight, ic, weight+ic}, variable, level, lead) on the inits common to all arms (112) | `tools/ic_weight_decomposition.py` (`tools/submit_ic_weight_decomposition.sh`) | `tools/make_ic_decomp_table.py` |
| `sigk_production.csv` | signature-kernel score of the production baselines at 240 h (128 pixels, 112 inits, 10 members) | `tools/signature_kernel_score.py` | reference values for `tools/submit_esvs_sigk_ic_decomp.sh` |

Units are those of the variables (K, Pa, m2 s-2, m s-1, kg kg-1); CRPS and MAE are cos-latitude-weighted
global means.

## store/ - mirror of the store files the figure and table scripts read

`store/` reproduces, with identical relative paths, exactly the files under
`/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` that the figure and table scripts open
(60 MB, ~290 files: intercomparison CSVs, per-baseline evaluation CSVs/NPZ, spatial-mean SSR CSVs,
spread-error CSV, per-init CRPS, Milton F1 tracks and z500 plume cache, ens-anomaly point/m50 NPZ, and the
regenerated rank-histogram and fixed-scale ES/VS/SIGK outputs). Point the scripts at it with

```bash
export AIENS_STORE=$PWD/tools/data/store AIENS_SCRATCH=$PWD/tools/data/store/analysis
```

The list of mirrored files is `store/MANIFEST.txt`. To refresh the mirror after rerunning a job on Clariden,
copy the new files with the same relative path, e.g.

```bash
S=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
rsync -a --files-from=tools/data/store/MANIFEST.txt $S/ tools/data/store/
```

Do not put anything here that is not read by a committed script, and keep single files well below
GitHub's 100 MB limit (largest today: `analysis/per_init_crps_production_v2.csv`, 8 MB).
