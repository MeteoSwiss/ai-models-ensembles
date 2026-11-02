Summary: ran the unperturbed deterministic AIFS (sigma=0, 1 member) on the 112-init production grid and added it as a grey line (single-member CRPS = MAE) to a TALK-ONLY copy of the headline CRPSS figure: 0.86 at 6 h, crosses climatology ~180 h, -0.29 at 240 h, -0.66 at 360 h (aifs_perturbed 0.221 at 240 h). Paper figure untouched. Persistence JSON had been scratch-purged; recomputed + vendored to tools/data/.

# Unperturbed AIFS on the headline CRPSS figure (2026-10-13, attended, talk-only)

## Request + scope
- User asked for "the unperturbed deterministic AIFS (sigma=0, single-member CRPS = MAE) as a grey line to
  the AIFS CRPSS-vs-lead panel". No AIFS-specific panel exists anywhere (code, paper, `visuals` talk
  builder); the only CRPSS-vs-lead plot is the single-panel headline figure. User chose: 112-init production
  run, but as a SEPARATE figure for presentations, NOT the paper version.
- Before this, unperturbed AIFS existed only on the 4-init ablation grid
  (`ablation/phase1/aifs/<init>/mag_0_layer_all/forecast.zarr`) and was never evaluated, so
  `tools/compute_perturbation_uplift.py` prints n/a for AIFS (the other 3 backbones have mag_0 MAE in
  `allphases/<model>/intercomparison/deterministic`). Still unfilled - not needed for the talk figure.

## Runs
- Inference: new arm `aifs_det` in `scripts/submit_all_inference.sh` (`--members 1`, no weight flags; new
  per-model `MEMBERS` map, default `NUM_MEMBERS=10`). DATA_SRC=cds -> mounts the capstor
  `$STORE/e2s_cache_backup` (IC fetch 5-13 s/init, cache hits). Jobs 3653150-52, 3653154-58 (8 week helpers),
  all COMPLETED in ~41 min each. 112/112 zarrs validated (shape ens=1, lead=61, 9 vars, no NaN at 360 h).
  Output `$STORE/baselines/aifs_det/<YYYYMMDD_HHMM>/forecast.zarr` (NEVER delete).
- Eval: `aifs_det` registered in `scripts/evaluate_baselines.sh` (MODELS, MODEL_KIND, case list); ran ONLY
  the deterministic module: `bash scripts/evaluate_baselines.sh aifs_det deterministic` -> job 3656106. SwissClim
  disables probabilistic for ensemble size 1 automatically. The long CSV
  `baselines/aifs_det/eval/deterministic/deterministic_metrics_by_lead_long_*.csv` was complete (12 keys x 61
  leads, no NaN) while the job was still writing per-variable line plots.
- MAE vs CRPS consistency: both use `scores.functions.create_latitude_weights` (cos-lat); CRPS is
  WBX CRPSEnsemble spatial-weighted mean, MAE `scores.continuous.mae` with the same weights. Same
  convention as `compute_perturbation_uplift.py` and the persistence curve.

## Figure
- `tools/plot_headline_crpss_vs_lead_8way.py --with-aifs-det` -> `figures/talk/headline_crpss_vs_lead_8way_aifs_det.{pdf,png}`
  (`--out` overrides). Grey `#A6A6A6` solid (IFS-ENS is `#7F8C8D` dotted), label `AIFS ($\sigma=0$)`.
  Y-axis lower limit extended to fit the grey line (-0.8), so persistence now shows its diurnal zigzag down to
  -0.8 before leaving the axis ~120 h (paper cuts at -0.1). Default (no flag) output is the unchanged paper
  figure; the 8 original lines reproduce the paper 240 h values (AIFS-ENS 0.256; gaps 0.036/0.089/0.122/0.131).
- Persistence: `persistence_mae_112inits.json` was gone from scratch (purge). Recomputed with
  `sbatch tools/submit_compute_persistence.sh` (job 3653164, 1 h 07 min, deterministic so identical to the
  paper curve), copied to `tools/data/persistence_mae_112inits.json`; the plot now prefers that path.

## Commits
- Code `e04f1a7` (main, unpushed at time of writing).

## Not done / open
- (talk-deck logistics kept in the local notes)
- Optional: hide persistence in the talk variant if the zigzag is too busy for a slide (user not asked yet).
