# ai-models-ensembles - project instructions for Claude Code

Thin orchestration layer over [earth2studio](https://github.com/NVIDIA/earth2studio) +
[SwissClim_Evaluations](https://github.com/swiss-ai/SwissClim_Evaluations) for the SPW paper
("Stochastically Perturbed Weights", AMS AIES, submitted): post-hoc weight perturbation turns deterministic
ML weather models (Aurora, GraphCast, SFNO, AIFS) into ensembles, compared with trained-probabilistic models
(AIFS-ENS, Atlas, FCN3) and IFS-ENS. earth2studio loads models / fetches ICs / rolls out; SwissClim verifies;
this repo adds the perturbations, model registry, Slurm scripts, one GH200 container per model, analysis.

Humans start at `README.md` and `docs/` (HANDOVER, EXPERIMENTS, DATA, REPRODUCING, ENVIRONMENT, HPC). This file
and `.claude/` (see `.claude/README.md`) hold the condensed conventions and the logbook of findings.

## Status (2026-11-02, handover)
- Paper submitted (AMS AIES, resubmitted after technical check 2026-09-18), no reviews yet. Release `v1.0.0`
  (= `69e3f9f`) = reported results; do not move it, tag revision code `v1.1.0`+.
- Original author unreachable from December 2026. Data stays on Clariden, owned by `sadamov`, readable by group
  `s83`. Open items: `docs/HANDOVER.md` section 4.

## Python / interpreters
- Host venv (analysis, no torch/earth2studio): `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python`
  (`.venv` symlink). Exact pins: `env/host-requirements.txt`. Elsewhere: `uv venv --managed-python --python 3.11 && uv pip install
  -r env/host-requirements.txt && uv pip install -e . --no-deps`.
- Never `uv pip install <new>` into the shared venv without adding it to pyproject.toml first (broke certifi once).
- Never site a venv on `/iopsstor` scratch: the purge strips files but leaves dirs, imports then fail while
  `site-packages` looks populated.
- Milton tracking: same venv + TempestExtremes from the capstor miniforge base via `tools/milton/env.sh`.

## CSCS / Clariden
- Slurm account `ab016` (`a122` is frozen, QOS stop). Scripts use `--account="${AIENS_ACCOUNT:-ab016}"`.
  Partitions `debug` (1.5 h, 1 job), `normal` (12 h), `xfer` (data movement). 4x GH200 per node, aarch64 SM 9.0.
- `$STORE` = `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` (`AIENS_STORE`). Login-node `$STORE`
  points elsewhere; always use the AIENS default. Scratch `AIENS_SCRATCH` (`/iopsstor/scratch/cscs/sadamov`) is
  purged without notice: durable outputs go to `$STORE/analysis/` (`ANALYSIS` in `tools/_env.py`).
- IFS inputs: `/capstor/store/cscs/swissai/ab016/IFS/{ifs_ens,ifs_analysis_perturbed_ic,ifs_control}.zarr`
  (verified copies of the a122 originals). All paths: `tools/_env.py`, `docs/HPC.md`.
- No building/installing/zarr-heavy analysis on login nodes - CPU sbatch instead.

## Containers (6, one Dockerfile each)
- aurora, graphcast, sfno, fcn3, atlas, aifsens (aifs runs use aifsens.sqsh). NGC PyTorch 25.12-py3 (atlas 26.01),
  CUDA 13.1, Python 3.12. Git deps pinned to the commits in the built images; full freezes `containers/locks/`
  (`python containers/freeze_sqsh.py`).
- Build: `bash containers/submit_build.sh <model|all> [partition]` -> `$STORE/<model>.sqsh`. Test: `bash tools/run_model_tests.sh <model|all>`.
- With bind-mounted code use `python -m ai_models_ensembles.cli` (the `ai-ens` entrypoint uses the baked-in package).

## Run / evaluate
- Ablation inference: `scripts/submit_ablation.sh`; eval: `scripts/evaluate_ablation.sh` (count forecasts first:
  with <4 inits present the generated configs break the later intercomparison).
- Production inference: `scripts/submit_all_inference.sh`; eval: `scripts/evaluate_baselines.sh all <experiment>`
  (8 per-module sbatch; NEVER without an experiment: that submits 100+ jobs); intercomp `evaluate_baselines.sh intercompare`.
- SFNO SIGSEGV escape hatch: `PER_INIT=1 bash scripts/submit_all_inference.sh ...`. AIFS/AIFS-ENS: CDS ICs, keep the
  persistent cache `$STORE/e2s_cache_backup` (only copy, do not delete). History: `.claude/reference/project-conventions.md`.
- Eval modules: maps, wd_kde, energy_spectra (LSD), multivariate (bivariate), probabilistic (CRPS/SSR),
  deterministic (RMSE), ssim, fss. ETS dropped 2026-05-27.

## Ablation study
- 4 backbones: aurora, graphcast_operational, sfno, aifs. 4 mid-season inits (2023-05-15, 2023-08-15, 2024-02-15,
  2024-11-15), 10 members, 240 h, levels 500+850, seed 42.
- Phase 1 magnitude sweep (all weights); 2 layer groups (`0.01*sqrt(N_total/N_group)`); 2b refinement; 3/3b
  coarse-scale only; 4 (Fisher) deferred; 5 IFS-ENS perturbed ICs; 6 noise refreshed during rollout. Details:
  `docs/EXPERIMENTS.md`, `.claude/logbook/phase*_*.md`.
- `--layer`: index, range "10:50", fractions "0.0:0.33", named groups (`_MODEL_LAYER_GROUPS` in e2s_perturbation.py), all.

## Production picks (`.claude/logbook/calibration_winners.md`)
- `aurora_encoder` (encoder, sigma 0.025, Phase 2b), `graphcast_all` (all, 0.01, Phase 1), `sfno_modes10`
  (modes l<10, 0.25, Phase 3), `aifs_perturbed` (decoder, 0.0275, Phase 2). Multi-metric selection, not argmin CRPSS.
- Trained-prob: aifsens, atlas, fcn3; reference ifs_ens (stratified members 0,5,...,45). 8-way headline.

## Model weight structure (audit: `.claude/reference/checkpoint_perturbation_audit.md`)
- Aurora: 644 float32 tensors under net.* (backbone 0:594 | decoder 594:611 | encoder 611:644).
- GraphCast: 264 NPZ params (g2m 0:36, m2g 36:66, m2m 66:264); `_perturb_npz` skips model_config:/task_config: keys.
- SFNO: 87 ckpt tensors (79 float32 + 8 complex64 spectral filters at idx 0,10,...,70); .npy stats NOT perturbed.
- AIFS: 250 keys / 255M params; encoder/processor/decoder; normaliser (pre/post_processors) never perturbed.

## Zarr output format (critical for eval perf)
- Zarr v3, sharding codec, one shard per variable. Inner chunks = SwissClim `DESIRED_CHUNKS`: ensemble=full,
  level=1, init=1, lead=1, lat/lon=full (~40 MB). Mismatch -> SwissClim rechunks on read (hours). `_INNER_CHUNKS`
  (e2s_inference.py) and `INNER_CHUNKS` (tools/reshard_zarr.py) must stay in sync.
- zarr v3 template NaN-fill gotcha: `.claude/rules/zarr-v3-fill-value.md`.

## Data rules
- NEVER delete a `forecast.zarr` (hours of GPU each). Superseded runs: keep or move to `_archived/`.
  `_e2s_work/` is a temporary merge workspace (deletable only when no inference runs on that init).
- Every dataset is described in `docs/DATA.md` + `docs/data/catalog.csv`; after creating a new experiment add a
  catalog line and rerun `python tools/inventory_store.py --dust --publish` (sizes via `dust`, ~2 min).
- Figure/table inputs vendored in `tools/data/store/` (mirror of `$STORE` paths); refresh it when a mirrored
  file is regenerated (`tools/data/README.md`). After any figure change diff the legend against the committed PDF.
- Geopotential is plotted in m2/s2, never as height (`.claude/rules/plot-units.md`).

## Reproducibility
- Seeds: weights `seed + member_id`, IC `seed + member_id + 10_000`, rollout `_seed_rngs(seed + member_id)`;
  cudnn.deterministic, benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. GraphCast (JAX/XLA) and AIFS are NOT
  bit-reproducible on GPU (~0.02 K mean 2t at 24 h, far below the spread).
- Phase 6 fresh hooks use `_mix_seed` (splitmix64); the pre-2026-07-10 `seed+step` seeding was a bug (`*_p6c` runs).

## Paper
- Separate private repo (Overleaf mirror), nested locally at `ai-models-ensembles-paper/` (gitignored). Its rules
  and the paper-process notes are local-only (`.claude/*/paper/`, gitignored); a copy is on the ab016 store
  (`docs/HANDOVER.md`). Figures: generators write to `figures/` here, the paper repo's `sync_figures.sh` copies them.

## Memory & filing rules
- `.claude/` is tracked (public repo!): no credentials, private emails, personal or journal-process details
  there. Paper-process notes go to the gitignored `.claude/{logbook,reference,rules}/paper/`.
- Project facts/conventions -> this file or `.claude/rules/` (`paths:` frontmatter if file-scoped); findings,
  decisions, lessons -> a dated `.claude/logbook/YYYY-MM-DD_<slug>.md` (one-line summary on top).
- Keep this file under ~200 lines; overflow into `.claude/rules/` or `.claude/reference/`.
- Claude Code's auto-memory (`~/.claude/projects/.../memory/MEMORY.md`) is per user and not shared; the
  2026-11-02 snapshot is `.claude/memory/MEMORY.md`. Keep it a slim pointer + in-flight tracker.
