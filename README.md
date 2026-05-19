# AI Model Ensembles for Weather Forecasting

Compare AI weather forecast models — GraphCastOperational, SFNO, Aurora
(deterministic, with post-training weight perturbation) against FCN3, Atlas
(probabilistic) and the on-disk IFS ENS physical baseline. AI models are
initialised from ARCO ERA5; IFS ENS is verified directly.
Verification via
[SwissClim Evaluations](https://github.com/swiss-ai/SwissClim_Evaluations).

This repo is a thin orchestration layer over [NVIDIA earth2studio](https://github.com/NVIDIA/earth2studio):

- **earth2studio** handles model loading, IC fetching, and rollout.
- **swissclim-evaluations** handles deterministic + probabilistic
  verification, plotting, and intercomparison.
- This repo wires them together: a curated 5-model registry, IC + weight
  perturbation helpers, Slurm scripts, and a GH200 container build.

## Model registry

| Model | Class | Resolution | Step | Role |
|---|---|---|---|---|
| `graphcast_operational` | `GraphCastOperational` | 0.25° | 6 h | deterministic, weight-perturbed |
| `sfno` | `SFNO` (FCNv2) | 0.25° | 6 h | deterministic, weight-perturbed |
| `aurora` | `Aurora` | 0.25° | 6 h | deterministic, weight-perturbed |
| `fcn3` | `FCN3` | 0.25° | 6 h | probabilistic, re-seeded per member |
| `atlas` | `Atlas` | 0.25° | 6 h | probabilistic, re-seeded per member |
| `aifsens` | `AIFSENS` | 0.25° | 6 h | probabilistic, re-seeded per member |

`ai-ens models` prints the live registry. Adding a model means appending one
`ModelSpec` in [ai_models_ensembles/e2s_models.py](ai_models_ensembles/e2s_models.py).

## Perturbation strategy

Weight perturbation is applied in four ablation phases, each progressively
more physically-motivated.

### Phase 2 — architectural layer groups

Per model, partition the learnable weights into encoder / processor /
decoder groups (model-specific naming). Apply Gaussian multiplicative
noise to a single group at a time, with `sqrt(N_total / N_partial)`
variance scaling so the partial-group output spread matches the full-weight
reference at `σ_full = 0.01`.

![Phase 2 schematic](figures/perturbation_schematic.png)

### Phase 3 — physics-inspired coarse-scale targeting

Perturb only the parameters responsible for spatial scales `λ ≳ 3000 km`
(planetary / large-synoptic). Each model uses a different mechanism for the
same physical objective: spectral-mode sub-slice (SFNO), bottleneck-layer
selection (Aurora), or runtime edge-embedding hook on long mesh edges
(GraphCast).

![Phase 3 schematic](figures/phase3_schematic.png)

Tensor counts and group definitions were verified empirically from
checkpoint dumps; see [tools/dump_*_keys.py](tools/) and the audit notes
in the team memory.

## Initial conditions

All five AI models start from **ARCO ERA5** by default. Reasons:

- All five were trained on ERA5; ERA5 ICs are in-distribution.
- The on-disk IFS download ([sadamov/ifs_download](https://github.com/sadamov/ifs_download))
  doesn't include the t+0 analysis or the full variable list any of these
  models need (missing `u100m`, `v100m`, `sp`, `tcwv`, `w`, `sst`,
  accumulated fluxes; missing 600 hPa).
- The IFS ENS forecast on disk plays the role of the physical baseline; it's
  plugged directly into SwissClim verification, not re-initialised.

To use a different IC source, pass `--data-source` to `ai-ens infer` or
edit the constant at the top of the relevant slurm script:
`arco | cds | gfs | ifs | ifs_ens | wb2 | file:/path | ifs_analysis:/path`.
The `ifs_analysis:` source is still wired up (extracts `lead_time=0` from a
SwissClim-format IFS zarr) for the day you have a download with t+0 analysis
and the full variable set.

## Quickstart

The heavy ML stack (`torch`, `earth2studio`, `jax`, model-specific deps) lives
inside per-model containers under [containers/](containers/). The host venv
holds only what's needed for editor support, slurm orchestration, and
post-hoc analysis of zarr outputs.

1. Build the per-model container(s)

```bash
bash containers/submit_build.sh <model|all>    # writes $STORE/<model>.sqsh
```

2. (Optional) Minimal host venv for editor / scripting

```bash
# Place the venv on fast scratch and symlink into the repo
VENV_DIR=$SCRATCH/venvs/ai-models-ensembles
uv venv --python 3.11 "$VENV_DIR"
ln -s "$VENV_DIR" .venv
source .venv/bin/activate

uv pip install \
  "typer>=0.12" "rich>=13.0" \
  "zarr>=3.0" "xarray>=2024.10" dask numcodecs \
  "numpy>=2.0" pandas matplotlib \
  "pyyaml>=6.0" netCDF4 cfgrib \
  "pytest>=8.0" "ruff>=0.6" "pre-commit>=3.7" ipython
uv pip install -e . --no-deps
```

`earth2studio`, `swissclim-evaluations`, `torch`, `jax`, etc. are
deliberately not installed on the host - run anything that needs them inside
a container via `srun --container-image=$STORE/<model>.sqsh ...`.

3. List available models (works on the host venv)

```bash
ai-ens models
```

4. Submit inference and evaluation jobs

```bash
bash scripts/submit_all_inference.sh          # probabilistic baselines (fcn3/atlas/aifsens)
bash scripts/submit_ablation.sh phase1        # weight-perturbation ablation
bash scripts/evaluate_ablation.sh phase1      # SwissClim verification
bash scripts/evaluate_ablation.sh intercompare phase1
```

See [scripts/README.md](scripts/README.md) for the full submitter docs.
Per-run parameters are constants at the top of each script - there is no
shared `config.sh`.

## Requirements

- Linux. Slurm + pyxis + enroot recommended for GH200.
- GPU (CUDA 12.x).
- For `arco` / `wb2` data sources: outbound HTTPS to GCS.
- For `cds`: `~/.cdsapirc` credentials.
- `envsubst` (gettext-base) for SwissClim YAML rendering.
- `podman` + `enroot` if building the container.

## CLI usage (Typer)

```bash
ai-ens --help
ai-ens models           # list available earth2studio models
```

All of the following must run inside a container (they need
`earth2studio`). Wrap each in `srun --container-image=$STORE/<model>.sqsh ...`.

```bash
# Single deterministic forecast
ai-ens infer --model graphcast_operational --init 2024-02-15T00:00 \
   --lead-hours 240 --members 1 --data-source arco \
   --output $STORE/test/gc_op/forecast.zarr

# 10-member IC-perturbed ensemble
ai-ens infer --model sfno --init 2024-02-15T00:00 --lead-hours 240 \
   --members 10 --ic-magnitude 0.005 --data-source arco \
   --output $STORE/test/sfno_ic/forecast.zarr

# 10-member weight-perturbed ensemble (per-member NGC mirror)
ai-ens infer --model graphcast_operational --init 2024-02-15T00:00 \
   --lead-hours 240 --members 10 --weight-magnitude 0.01 --layer all \
   --data-source arco --output $STORE/test/gc_w01/forecast.zarr

# Verification (delegates to swissclim-evaluations)
ai-ens verify --config /abs/path/to/swissclim_eval.yaml

# Intercompare a set of swissclim_* output roots
ai-ens intercompare /path/A/swissclim_graphcast_operational \
                    /path/B/swissclim_sfno \
                    /path/C/swissclim_aurora \
                    --label GC --label SFNO --label Aurora
```

`--layer` accepts a single weight-tensor index, a `start:end` range,
fractional `0.0:0.33`, named architectural groups, or `all`. The named
groups are model-specific (defined in `_MODEL_LAYER_GROUPS` in
[ai_models_ensembles/e2s_perturbation.py](ai_models_ensembles/e2s_perturbation.py)):

| Model | Named groups |
|---|---|
| `aurora` | `encoder`, `backbone`, `decoder` |
| `graphcast_operational` | `g2m`, `m2m`, `m2g` |
| `sfno` | `encoder`, `processor`, `decoder`, `residual` |

The slurm submitters in `scripts/` hard-code their per-experiment defaults
and pass them through `ai-ens infer ...`.

## Zarr format

Inference and verification both speak the SwissClim Evaluations schema:

- dims: `(init_time, lead_time, ensemble, latitude, longitude[, level])`
- `init_time` is `datetime64[ns]`, `lead_time` is `timedelta64[ns]`
- `level` is integer hPa (only on 3D variables)
- variables use ECMWF long names (`10m_u_component_of_wind`, `2m_temperature`,
  `temperature`, `u_component_of_wind`, `geopotential`, ...)

The bridge from earth2studio output to SwissClim is in
[ai_models_ensembles/swissclim_format.py](ai_models_ensembles/swissclim_format.py)
(`e2s_to_swissclim`, exercised by
[tests/test_swissclim_format.py](tests/test_swissclim_format.py)).

## Outputs

```text
$STORE/
  ├─ baselines/<model_id>/<YYYYMMDD_HHMM>/forecast.zarr      # probabilistic baselines
  └─ ablation/
     └─ <phase>/<model_id>/
        ├─ <init_tag>/<run_tag>/forecast.zarr                # per-run forecast
        ├─ eval/<run_tag>/                                   # SwissClim eval modules
        │  ├─ maps/  wd_kde/  energy_spectra/  multivariate/
        │  └─ deterministic/  probabilistic/  ets/  fss/  ssim/
        └─ intercomparison/                                  # cross-run plots
```

## Containers (GH200)

One container per model lives under [containers/](containers/), each pinned
to the deps that model actually needs:

```bash
bash containers/submit_build.sh <model|all> [partition]
# writes $STORE/<model>.sqsh for: aurora, graphcast, sfno, fcn3, atlas, aifsens
```

Base images are NGC PyTorch 25.12 (or 26.01 for `atlas`). Inside the
container, run with the bind-mounted source via
`python -m ai_models_ensembles.cli ...` (the baked-in `ai-ens` entrypoint
uses the image's installed package).

## Architecture

- **`ai_models_ensembles/cli.py`** - Typer CLI (`infer`, `verify`,
  `intercompare`, `models`)
- **`ai_models_ensembles/e2s_models.py`** - registry of earth2studio
  prognostic models with step_hours and probabilistic flag
- **`ai_models_ensembles/e2s_data.py`** - DataSource adapters: factory for
  `ARCO/CDS/GFS/IFS/IFS_ENS/WB2/file:`, plus `XarrayDataSource` for
  in-memory perturbed ICs
- **`ai_models_ensembles/e2s_perturbation.py`** - multiplicative IC noise +
  generic weight-perturbation walker (`.npz / .pt / .pth / .ckpt /
  .safetensors`)
- **`ai_models_ensembles/e2s_inference.py`** - inference driver that handles
  all four perturbation combinations and writes SwissClim-format zarr
- **`ai_models_ensembles/swissclim_format.py`** - schema bridge
  (`e2s_to_swissclim`, `to_swissclim_forecast`, `to_swissclim_target`)
- **`config/`** - SwissClim YAML template + intercomparison config
- **`scripts/`** - Slurm submitters for inference and SwissClim eval
- **`containers/`** - per-model Dockerfiles and `submit_build.sh`
- **`tools/`** - host venv helpers, weight inspection, zarr resharding
- **`figures/`** - perturbation schematics (regenerable SVG/PDF/PNG)

## Pre-commit & Ruff

```bash
pre-commit install
pre-commit run --all-files
```

Hooks: Ruff lint + format, end-of-file fixer, trailing-whitespace cleanup,
merge-conflict checks. `ruff` and `pre-commit` are part of the minimal host
venv install above.

## Testing

```bash
pytest -q
```

The repo's unit tests cover only the schema bridge and CLI loadability;
inference + perturbation are exercised end-to-end on a GPU node.

## Troubleshooting

- **`import earth2studio` fails on the host**: expected. `earth2studio` and
  the model deps only live inside containers; the host venv is intentionally
  minimal. Run inside `$STORE/<model>.sqsh` via `srun --container-image=...`.
- **CDS data source fails**: needs `~/.cdsapirc`. Prefer `arco` for ERA5.
- **`envsubst: command not found`**: install `gettext-base`.
- **Container build fails**: must be submitted to a compute node via
  `containers/submit_build.sh`; the login nodes can't build.
- **Numpy 2 incompatibility**: pin in `pyproject.toml`. earth2studio and
  swissclim both require `numpy >= 2`.
