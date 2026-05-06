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

`ai-ens models` prints the live registry. Adding a model means appending one
`ModelSpec` in [ai_models_ensembles/e2s_models.py](ai_models_ensembles/e2s_models.py).

## Initial conditions

All five AI models start from **ARCO ERA5** by default. Reasons:

- All five were trained on ERA5; ERA5 ICs are in-distribution.
- The on-disk IFS download ([sadamov/ifs_download](https://github.com/sadamov/ifs_download))
  doesn't include the t+0 analysis or the full variable list any of these
  models need (missing `u100m`, `v100m`, `sp`, `tcwv`, `w`, `sst`,
  accumulated fluxes; missing 600 hPa).
- The IFS ENS forecast on disk plays the role of the physical baseline; it's
  plugged directly into SwissClim verification, not re-initialised.

To use a different IC source, override `DATA_SOURCE` in `scripts/config.sh`:
`arco | cds | gfs | ifs | ifs_ens | wb2 | file:/path | ifs_analysis:/path`.
The `ifs_analysis:` source is still wired up (extracts `lead_time=0` from a
SwissClim-format IFS zarr) for the day you have a download with t+0 analysis
and the full variable set.

## Quickstart (uv + venv)

1. Create a Python 3.11 virtual env and install deps

```bash
bash tools/setup_uv.sh
source .venv/bin/activate
```

This pulls `earth2studio` (with model extras) and `swissclim-evaluations`
from the `research` branch of `swiss-ai/SwissClim_Evaluations`.

2. Validate the environment and GPU availability

```bash
python tools/check_gpu.py
bash ./tools/validate.sh
```

3. List available models

```bash
ai-ens models
```

4. Submit jobs (Slurm)

```bash
# (Optional) build the GH200 container
./containers/build.sh
export CONTAINER_IMAGE=$PWD/ai-ens.sqsh

sbatch scripts/submit_ml_inference.sh   # earth2studio inference
sbatch scripts/submit_verification.sh   # SwissClim verification
ai-ens intercompare                     # compare runs
```

## Configure `scripts/config.sh`

- `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME` (registry name; see `ai-ens models`)
- `NUM_MEMBERS`, `LEAD_TIME` (hours), `CROP_REGION`
- `IFS_ENS_PATH`: absolute path to your on-disk SwissClim-format IFS ENS
  zarr (8 weeks of init times, 50 members, full lead time). Physical-model
  baseline for SwissClim verification.
- `TARGET_PATH`: absolute path to the ERA5 zarr used as SwissClim's
  verification target (e.g. WB2 ERA5).
- `DATA_SOURCE`: `arco | cds | gfs | ifs | ifs_ens | wb2 | file:/path |
  ifs_analysis:/path` (default: `arco`).
- `PERTURBATION_INIT`, `PERTURBATION_LATENT`, `PERTURBATION_LATENTS`, `LAYER`
- `CONTAINER_IMAGE`: `.sqsh` path; empty = run on host venv
- `SWISSCLIM_CONFIG_TEMPLATE` / `SWISSCLIM_CONFIG`

See [scripts/README.md](scripts/README.md) for the full set.

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

```bash
# Single deterministic forecast (host venv)
ai-ens infer --model graphcast_operational --init 2023-01-02T00 \
   --lead-hours 336 --members 1 --data-source arco \
   --output /scratch/$USER/runs/gc_op/forecast.zarr

# 10-member IC-perturbed ensemble
ai-ens infer --model sfno --init 2023-01-02T00 --lead-hours 336 \
   --members 10 --ic-magnitude 0.005 --data-source arco \
   --output $PERTURBATION_DIR/forecast.zarr

# 10-member weight-perturbed ensemble (per-member NGC mirror)
ai-ens infer --model graphcast_operational --init 2023-01-02T00 \
   --lead-hours 336 --members 10 --weight-magnitude 0.01 --layer 13 \
   --data-source arco --output $PERTURBATION_DIR/forecast.zarr

# Verification (delegates to swissclim-evaluations)
ai-ens verify --config $REGION_DIR/swissclim_eval.yaml

# Intercompare every verify output under $OUTPUT_DIR/$DATE_TIME (recursive),
# including AI-model perturbation runs and the IFS ENS baseline.
ai-ens intercompare
ai-ens intercompare path/A/swissclim_graphcast_operational \
                    path/B/swissclim_sfno \
                    path/C/swissclim_ifs_ens \
                    --label GC --label SFNO --label "IFS ENS"
```

When env vars from `config.sh` are present, almost every flag has a sensible
default (e.g. `--model` falls back to `$MODEL_NAME`, `--init` to `$DATE_TIME`,
`--ic-magnitude` to `$PERTURBATION_INIT`, `--weight-magnitude` to
`$PERTURBATION_LATENT`, etc.).

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
$OUTPUT_DIR/
  └─ $DATE_TIME/
     ├─ $MODEL_NAME/                          # one tree per AI model
     │  └─ init_{INIT}_latent_{LAT}_layer_{LAYER}/
     │     ├─ forecast.zarr/                  # SwissClim-format ensemble forecast
     │     ├─ _e2s_work/                      # per-member perturbed checkpoints (transient)
     │     └─ {CROP_REGION}/
     │        ├─ swissclim_eval.yaml          # rendered SwissClim config
     │        └─ swissclim_{MODEL_NAME}/      # SwissClim Evaluations output_root
     │           ├─ maps/  histograms/  wd_kde/
     │           ├─ energy_spectra/  vertical_profiles/
     │           └─ deterministic/  ets/  probabilistic/  ssim/
     ├─ _ifs_ens/{CROP_REGION}/swissclim_ifs_ens/         # IFS ENS baseline
     └─ intercomparison_{N}/                              # `ai-ens intercompare` output
```

## Containers (GH200)

A single Dockerfile + build script live under [containers/](containers/):

```bash
./containers/build.sh                  # podman build → enroot import → ai-ens.sqsh
IMAGE_TAG=ai-ens:dev OUTPUT=ai-ens-dev.sqsh ./containers/build.sh
```

The image is based on `nvcr.io/nvidia/pytorch:25.12-py3`, installs uv, and
editable-installs this repo with all earth2studio model extras enabled. Use
it via `srun --container-image=$PWD/ai-ens.sqsh -- ai-ens infer ...`. The
Slurm scripts pick up the image automatically when `CONTAINER_IMAGE` is set.

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
- **`scripts/`** - Slurm wrappers + `config.sh`
- **`containers/`** - GH200 container build pipeline
- **`tools/`** - environment setup, validation, status checks

## Pre-commit & Ruff

```bash
pre-commit install
pre-commit run --all-files
```

Hooks: Ruff lint + format, end-of-file fixer, trailing-whitespace cleanup,
merge-conflict checks.

```bash
uv pip install -e .[dev]
```

## Testing

```bash
pytest -q
```

The repo's unit tests cover only the schema bridge and CLI loadability;
inference + perturbation are exercised end-to-end on a GPU node.

## Troubleshooting

- **`ai-ens models` empty / import errors**: earth2studio's per-model extras
  are large; ensure `uv pip install -e .` completed without resolution errors.
- **CDS data source fails**: needs `~/.cdsapirc`. Prefer `arco` for ERA5.
- **`envsubst: command not found`**: install `gettext-base`.
- **Container build fails**: `containers/build.sh` requires `podman` and
  `enroot` on the build host.
- **Numpy 2 incompatibility**: report and pin in `pyproject.toml`. earth2studio
  and swissclim both require numpy >= 2.
