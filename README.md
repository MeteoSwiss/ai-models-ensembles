# AI Model Ensembles for Weather Forecasting

Turns deterministic AI weather models into calibrated ensembles by
perturbing their frozen weights post-hoc, and benchmarks them against
trained-probabilistic models and IFS-ENS with
[SwissClim Evaluations](https://github.com/swiss-ai/SwissClim_Evaluations).
The method, experiments, and results are in the paper:
**[paper link TBD](https://example.org/ai-models-ensembles-paper)**.

The repo is a thin orchestration layer:
[earth2studio](https://github.com/NVIDIA/earth2studio) does model loading,
IC fetch, and rollout; swissclim-evaluations does verification; this code
wires them together (model registry, IC + weight perturbation, Slurm
scripts, per-model GH200 containers).

## Layout

| Path | What |
|---|---|
| `ai_models_ensembles/` | the package: Typer CLI, perturbation, the earth2studio to SwissClim bridge |
| `scripts/` | Slurm submitters for inference and verification - [scripts/README.md](scripts/README.md) |
| `tools/` | one analysis script per paper figure/table, plus the Milton case study - [tools/README.md](tools/README.md) |
| `figures/` | schematic sources (`draw_*.py`) and the generated figure/table files (all PDF) |
| `containers/` | one Dockerfile per model + `submit_build.sh` |
| `config/` | SwissClim YAML templates |
| `tests/` | schema-bridge and CLI-loadability unit tests |

`ai-ens models` prints the model registry; `ai-ens --help` the CLI. Adding
a model is one `ModelSpec` in `ai_models_ensembles/e2s_models.py`.

## Quickstart

```bash
# 1. per-model container(s) -> $STORE/<model>.sqsh
bash containers/submit_build.sh <model|all>

# 2. host venv (orchestration + analysis only; no torch / earth2studio / jax)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . --no-deps
uv pip install -e ./SwissClim_Evaluations --no-deps      # ai-ens verify, plot_bivariate.py
uv pip install "typer>=0.12" "zarr>=3.0" "xarray>=2024.10" "numpy>=2.0" \
  pandas matplotlib dask numcodecs pyyaml netCDF4 cfgrib rich pytest ruff pre-commit

# 3. run (each job submits to Slurm; parameters are constants at the top of each script)
bash scripts/submit_all_inference.sh [baseline]
bash scripts/submit_ablation.sh {phase1|phase2|phase2b|phase3|phase3b} [model]
bash scripts/evaluate_baselines.sh all [model]
bash scripts/evaluate_ablation.sh {<phase>|intercompare <phase>|allphases_intercompare} [model]
```

Anything that imports `earth2studio` runs inside a container:
`srun --container-image=$STORE/<model>.sqsh python -m ai_models_ensembles.cli ...`.
`ai-ens infer` flags (`--data-source`, `--ic-magnitude`, `--ic-zarr`,
`--weight-magnitude`, `--layer`) are documented in `ai-ens infer --help`;
the `--layer` groups per model live in `_MODEL_LAYER_GROUPS`
([e2s_perturbation.py](ai_models_ensembles/e2s_perturbation.py)).

## Reproducing the figures and tables

`tools/*.py` regenerate one paper figure or table each. They read the
cached intermediates committed under [tools/data/](tools/data/) plus the
eval outputs under `$STORE`, and resolve every machine path through
[tools/_env.py](tools/_env.py) / `${AIENS_*:-default}` shell variables, so
they run on any machine once these point at your data:

| Variable | Meaning |
|---|---|
| `AIENS_STORE` | forecasts / baselines / intercomparison root |
| `AIENS_SCRATCH` | scratch for large CSV/JSON intermediates |
| `AIENS_WB2_22` / `AIENS_WB2_24` | WeatherBench2 ERA5 truth zarrs |
| `AIENS_IFS_ENS` | IFS-ENS reference zarr |
| `AIENS_PY` / `AIENS_REPO` | interpreter / repo root for the Slurm scripts |

Defaults are the CSCS paths (see `tools/_env.py`). The table scripts that
read only `tools/data/` need no `$STORE`. Full script-to-output map in
[tools/README.md](tools/README.md).

## Output format

Inference and verification speak the SwissClim schema: dims
`(init_time, lead_time, ensemble, latitude, longitude[, level])`,
`datetime64` / `timedelta64` coordinates, integer-hPa `level`, ECMWF long
variable names. Bridge and tests:
[swissclim_format.py](ai_models_ensembles/swissclim_format.py),
[tests/test_swissclim_format.py](tests/test_swissclim_format.py). The
on-disk `$STORE` tree is documented in [scripts/README.md](scripts/README.md).

## Development

```bash
pre-commit install && pre-commit run --all-files
pytest -q          # schema bridge + CLI loadability only
```

Requirements: Linux, Slurm + pyxis/enroot for the GH200 containers, CUDA
12.x, outbound HTTPS for the `arco` / `wb2` data sources, `~/.cdsapirc` for
`cds`, `envsubst` (gettext-base). HPC operational gotchas (SIGSEGV retries,
poisoned fsspec cache, degraded capstor mount) are in
[scripts/README.md](scripts/README.md).
