# AI Model Ensembles for Weather Forecasting

Run GraphCast, FourCastNetV2, and GenCast ensembles, convert outputs to Zarr, and verify against IFS/ERA5 with ready-made plots and animations.

## Quickstart (uv + venv)

1. Create a Python 3.10 virtual env with uv and install deps

```bash
bash tools/setup_uv.sh
source .venv/bin/activate
```

2. Validate the environment and GPU availability

```bash
python tools/check_gpu.py
bash ./tools/validate.sh
```

1. **[Recommended]** Test the installation
1. Enable pre-commit hooks (format + lint)

```bash
pre-commit install
```

This will enforce Ruff linting and formatting on staged Python files.

```bash
# Quick functionality test
python tools/test_basic_functionality.py

# Minimal workflow test
./tools/run_minimal_test.sh

# Check workflow status anytime
./tools/check_workflow_status.sh
```

See [tools/README.md](tools/README.md) for detailed testing documentation and [tools/QUICKSTART_TEST.md](tools/QUICKSTART_TEST.md) for a step-by-step example workflow.

1. Submit jobs (Slurm)
   - Download ERA5 + IFS: `scripts/submit_download_data.sh`
   - ML inference (array-ready): `scripts/submit_ml_inference.sh`
   - Convert to Zarr: `scripts/submit_convert_zarr.sh`
   - Verify + plots + artefact bundles: `scripts/submit_verification.sh`

Logs are written to `logs/`. Adjust `scripts/config.sh` to tailor runs.

## Configure `scripts/config.sh`

- Set `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME`, `NUM_MEMBERS`, perturbation values, and region.
- Optional: set `EARTHKIT_CACHE_DIR` to a writable path for Earthkit cache.

```bash
bash ./tools/validate.sh
```

This checks your Python env and packages, `ai-models`, GRIB tools (`eccodes/cfgrib`), ImageMagick, ECMWF credentials (`~/.cdsapirc`), `OUTPUT_DIR` writability, and more. Fix any warnings/errors before submitting jobs.

Notes:

- On Linux aarch64 with NVIDIA GPUs, `tools/setup_uv.sh` installs JAX GPU wheels via `pip install "jax[cuda12]"` (no source build) and PyTorch from NVIDIA's aarch64 index.
- On x86_64 with NVIDIA GPUs, PyTorch is installed targeting CUDA 12.4 wheels; JAX defaults to CPU unless you add `JAX_OVERRIDE=cuda` and adapt.
- GenCast is inherently probabilistic; weight/latent perturbations are skipped and ensembles are generated via the model itself.

See [scripts/README.md](scripts/README.md) for detailed workflow documentation and Slurm configuration options.

## Requirements

- Linux. Slurm recommended for the provided submit scripts.
- ECMWF credentials for ERA5/IFS (configure `~/.cdsapirc`; MARS if needed).
- GPU (CUDA 12.x) for model inference; CPU is fine for plotting/conversion. On ARM (linux-aarch64), PyTorch GPU comes from NVIDIA's pip index; JAX GPU is installed from prebuilt wheels with `jax[cuda12]`.
- GRIB readers: `cfgrib` + `ecCodes`. Install ecCodes via your OS package manager (e.g., `apt install eccodes`) and `pip install cfgrib`.

## CLI usage (Typer)

You can run individual steps via the Typer CLI. After activating the environment, either use the module or the `ai-ens` console command:

```bash
# Module form
python -m ai_models_ensembles.cli --help

# Console script form
ai-ens --help
```

Examples:

```bash
# Download fields
ai-ens download-reanalysis --out-dir "$OUTPUT_DIR" --start "$DATE_TIME" --end "$END_DATE_TIME" \
   --interval "$INTERVAL" --model "$MODEL_NAME"
ai-ens download-ifs-ensemble --out-dir "$OUTPUT_DIR" --date-time "$DATE_TIME" \
   --interval "$INTERVAL" --num-days "$NUM_DAYS" --model "$MODEL_NAME"
ai-ens download-ifs-control --out-dir "$OUTPUT_DIR" --date-time "$DATE_TIME" \
   --interval "$INTERVAL" --num-days "$NUM_DAYS" --model "$MODEL_NAME"

# Convert GRIB to Zarr
ai-ens convert --path "$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME"
ai-ens convert --path "$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}" \
   --subdir-search

# Inference
ai-ens infer                 # run full member loop using env from config.sh
ai-ens infer --member 7      # run a single member (array mode)

# Verification & Artefacts
ai-ens verify

# Intercompare saved artefacts from multiple models
ai-ens intercompare /path/to/artifacts_graphcast /path/to/artifacts_fourcastnet \
   --label GraphCast --label FourCastNet --out-dir comparisons
```

### Notes

- The CLI uses Typer with Rich for colorful help and readable tracebacks; the same output goes to the Slurm logs under `LOG_DIR`.
- Show help any time: `ai-ens --help` or `python -m ai_models_ensembles.cli --help`.

Generate fields file:

```bash
ai-models --fields graphcast > $OUTPUT_DIR/$DATE_TIME/graphcast/fields.txt
```

## Outputs

```text
$OUTPUT_DIR/
  └─ $DATE_TIME/
     └─ $MODEL_NAME/
        ├─ fields.txt
        ├─ init_field.grib
        ├─ ground_truth.zarr/
        ├─ ifs_ens.zarr/
        ├─ ifs_control.zarr/
        ├─ forecast.zarr/                 # unperturbed
        └─ init_{INIT}_latent_{LAT}_layer_{LAYER}/
           ├─ forecast.zarr/              # perturbed ensemble merged
           ├─ png_*/                      # verification figures
           ├─ png_ifs/
           ├─ artifacts_{MODEL}/          # NetCDF bundles for plots/animations (per model)
           │  ├─ ensemble/
           │  │  └─ data/{metric}/
           │  └─ member_{MEMBER}/
           │     └─ data/{metric}/
           └─ ${MEMBER}/
             ├─ init_field.grib          # link or perturbed
             ├─ weights.tar | params/ | assets/  # model-specific assets (symlink or perturbed)
              └─ animations/              # GIFs + static PNGs

```

Each plot or animation is now accompanied by a data artefact saved beforehand:

- Static plots store their inputs (NetCDF/NPZ/CSV) under the matching `artifacts_*` tree.
- Animations (2D maps, 3D difference volumes) persist the xarray payloads used for each GIF.
- Density/rank histograms, RMSE, spread-skill, timeseries, and energy spectra export comparable datasets that can be reloaded later.

### Pre-commit & Ruff

A `.pre-commit-config.yaml` file is provided. Install hooks with:

```bash
pre-commit install
pre-commit run --all-files  # optional initial pass
```

Hooks executed:

- Ruff lint (auto-fix enabled)
- Ruff format (code formatting)
- End-of-file fixer, trailing whitespace cleanup, merge-conflict checks
- Custom: forbid stray `print(` outside the CLI module

Dev dependencies are available via the `dev` extra:

```bash
uv pip install -e .[dev]
```

### Testing

Repository-level smoke tests (in `tests/test_repository_smoke.py`) validate that:

1. Core imports and CLI are loadable.
2. Vertical profile plotting (`plot_vertical_profile_metrics` from `plot_1d_timeseries.py`) produces PNG + NPZ artefacts.
3. PIT histogram functions generate expected NPZ payloads.

Run tests:

```bash
pytest -q
```

All smoke tests use synthetic xarray datasets and execute quickly (<5s typical).

File naming follows the SwissClim-style pattern produced by `build_output_filename`:

```text
<metric>_<variable>_<level?>_<qualifier?>_init<YYYYMMDDHH-YYYYMMDDHH?>_lead<000h-XXXh?>_<ensemble-token>.npz
```

Ensemble tokens automatically normalise (e.g. `graphcast` -> `ensgraphcast`, probabilistic -> `ensprob`). When no ensemble is provided the suffix `ensnone` is used.

### Reproducing & Intercomparison

All artefacts live under `artifacts_<model_name>/.../data/<metric>/`. You can safely load NPZ bundles with `numpy.load(path, allow_pickle=True)` (object arrays are used for variable-length histogram edges/densities). The intercomparison CLI (`ai-ens intercompare`) consumes existing metrics (density, energy spectra, RMSE, timeseries, rank histograms) and can be extended to include the new histogram/KDE/PIT outputs if desired.

With these artefacts on disk you can generate bespoke visualisations or overlay multiple models using the `ai-ens intercompare` command without rerunning verification.

## Directory Structure

- **`ai_models_ensembles/`**: Core Python package with CLI and workflow modules
- **`scripts/`**: Workflow execution scripts and configuration ([see scripts/README.md](scripts/README.md))
- **`tools/`**: Development utilities for setup, testing, and monitoring ([see tools/README.md](tools/README.md))

## Troubleshooting

- **GRIB readers**: Install OS packages for ecCodes and `pip install cfgrib` if missing.
- **ECMWF credentials**: Ensure `~/.cdsapirc` (and MARS) credentials are valid for ERA5/IFS access.
- **GPU requirements**: Ensure NVIDIA drivers meet JAX/PyTorch requirements (driver >= 525 for CUDA 12). On aarch64, PyTorch GPU is installed from NVIDIA's pip wheels; JAX GPU comes from `jax[cuda12]` wheels (no local toolkit required).
- **CuDNN warnings**: If you see "Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.8.0" when importing both Torch and JAX in the same process: Import JAX before Torch (JAX preloads pip's cuDNN 9.14, which Torch can reuse).
