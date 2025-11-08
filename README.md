# AI Model Ensembles for Weather Forecasting

Run GraphCast, FourCastNetV2, and GenCast ensembles, convert outputs to Zarr, and verify against IFS/ERA5 with ready-made plots and animations.

## Quickstart (uv + venv)

1. Create a Python 3.10 virtual env with uv and install deps

```bash
bash scripts/setup_uv.sh
source .venv/bin/activate
```

2. Validate the environment and GPU availability

```bash
python scripts/check_gpu.py
bash ./tests/validate.sh
```

3. **[Recommended]** Test the installation

```bash
# Quick functionality test
python tests/test_basic_functionality.py

# Minimal workflow test
./tests/run_minimal_test.sh

# Check workflow status anytime
./tests/check_workflow_status.sh
```

See [tests/README.md](tests/README.md) for detailed testing documentation and [tests/QUICKSTART_TEST.md](tests/QUICKSTART_TEST.md) for a step-by-step example workflow.

4. Submit jobs (Slurm)
   - Download ERA5 + IFS: `submit_download_data.sh`
   - ML inference (array-ready): `submit_ml_inference.sh`
   - Convert to Zarr: `submit_convert_zarr.sh`
   - Verify + plots + artefact bundles: `submit_verification.sh`

Logs are written to `logs/`. Adjust `config.sh` to tailor runs.

## Configure `config.sh`

- Set `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME`, `NUM_MEMBERS`, perturbation values, and region.
- Optional: set `EARTHKIT_CACHE_DIR` to a writable path for Earthkit cache.

```bash
bash ./tests/validate.sh
```

This checks your Python env and packages, `ai-models`, GRIB tools (`eccodes/cfgrib`), ImageMagick, ECMWF credentials (`~/.cdsapirc`), `OUTPUT_DIR` writability, and more. Fix any warnings/errors before submitting jobs.

Notes:

- On Linux aarch64 with NVIDIA GPUs, `scripts/setup_uv.sh` installs JAX GPU wheels via `pip install "jax[cuda12]"` (no source build) and PyTorch from NVIDIA's aarch64 index.
- On x86_64 with NVIDIA GPUs, PyTorch is installed targeting CUDA 12.4 wheels; JAX defaults to CPU unless you add `JAX_OVERRIDE=cuda` and adapt.
- GenCast is inherently probabilistic; weight/latent perturbations are skipped and ensembles are generated via the model itself.

### Centralized Slurm settings

All `#SBATCH` settings are defined in `config.sh` so you don't need to edit the submit scripts. Defaults match the current headers.

- Download (`submit_download_data.sh`): `DL_JOB_NAME`, `DL_NODES`, `DL_NTASKS`, `DL_CPUS_PER_TASK`, `DL_MEM_PER_CPU`, `DL_PARTITION`, `DL_ACCOUNT`, `DL_TIME`
- Inference (`submit_ml_inference.sh`): `INF_JOB_NAME`, `INF_NODES_SB`, `INF_NTASKS_SB`, `INF_CPUS_PER_TASK_SB`, `INF_MEM_PER_CPU_SB`, `INF_PARTITION_SB`, `INF_GRES_SB`, `INF_ACCOUNT_SB`, `INF_TIME_SB`
- Convert (`submit_convert_zarr.sh`): `ZARR_JOB_NAME`, `ZARR_NODES_SB`, `ZARR_NTASKS_SB`, `ZARR_CPUS_PER_TASK_SB`, `ZARR_MEM_PER_CPU_SB`, `ZARR_PARTITION_SB`, `ZARR_ACCOUNT_SB`, `ZARR_TIME_SB`
- Verify (`submit_verification.sh`): `VERIF_JOB_NAME`, `VERIF_NODES_SB`, `VERIF_NTASKS_SB`, `VERIF_PARTITION_SB`, `VERIF_ACCOUNT_SB`, `VERIF_TIME_SB`, `VERIF_EXCLUSIVE`

Example override (edit `config.sh`):

```bash
export INF_PARTITION_SB=normal
export INF_GRES_SB=gpu:2
export INF_TIME_SB=12:00:00
```

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

With these artefacts on disk you can generate bespoke visualisations or overlay multiple models using the `ai-ens intercompare` command without rerunning verification.

### Documentation

- **[tests/QUICKSTART_TEST.md](tests/QUICKSTART_TEST.md)**: Step-by-step example workflow

## Troubleshooting

- **GRIB readers**: Install OS packages for ecCodes and `pip install cfgrib` if missing.
- **ECMWF credentials**: Ensure `~/.cdsapirc` (and MARS) credentials are valid for ERA5/IFS access.
- **GPU requirements**: Ensure NVIDIA drivers meet JAX/PyTorch requirements (driver >= 525 for CUDA 12). On aarch64, PyTorch GPU is installed from NVIDIA's pip wheels; JAX GPU comes from `jax[cuda12]` wheels (no local toolkit required).
- **CuDNN warnings**: If you see "Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.8.0" when importing both Torch and JAX in the same process: Import JAX before Torch (JAX preloads pip's cuDNN 9.14, which Torch can reuse).