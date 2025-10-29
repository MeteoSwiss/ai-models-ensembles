# AI Model Ensembles for Weather Forecasting

Run GraphCast and FourCastNetV2 ensembles, convert outputs to Zarr, and verify against IFS/ERA5 with ready-made plots and animations.

## Quickstart (uv + venv)

1. Create a Python 3.10 virtual env with uv and install deps

```bash
bash scripts/setup_uv.sh
source .venv/bin/activate
```

1. Validate the environment and GPU availability

```bash
python scripts/check_gpu.py
bash ./validate.sh
```

1. Submit jobs (Slurm)
   - Download ERA5 + IFS: `submit_download_data.sh`
   - ML inference (array-ready): `submit_ml_inference.sh`
   - Convert to Zarr: `submit_convert_zarr.sh`
   - Verify + plots: `submit_verification.sh`

Logs are written to `logs/`. Adjust `config.sh` to tailor runs.

## Configure `config.sh`

- Set `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME`, `NUM_MEMBERS`, perturbation values, and region.
- Optional: set `EARTHKIT_CACHE_DIR` to a writable path for Earthkit cache.

```bash
bash ./validate.sh
```

This checks your Python env and packages, `ai-models`, GRIB tools (`eccodes/cfgrib`), ImageMagick, ECMWF credentials (`~/.cdsapirc`), `OUTPUT_DIR` writability, and more. Fix any warnings/errors before submitting jobs.

Notes:

- On Linux aarch64 with NVIDIA GPUs, `scripts/setup_uv.sh` installs JAX GPU wheels via `pip install "jax[cuda12]"` (no source build) and PyTorch from NVIDIA's aarch64 index.
- On x86_64 with NVIDIA GPUs, PyTorch is installed targeting CUDA 12.4 wheels; JAX defaults to CPU unless you add `JAX_OVERRIDE=cuda` and adapt.

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

# Verification
ai-ens verify
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
           └─ ${MEMBER}/
              ├─ init_field.grib          # link or perturbed
              ├─ weights.tar | params/    # link or perturbed
              └─ animations/, png_*/      # verification artifacts
```

## Troubleshooting

- Install GRIB readers if missing: use OS packages for ecCodes and `pip install cfgrib`.
- Ensure `~/.cdsapirc` (and MARS) credentials are valid for ERA5/IFS access
- For GPU inference, ensure NVIDIA drivers meet JAX/PyTorch requirements (driver >= 525 for CUDA 12). On aarch64, PyTorch GPU is installed from NVIDIA's pip wheels; JAX GPU comes from `jax[cuda12]` wheels (no local toolkit required).
- GraphCast is installed from PyPI (`ai-models-graphcast`); no local source checkout is required.
- If you see a message like "Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.8.0" when importing both Torch and JAX in the same process: Import JAX before Torch (JAX preloads pip’s cuDNN 9.14, which Torch can reuse)

## License

See `LICENSE`.
