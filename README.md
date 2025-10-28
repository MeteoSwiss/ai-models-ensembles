# AI Model Ensembles for Weather Forecasting

Run GraphCast and FourCastNetV2 ensembles, convert outputs to Zarr, and verify against IFS/ERA5 with ready-made plots and animations.

## Quickstart

1. Configure `config.sh`
   - Set `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME`, `NUM_MEMBERS`, perturbation values, and region.
   - Optional: set `EARTHKIT_CACHE_DIR` to a writable path for Earthkit cache.

```bash
bash ./validate.sh
```

This checks your Python env and packages, `ai-models`, GRIB tools (`eccodes/cfgrib`), ImageMagick, ECMWF credentials (`~/.cdsapirc`), `OUTPUT_DIR` writability, and more. Fix any warnings/errors before submitting jobs.

1. Create env and activate

```bash
mamba env create -f environment.yml
conda activate ai_models_ens
```

1. Submit jobs (Slurm)
   - Download ERA5 + IFS: `submit_download_data.sh`
   - ML inference (array-ready): `submit_ml_inference.sh`
   - Convert to Zarr: `submit_convert_zarr.sh`
   - Verify + plots: `submit_verification.sh`

Logs are written to `logs/`. Adjust `config.sh` to tailor runs.

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
- GPU (CUDA 12.X) for model inference; CPU is fine for plotting/conversion.
- GRIB readers: `cfgrib` + `ecCodes` (install via conda-forge if missing).

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

- Install GRIB readers if missing: `conda install -c conda-forge cfgrib eccodes`
- Ensure `~/.cdsapirc` (and MARS) credentials are valid for ERA5/IFS access
- For GPU inference, match CUDA 12.X drivers/toolkit to the environment
- If GraphCast local repo is missing and env install fails, clone `../ai-models-graphcast` or update `environment.yml`

## License

See `LICENSE`.
