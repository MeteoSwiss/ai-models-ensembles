# Workflow Execution Scripts

This directory contains the main workflow execution scripts for the `ai-models-ensembles` repository.

## Configuration

### config.sh

Central configuration file for all workflow scripts.

**Key variables:**

- `OUTPUT_DIR`: Base output directory for all data and results
- `DATE_TIME`: Forecast initialization time (YYYYMMDDHHMM format)
- `MODEL_NAME`: AI model to use (graphcast, fourcastnetv2-small, gencast)
- `NUM_MEMBERS`: Number of ensemble members (max 50)
- `PERTURBATION_INIT`: Initial condition perturbation magnitude
- `PERTURBATION_LATENT`: Model weights/latent perturbation magnitude
- `PERTURBATION_LATENTS`: Space-separated list of latent perturbations to run
- `LAYER`: Neural network layer to perturb
- `CROP_REGION`: Region for verification (europe or global)
- `LEAD_TIME`: Forecast lead time in hours
- `INTERVAL`: Time interval between analyses/forecasts (hours)
- `NUM_DAYS`: Number of forecast days

**Slurm settings:**
All Slurm job parameters are configurable in `config.sh`:

- Download job: `DL_*` variables
- Inference job: `INF_*` variables
- Convert job: `ZARR_*` variables
- Verification job: `VERIF_*` variables

**Usage:**

```bash
# Edit config.sh to set your parameters
vim scripts/config.sh

# Source it in interactive sessions
source scripts/config.sh
```

## Workflow Scripts

### submit_download_data.sh

Download initial conditions and reference data.

**What it does:**

1. Downloads ERA5 reanalysis for model initialization
2. Downloads IFS ensemble forecasts (51 members) for comparison
3. Downloads IFS control forecast for comparison
4. Generates model fields file

**Usage:**

```bash
# Edit config.sh first, then:
sbatch scripts/submit_download_data.sh

# Or with DRY_RUN to see commands without executing:
DRY_RUN=1 sbatch scripts/submit_download_data.sh
```

**Requirements:**

- ECMWF MARS API credentials in `~/.ecmwfapirc`
- Sufficient disk space (~350 GB for full download)

**Outputs:**

- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_field.grib`
- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/fields.txt`
- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/ground_truth.zarr/`
- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/ifs_ens.zarr/`
- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/ifs_control.zarr/`

### submit_ml_inference.sh

Run ML model inference to generate ensemble forecasts.

**What it does:**

1. Downloads model assets (weights, normalizing stats)
2. Generates perturbed initial conditions (if `PERTURBATION_INIT > 0`)
3. Generates perturbed model weights (if `PERTURBATION_LATENT > 0`)
4. Runs ensemble members in parallel
5. Merges results into Zarr format

**Usage:**

```bash
# Standard mode (runs all members)
sbatch scripts/submit_ml_inference.sh

# Array mode (distribute members across multiple jobs)
sbatch --array=0-49 scripts/submit_ml_inference.sh

# For different perturbation values, edit PERTURBATION_LATENTS in config.sh
# It will spawn parallel jobs for each value
```

**Requirements:**

- GPU nodes for model inference
- Initial conditions from download step
- ~7 GB storage per ensemble member

**Outputs:**

- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_0.0_latent_X.X_layer_Y/forecast.zarr/`
- Individual member directories with model runs

### submit_convert_zarr.sh

Convert GRIB outputs to Zarr format for analysis.

**What it does:**

1. Finds all GRIB files in the model directory
2. Converts to Zarr with proper chunking and compression
3. Organizes by member/time dimensions

**Usage:**

```bash
sbatch scripts/submit_convert_zarr.sh
```

**Requirements:**

- Completed inference runs
- GRIB readers (eccodes, cfgrib)

**Outputs:**

- Consolidated Zarr datasets ready for verification

### submit_verification.sh

Generate verification plots, animations, and data artifacts.

**What it does:**

1. Loads forecast and reference data
2. Computes verification metrics (RMSE, spread-skill, rank histograms, energy spectra)
3. Generates plots (2D maps, timeseries, distributions, scorecards)
4. Creates animations (2D field evolution, 3D difference volumes)
5. Saves data artifacts for later reuse
6. Compares against IFS forecasts

**Usage:**

```bash
sbatch scripts/submit_verification.sh
```

**Requirements:**

- Completed convert step
- ImageMagick for GIF generation
- Sufficient memory for data loading

**Outputs:**

- `png_*/`: Verification plots
- `artifacts_*/`: Data bundles (NetCDF, NPZ, CSV)
- `*/animations/`: Animated GIFs
- Error maps and scorecards

## Typical Workflow Sequence

```bash
# 1. Configure
vim scripts/config.sh

# 2. Validate setup
bash ./tools/validate.sh

# 3. Download data
sbatch scripts/submit_download_data.sh
# Wait for completion, check logs/out_dl_*.log

# 4. Run inference
sbatch scripts/submit_ml_inference.sh
# Or for array mode: sbatch --array=0-49 scripts/submit_ml_inference.sh
# Wait for completion, check logs/out_ml_*.log

# 5. Convert to Zarr
sbatch scripts/submit_convert_zarr.sh
# Wait for completion, check logs/out_zarr_*.log

# 6. Generate verification
sbatch scripts/submit_verification.sh
# Wait for completion, check logs/out_verif_*.log

# 7. Check status anytime
./tools/check_workflow_status.sh
```

## Monitoring

**Check job status:**

```bash
squeue -u $USER
```

**Check logs:**

```bash
# List recent logs
ls -lt logs/

# Follow latest download log
tail -f logs/out_dl_*.log

# Check for errors
grep -i error logs/err_*.log
```

**Check disk usage:**

```bash
du -sh $OUTPUT_DIR/$DATE_TIME
```

## Advanced Usage

### Multiple Perturbation Levels

Edit `config.sh`:

```bash
export PERTURBATION_LATENTS="0.0 0.01 0.05 0.1"
```

The inference script will run all levels in parallel.

### Different Models

```bash
# GraphCast (high-resolution)
export MODEL_NAME=graphcast

# FourCastNet V2 (faster, lower resolution)
export MODEL_NAME=fourcastnetv2-small

# GenCast (inherently probabilistic)
export MODEL_NAME=gencast
```

### Custom Regions

```bash
# Europe (default)
export CROP_REGION=europe

# Global
export CROP_REGION=global
```

### Dry Run Mode

Test scripts without executing heavy operations:

```bash
DRY_RUN=1 sbatch scripts/submit_download_data.sh
```

## Troubleshooting

- **Jobs fail to start**: Check Slurm parameters with `sinfo` and verify account access
- **Downloads fail**: Verify ECMWF credentials in `~/.ecmwfapirc`
- **Inference fails**: Check GPU availability with `python tools/check_gpu.py`
- **Out of memory**: Reduce `NUM_MEMBERS` or increase `INF_MEM_PER_CPU_SB` in config.sh
- **Disk space**: Monitor with `du -sh $OUTPUT_DIR/$DATE_TIME`

For setup and validation issues, see [tools/README.md](../tools/README.md).

## See Also

- [Main README](../README.md) - Full repository documentation
- [tools/](../tools/) - Development and testing utilities
- [ai_models_ensembles/](../ai_models_ensembles/) - Core Python package
