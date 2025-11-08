# Quick Start Test Guide

This guide will walk you through testing the `ai-models-ensembles` repository step by step.

**Note:** This guide is located in the `tests/` directory. All paths assume you're running commands from the repository root.

## Prerequisites Check

1. **Activate the virtual environment:**

   ```bash
   cd /users/sadamov/pyprojects/ai-models-ensembles
   source .venv/bin/activate
   ```

2. **Verify the environment:**

   ```bash
   bash ./tests/validate.sh
   ```

   This should complete with "Validation completed" message. Minor warnings are OK.

3. **Check the CLI is working:**

   ```bash
   python -m ai_models_ensembles.cli --help
   ```

   You should see the list of available commands.

4. **Run the test scripts:**

   ```bash
   # Comprehensive functionality test
   python tests/test_basic_functionality.py
   
   # Quick minimal test
   ./tests/run_minimal_test.sh
   
   # Check workflow status
   ./tests/check_workflow_status.sh
   ```

## Configuration

The main configuration is in `config.sh`. Key settings:

```bash
# Current settings (from config.sh):
export DATE_TIME=201801010000      # Storm Burglind: Jan 1, 2018
export MODEL_NAME=graphcast         # Options: graphcast, fourcastnetv2-small, gencast
export NUM_MEMBERS=50               # Number of ensemble members
export PERTURBATION_INIT=0.0        # Initial condition perturbation
export PERTURBATION_LATENT=0.01     # Model weights/latent perturbation
export CROP_REGION=europe           # Region: europe or global
export OUTPUT_DIR=$STORE/sadamov/ai-models-ensembles
```

## Step-by-Step Workflow

### Step 1: Download Initial Conditions (ERA5 Reanalysis)

This downloads the ERA5 reanalysis data for initializing the model:

```bash
source .venv/bin/activate
source config.sh

# Download reanalysis data
ai-ens download-reanalysis \
  --out-dir "$OUTPUT_DIR" \
  --start "$DATE_TIME" \
  --end "$END_DATE_TIME" \
  --interval "$INTERVAL" \
  --model "$MODEL_NAME"
```

**Expected output:** GRIB files in `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_field.grib`

**Note:** This requires ECMWF API credentials configured in `~/.ecmwfapirc`

### Step 2: Download IFS Reference Data (Optional)

Download IFS ensemble and control forecasts for comparison:

```bash
# Download IFS ensemble
ai-ens download-ifs-ensemble \
  --out-dir "$OUTPUT_DIR" \
  --date-time "$DATE_TIME" \
  --interval "$INTERVAL" \
  --num-days "$NUM_DAYS" \
  --model "$MODEL_NAME"

# Download IFS control
ai-ens download-ifs-control \
  --out-dir "$OUTPUT_DIR" \
  --date-time "$DATE_TIME" \
  --interval "$INTERVAL" \
  --num-days "$NUM_DAYS" \
  --model "$MODEL_NAME"
```

**Expected output:** IFS GRIB files in the model directory

### Step 3: Generate Field List

Generate the list of fields needed by the model:

```bash
source .venv/bin/activate

# Generate fields file
ai-models --fields graphcast > $OUTPUT_DIR/$DATE_TIME/graphcast/fields.txt
```

**Expected output:** A text file listing all required fields

### Step 4: Run Model Inference (Single Member Test)

Test running a single ensemble member:

```bash
source .venv/bin/activate
source config.sh

# Run single member (e.g., member 0)
ai-ens infer --member 0
```

**What this does:**

- Creates perturbation directory
- Perturbs initial conditions (if PERTURBATION_INIT > 0)
- Perturbs model weights (if PERTURBATION_LATENT > 0)
- Runs the AI model
- Saves forecast to Zarr format

**Expected output:**

- `$PERTURBATION_DIR/0/init_field.grib`
- Forecast files in the member directory

**Note:** This requires GPU access for GraphCast

### Step 5: Run Full Ensemble (Job Array Mode)

For running all ensemble members, you can:

```bash
# Run all members sequentially (for testing)
ai-ens infer

# Or submit as Slurm job array (recommended for production)
sbatch submit_ml_inference.sh
```

### Step 6: Convert GRIB to Zarr

Convert the forecast GRIB files to Zarr format for analysis:

```bash
source .venv/bin/activate
source config.sh

# Convert forecasts
ai-ens convert --path "$PERTURBATION_DIR" --subdir-search
```

**Expected output:** `forecast.zarr/` directory with ensemble data

### Step 7: Verification and Plotting

Run verification metrics and create plots:

```bash
source .venv/bin/activate
source config.sh

ai-ens verify
```

**What this does:**

- Loads forecast, ground truth, and IFS data
- Calculates verification metrics (RMSE, spread-skill, energy spectra)
- Creates density plots, timeseries, scorecards
- Generates 2D map animations
- Generates 3D difference volumes
- Saves artifacts (NetCDF bundles) for later comparison

**Expected outputs:**

- PNG plots in `$REGION_DIR/png_graphcast/`
- Animations in `$REGION_DIR/0/animations/`
- Artifact bundles in `$REGION_DIR/artifacts_graphcast/`

### Step 8: Compare Multiple Models (Optional)

If you have run multiple models, compare them:

```bash
ai-ens intercompare \
  $REGION_DIR/artifacts_graphcast \
  $REGION_DIR/artifacts_fourcastnet \
  --label GraphCast \
  --label FourCastNet \
  --out-dir $REGION_DIR/comparisons
```

## Minimal Test Without Downloads

If you want to test the CLI without downloading data, you can:

1. **Test CLI help commands:**

   ```bash
   source .venv/bin/activate
   ai-ens --help
   ai-ens download-reanalysis --help
   ai-ens infer --help
   ai-ens verify --help
   ```

2. **Check Python imports:**

   ```bash
   python -c "import ai_models_ensembles.cli; print('CLI module OK')"
   python -c "import ai_models_ensembles.preprocess_data; print('Preprocessing module OK')"
   ```

3. **Verify ai-models installation:**

   ```bash
   ai-models --models
   ```

## Troubleshooting

### Issue: JAX/Haiku import errors

**Solution:** The models require specific JAX versions. Check compatibility:

```bash
python -c "import jax; print(jax.__version__)"
```

If errors persist, you may need to reinstall JAX with GPU support or update dependencies.

### Issue: ECMWF credentials not found

**Error:** `~/.ecmwfapirc` not found

**Solution:**

1. Sign up at <https://apps.ecmwf.int/registration/>
2. Create `~/.ecmwfapirc` with your API key

### Issue: GPU not available

**Error:** Model requires GPU but none found

**Solution:**

- Run on a node with GPU: `srun -p gpu --gres=gpu:1 --pty bash`
- Or use `--deterministic` flag (if supported)

### Issue: Out of memory

**Solution:** Reduce `NUM_MEMBERS` or increase memory allocation in `config.sh`

## Using Slurm Scripts

For production runs on a cluster, use the provided Slurm scripts:

```bash
# 1. Download data
sbatch submit_download_data.sh

# 2. Run inference
sbatch submit_ml_inference.sh

# 3. Convert to Zarr
sbatch submit_convert_zarr.sh

# 4. Verification
sbatch submit_verification.sh
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/*.out
```

## Expected Directory Structure After Run

```
$OUTPUT_DIR/
└── 201801010000/
    └── graphcast/
        ├── fields.txt
        ├── init_field.grib
        ├── ground_truth.zarr/
        ├── ifs_ens.zarr/
        ├── ifs_control.zarr/
        └── init_0.0_latent_0.01_layer_13/
            ├── forecast.zarr/
            ├── europe/
            │   ├── png_graphcast/
            │   ├── png_ifs/
            │   ├── artifacts_graphcast/
            │   └── 0/
            │       └── animations/
            └── {0..49}/
                ├── init_field.grib
                ├── params/
                └── [other member files]
```

## Next Steps

- Adjust configuration in `config.sh` for different dates, models, or perturbations
- Explore the plots and animations generated
- Use `ai-ens intercompare` to compare different model configurations
- Modify plotting scripts in `ai_models_ensembles/plot_*.py` for custom visualizations
