# Quick Start Test Guide

Step-by-step walkthrough of the `ai-models-ensembles` repository. All paths
assume you're running from the repository root.

## Prerequisites Check

1. **Activate the virtual environment:**

   ```bash
   cd /users/sadamov/pyprojects/ai-models-ensembles
   source .venv/bin/activate
   ```

2. **Verify the environment:**

   ```bash
   bash ./tools/validate.sh
   ```

3. **Check the CLI is working:**

   ```bash
   ai-ens --help
   ai-ens models
   ```

   You should see the five registered models: `graphcast_operational`, `sfno`,
   `aurora` (deterministic, weight-perturbed) and `fcn3`, `atlas`
   (probabilistic, re-seeded per member).

4. **Run sanity tests:**

   ```bash
   python tools/test_basic_functionality.py
   ./tools/check_workflow_status.sh
   ```

## Configuration

Main config is in `scripts/config.sh`. Key settings:

```bash
export DATE_TIME=201801010000              # YYYYMMDDHHMM
export MODEL_NAME=graphcast_operational    # see `ai-ens models` for all five
export NUM_MEMBERS=10                      # 50-member IFS ENS is subsampled to 10 to match
export LEAD_TIME=336                       # hours (14 days)
export PERTURBATION_INIT=0.0               # IC noise sigma
export PERTURBATION_LATENT=0.01            # weight noise sigma (deterministic models)
export LAYER=                              # int weight-tensor index, or empty
export IFS_ENS_PATH=/path/to/ifs_ens.zarr           # physical baseline (50 members, full lead time)
export TARGET_PATH=/path/to/era5.zarr               # SwissClim verification target
export DATA_SOURCE=arco                             # ARCO ERA5 (default)
export CROP_REGION=europe
export OUTPUT_DIR=$STORE/sadamov/ai-models-ensembles
```

Verification config lives in [config/swissclim_eval.yaml.template](../config/swissclim_eval.yaml.template).
`submit_verification.sh` renders it with `envsubst` into
`$REGION_DIR/swissclim_eval.yaml` and passes it to `swissclim-evaluations`
through `ai-ens verify`.

## Step-by-Step Workflow

### Step 1: Inference (single deterministic forecast)

```bash
source scripts/config.sh
ai-ens infer --members 1
```

This loads `graphcast_operational` from earth2studio's default `Package`
(downloaded on first use), pulls the IC for `$DATE_TIME` from ARCO, runs
the rollout for `$LEAD_TIME` hours, and writes
`$PERTURBATION_DIR/forecast.zarr` in SwissClim format.

### Step 2: Inference (perturbed ensemble)

For 10 members with weight perturbation:

```bash
ai-ens infer --members 10 --weight-magnitude 0.01 --layer 13
```

For combined IC + weight perturbation:

```bash
ai-ens infer --members 10 --ic-magnitude 0.005 --weight-magnitude 0.01 --layer 13
```

### Step 3: Slurm orchestration (sweep over weight magnitudes)

Edit `PERTURBATION_LATENTS` in `scripts/config.sh`, then:

```bash
sbatch scripts/submit_ml_inference.sh
```

One srun per perturbation magnitude; each srun runs the full member loop.

### Step 4: Verification (SwissClim Evaluations)

```bash
sbatch scripts/submit_verification.sh
```

Two things happen in parallel:

1. For each value of `$PERTURBATION_LATENTS`, renders
   [config/swissclim_eval.yaml.template](../config/swissclim_eval.yaml.template)
   into `$REGION_DIR/swissclim_eval.yaml` and verifies the AI-model forecast.
2. Renders [config/swissclim_ifs_ens.yaml.template](../config/swissclim_ifs_ens.yaml.template)
   targeting `$IFS_ENS_PATH`; output goes to
   `$OUTPUT_DIR/$DATE_TIME/_ifs_ens/$CROP_REGION/swissclim_ifs_ens/`.

Skip the IFS baseline if you only want the AI side:

```bash
VERIFY_IFS_ENS=0 sbatch scripts/submit_verification.sh
```

### Step 5: Compare multiple runs

```bash
# Recursively discovers every swissclim_* dir under $OUTPUT_DIR/$DATE_TIME -
# AI models (one per perturbation latent) plus the IFS ENS baseline.
ai-ens intercompare

# Or pass paths/globs explicitly
ai-ens intercompare \
   "$OUTPUT_DIR/$DATE_TIME/graphcast_operational/init_*/$CROP_REGION/swissclim_*" \
   "$OUTPUT_DIR/$DATE_TIME/sfno/init_*/$CROP_REGION/swissclim_*" \
   "$OUTPUT_DIR/$DATE_TIME/_ifs_ens/$CROP_REGION/swissclim_ifs_ens" \
   --out-dir $OUTPUT_DIR/comparisons/full
```

The wrapper renders an intercomparison YAML on the fly and shells out to
`swissclim-evaluations-compare`.

## Containers (GH200)

```bash
./containers/build.sh
export CONTAINER_IMAGE=$PWD/ai-ens.sqsh
sbatch scripts/submit_ml_inference.sh
```

The Slurm scripts auto-detect `CONTAINER_IMAGE` and pass
`--container-image=$CONTAINER_IMAGE` to `srun`.

## Minimal Test Without Inference

```bash
ai-ens --help
ai-ens models
ai-ens verify --help

python -c "import ai_models_ensembles.cli; print('CLI OK')"
python -c "import earth2studio; print('earth2studio OK', earth2studio.__version__)"
python -c "import swissclim_evaluations; print('SwissClim OK')"
```

## Troubleshooting

- **`ai-ens models` empty / import errors**: earth2studio's per-model extras
  are large; ensure `uv pip install -e .` completed without resolution errors.
- **CDS data source fails**: needs `~/.cdsapirc`. Prefer `arco` for ERA5.
- **`envsubst: command not found`**: install `gettext` (`apt install gettext-base`).
- **Out of memory**: reduce `NUM_MEMBERS` or bump `INF_MEM_PER_CPU_SB` in `config.sh`.
- **Container build fails**: needs `podman` and `enroot`; see
  [../containers/build.sh](../containers/build.sh).

## Slurm shortcuts

```bash
sbatch scripts/submit_ml_inference.sh
sbatch scripts/submit_verification.sh

squeue -u $USER
tail -f logs/*.out
```

## Expected directory structure

```
$OUTPUT_DIR/
└── 201801010000/
    ├── graphcast_operational/
    │   └── init_0.0_latent_0.01_layer_13/
    │       ├── forecast.zarr/
    │       ├── _e2s_work/                # per-member perturbed checkpoints
    │       └── europe/
    │           ├── swissclim_eval.yaml
    │           └── swissclim_graphcast_operational/
    │               ├── maps/  histograms/  deterministic/  ...
    ├── sfno/        ...
    ├── aurora/      ...
    ├── fcn3/        ...
    ├── atlas/       ...
    └── _ifs_ens/europe/swissclim_ifs_ens/         # IFS ENS baseline verify
```

## Next steps

- Tune `config/swissclim_eval.yaml.template` modules / metrics / plotting.
- Sweep `PERTURBATION_LATENTS` to study weight-perturbation sensitivity.
- Compare runs with `ai-ens intercompare`.
