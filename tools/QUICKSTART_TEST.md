# Quick Start Test Guide

Step-by-step walkthrough of `ai-models-ensembles`. All paths assume you're
running from the repository root.

## Prerequisites Check

1. **Activate the host venv** (see [main README](../README.md) for the
   one-time setup):

   ```bash
   cd /users/sadamov/pyprojects/ai-models-ensembles
   source .venv/bin/activate
   ```

2. **Check the CLI is loadable on the host:**

   ```bash
   python -m ai_models_ensembles.cli --help
   python -m ai_models_ensembles.cli models
   ```

   You should see the five registered models: `graphcast_operational`,
   `sfno`, `aurora` (deterministic, weight-perturbed) and `fcn3`, `atlas`,
   `aifsens` (probabilistic, re-seeded per member).

3. **Smoke-test inside a container** (only this needs `earth2studio` etc.):

   ```bash
   srun --container-image=$STORE/aurora.sqsh \
     python tools/test_basic_functionality.py
   ```

## Configuration

There is no shared config file. Per-experiment constants (init times,
magnitudes, models, levels, output paths) live at the top of each slurm
submitter under [`../scripts/`](../scripts/). Edit those constants directly
to tweak a run.

For ad-hoc `ai-ens infer ...` calls (outside slurm), every parameter is an
explicit flag - see `ai-ens infer --help`.

## Step-by-Step Workflow

### Step 1: Quick smoke inference

Inside an interactive job on a GPU node, using the bind-mounted source:

```bash
srun --container-image=$STORE/aurora.sqsh \
  python -m ai_models_ensembles.cli infer \
  --model aurora --init 2024-02-15T00:00 \
  --lead-hours 24 --members 1 --data-source arco \
  --output /tmp/$USER/smoke/forecast.zarr
```

### Step 2: Weight-perturbed ensemble

```bash
srun --container-image=$STORE/graphcast.sqsh \
  python -m ai_models_ensembles.cli infer \
  --model graphcast_operational --init 2024-02-15T00:00 \
  --lead-hours 240 --members 10 \
  --weight-magnitude 0.01 --layer all \
  --data-source arco \
  --output $STORE/test/gc_w01/forecast.zarr
```

### Step 3: Full ablation (slurm)

```bash
# Phase 1: magnitude sweep across 3 deterministic models, 4 init times
bash scripts/submit_ablation.sh phase1

# Probabilistic baselines for 3 models, 8 weeks
bash scripts/submit_all_inference.sh
```

### Step 4: SwissClim eval

```bash
bash scripts/evaluate_ablation.sh phase1
bash scripts/evaluate_ablation.sh intercompare phase1
```

The eval submitter generates per-run YAMLs and submits one sbatch each;
`intercompare` reads every per-run output and produces cross-run plots.

### Step 5: Ad-hoc intercompare

```bash
python -m ai_models_ensembles.cli intercompare \
   /path/to/swissclim_aurora \
   /path/to/swissclim_graphcast_operational \
   /path/to/swissclim_sfno \
   --label Aurora --label GC --label SFNO
```

## Containers (GH200)

```bash
bash containers/submit_build.sh <model|all>   # writes $STORE/<model>.sqsh
```

Six per-model images: `aurora`, `graphcast`, `sfno`, `fcn3`, `atlas`,
`aifsens`. The slurm scripts pick `$STORE/<model>.sqsh` based on the model
they're submitting.

## Minimal Test Without Inference

```bash
# Host venv:
python -m ai_models_ensembles.cli --help
python -m ai_models_ensembles.cli models
python -c "import ai_models_ensembles.cli; print('CLI OK')"

# Inside a container (heavy deps):
srun --container-image=$STORE/aurora.sqsh \
  python -c "import earth2studio, swissclim_evaluations; print('container OK')"
```

## Troubleshooting

- **`import earth2studio` fails on the host**: expected. Host venv is
  minimal; the model deps live in the per-model `.sqsh` images. Run inside
  a container via `srun --container-image=$STORE/<model>.sqsh ...`.
- **CDS data source fails**: needs `~/.cdsapirc`. Prefer `arco` for ERA5.
- **Out of memory**: reduce `NUM_MEMBERS` or bump the slurm memory knob in
  the relevant submit script.
- **Container build fails**: must be submitted to a compute node via
  `containers/submit_build.sh`; login nodes can't build.
- **Eval reads zarr slowly**: pre-fix chunking. Reshard with
  [submit_reshard.sh](submit_reshard.sh).

## Slurm shortcuts

```bash
squeue -u $USER
tail -f $STORE/ablation_logs/*.out
tail -f $STORE/baseline_logs/*.out
```

## Expected directory structure

```
$STORE/
├── baselines/<model_id>/<YYYYMMDD_HHMM>/forecast.zarr
└── ablation/
    └── <phase>/<model_id>/
        ├── <init_tag>/<run_tag>/forecast.zarr
        ├── eval/<run_tag>/
        │   ├── maps/  histograms/  energy_spectra/
        │   └── deterministic/  ets/  fss/  probabilistic/  ssim/
        └── intercomparison/
```

## Next steps

- Tune `config/swissclim_eval.yaml.template` modules / metrics / plotting.
- Sweep magnitudes by editing `PHASE1_MAGNITUDES` in `submit_ablation.sh`.
- Compare runs with `evaluate_ablation.sh intercompare <phase>`.
