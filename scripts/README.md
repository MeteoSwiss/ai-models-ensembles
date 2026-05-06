# Workflow Execution Scripts

Slurm wrappers around `ai-ens infer` (earth2studio) and `ai-ens verify`
(SwissClim Evaluations). Configuration lives in [config.sh](config.sh).

## config.sh

**Run-level inputs:**

- `OUTPUT_DIR`: base output directory
- `DATE_TIME`: forecast init time (YYYYMMDDHHMM); converted to ISO-8601 inside
  the inference script
- `MODEL_NAME`: one of `graphcast_operational`, `sfno`, `aurora`, `fcn3`,
  `atlas` (see `ai-ens models`)
- `NUM_MEMBERS`: ensemble members per perturbation (deterministic models) or
  per re-seeded sample (probabilistic models)
- `LEAD_TIME`: total forecast lead in hours (must be a multiple of the model step)
- `CROP_REGION`: passed through to the SwissClim YAML templates
- `IFS_ENS_PATH`: absolute path to your on-disk IFS ENS zarr (SwissClim
  format, full lead time, 50 members). Verified directly as the physical-
  model baseline.
- `TARGET_PATH`: ERA5 zarr used as SwissClim's verification target
- `DATA_SOURCE`: earth2studio data source for the IC fields. Defaults to
  `arco` (ARCO ERA5). Other valid values:
  `arco | cds | gfs | ifs | ifs_ens | wb2 | file:/path | ifs_analysis:/path`

**Perturbations:**

- `PERTURBATION_INIT`: sigma of multiplicative IC noise (0 disables)
- `PERTURBATION_LATENT`: sigma of multiplicative weight noise (0 disables)
- `PERTURBATION_LATENTS`: space-separated sweep values; one Slurm task per value
- `LAYER`: integer index of the weight tensor to perturb, or empty for all

**Container / SwissClim:**

- `CONTAINER_IMAGE`: optional `.sqsh` produced by `containers/build.sh`. When
  set, srun launches `ai-ens` inside the container via pyxis. When empty, runs
  on the host's `.venv`.
- `SWISSCLIM_CONFIG_TEMPLATE`: defaults to `config/swissclim_eval.yaml.template`
- `SWISSCLIM_CONFIG`: optional explicit YAML; bypasses templating

Slurm resource knobs are at the bottom of `config.sh` (`INF_*`, `VERIF_*`).

## submit_ml_inference.sh

For each value of `$PERTURBATION_LATENTS`, launches one srun running
`ai-ens infer` with the corresponding sigma. Each srun:

1. Loads the registered earth2studio model (downloading the default `Package`
   on first use).
2. Builds the IC source from `$DATA_SOURCE`.
3. Optionally perturbs the IC and/or model weights per member
   (`materialise_perturbed_package` mirrors the NGC layout under
   `$PERTURBATION_DIR/_e2s_work/weights_member_NNN/`).
4. Rolls `NUM_MEMBERS` members into a single SwissClim-format
   `$PERTURBATION_DIR/forecast.zarr`.

```bash
sbatch scripts/submit_ml_inference.sh
DRY_RUN=1 sbatch scripts/submit_ml_inference.sh   # echo srun lines only
```

Outputs:

- `$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_${INIT}_latent_${LAT}_layer_${LAYER}/forecast.zarr/`
- Per-member checkpoint mirrors under `_e2s_work/` (only when
  `PERTURBATION_LATENT > 0`)

## submit_verification.sh

Runs SwissClim Evaluations against the AI-model forecasts and the on-disk
IFS ENS forecast in one shot:

1. **AI models.** For each value of `$PERTURBATION_LATENTS`, renders
   [`config/swissclim_eval.yaml.template`](../config/swissclim_eval.yaml.template)
   into `$REGION_DIR/swissclim_eval.yaml` and runs `ai-ens verify`.
2. **IFS ENS baseline.** Renders
   [`config/swissclim_ifs_ens.yaml.template`](../config/swissclim_ifs_ens.yaml.template)
   targeting `$IFS_ENS_PATH` as the prediction; output goes to
   `$OUTPUT_DIR/$DATE_TIME/_ifs_ens/$CROP_REGION/swissclim_ifs_ens/`.

```bash
sbatch scripts/submit_verification.sh
SWISSCLIM_CONFIG=/abs/path/to/my_run.yaml sbatch scripts/submit_verification.sh

# Skip IFS baseline (e.g. when re-running just the AI side):
VERIFY_IFS_ENS=0 sbatch scripts/submit_verification.sh
```

The IFS ENS job is skipped automatically if `IFS_ENS_PATH` is still at its
placeholder default.

Outputs (one `swissclim_*` tree per verify run):

- `maps/`, `histograms/`, `wd_kde/`, `energy_spectra/`, `vertical_profiles/`
- `deterministic/`, `ets/`, `probabilistic/`, `ssim/`

See [the SwissClim docs](https://github.com/swiss-ai/SwissClim_Evaluations/blob/research/docs/OUTPUTS.md)
for the full output schema.

## Typical workflow sequence

```bash
# 1. Configure
vim scripts/config.sh

# 2. (Optional) build the GH200 container
./containers/build.sh
export CONTAINER_IMAGE=$PWD/ai-ens.sqsh

# 3. Validate
bash ./tools/validate.sh

# 4. Inference (one srun per PERTURBATION_LATENTS value)
sbatch scripts/submit_ml_inference.sh

# 5. Verification (SwissClim)
sbatch scripts/submit_verification.sh

# 6. Intercompare runs (recursive: AI models + IFS baselines)
ai-ens intercompare    # auto-discovers all swissclim_* roots under $OUTPUT_DIR/$DATE_TIME

# 7. Status
./tools/check_workflow_status.sh
```

## Advanced usage

**Sweep over weight perturbation magnitudes** - edit `PERTURBATION_LATENTS` in
`config.sh`; the inference script runs all values in parallel as separate srun
tasks.

**Different models** - set `MODEL_NAME` to any entry in `ai-ens models`. The
data-source string in `DATA_SOURCE` is independent of model choice.

**Custom IC source** - set `DATA_SOURCE=file:/path/to/era5_ic.zarr` to feed a
locally-prepared zarr instead of pulling from ARCO/CDS.

**Dry run** - `DRY_RUN=1 sbatch scripts/submit_ml_inference.sh` echoes the
srun lines without executing them.

## Troubleshooting

- **Job won't start**: check `sinfo` and account access.
- **Container launch fails**: confirm `CONTAINER_IMAGE` points at an existing
  `.sqsh` and that pyxis is available on your partition.
- **Out of memory during inference**: reduce `NUM_MEMBERS` or bump
  `INF_MEM_PER_CPU_SB` in `config.sh`.
- **`ai-ens models` not found**: activate the venv (`source .venv/bin/activate`)
  or run inside the built container.
- **earth2studio download stalls**: set `EARTH2STUDIO_CACHE` to a writable
  scratch path; first model load downloads the NGC `Package` (~GB-scale).

## See also

- [../README.md](../README.md) - top-level overview
- [../tools/README.md](../tools/README.md) - environment setup + monitoring
- [../containers/](../containers/) - GH200 container build pipeline
