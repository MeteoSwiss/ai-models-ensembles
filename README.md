# AI Model Ensembles for Weather Forecasting

Run GraphCast and FourCastNetV2 ensembles, convert outputs to Zarr, and verify against IFS/ERA5 with ready-made plots and animations.

## Quickstart

1. Configure `config.sh`
   - Set `OUTPUT_DIR`, `DATE_TIME`, `MODEL_NAME`, `NUM_MEMBERS`, perturbation values, and region.
   - Optional: set `EARTHKIT_CACHE_DIR` to a writable path for Earthkit cache.
1. Create env and activate

```bash
mamba env create -f environment.yml
conda activate ai_models_ens
```

1. Submit jobs (Slurm)
   - Download ERA5 + IFS: `submit_download_data.sh`
   - ML inference: `submit_ml_inference.sh`
   - Convert to Zarr: `submit_convert_zarr.sh`
   - Verify + plots: `submit_verification.sh`

Logs are written to `logs/`. Adjust `config.sh` to tailor runs.

## Requirements (short)

- Linux. Slurm recommended for the provided submit scripts.
- ECMWF credentials for ERA5/IFS (configure `~/.cdsapirc`; MARS if needed).
- GPU (CUDA 12.2) for model inference; CPU is fine for plotting/conversion.
- GRIB readers: `cfgrib` + `ecCodes` (install via conda-forge if missing).

## Configuration

Edit `config.sh`:

- `DATE_TIME`, `MODEL_NAME`, `LAYER`, `PERTURBATION_INIT`, `PERTURBATION_LATENT`, `NUM_MEMBERS`, `CROP_REGION`, `OUTPUT_DIR`
- Optional: set an Earthkit cache directory before running (applies to downloads):

```bash
export EARTHKIT_CACHE_DIR=/your/writable/cache
```

## Manual usage (advanced)

Generate fields file:

```bash
ai-models --fields graphcast > $OUTPUT_DIR/$DATE_TIME/graphcast/fields.txt
```

Downloads:

```bash
python -m ai_models_ensembles.download_re_analysis $OUTPUT_DIR $DATE_TIME $END_DATE_TIME $INTERVAL $MODEL_NAME
python -m ai_models_ensembles.download_ifs_ensemble $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME
python -m ai_models_ensembles.download_ifs_control $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME
```

Perturbations and inference:

```bash
python -m ai_models_ensembles.perturb_fourcastnet_weights $OUTPUT_DIR $DATE_TIME $MODEL_NAME $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER $LAYER
python -m ai_models_ensembles.perturb_graphcast_weights $OUTPUT_DIR $DATE_TIME $MODEL_NAME $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER $LAYER
python -m ai_models_ensembles.perturb_era5 $OUTPUT_DIR $DATE_TIME $MODEL_NAME $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER

# inside $OUTPUT_DIR/$DATE_TIME/$MODEL_NAME
ai-models --input file --file init_field.grib --lead-time 240 --download-assets $MODEL_NAME
```

Convert and verify:

```bash
python -m ai_models_ensembles.convert_grib_to_zarr "$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME"
python -m ai_models_ensembles.convert_grib_to_zarr "$OUTPUT_DIR/$DATE_TIME/$MODEL_NAME/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}" --subdir_search True
python -m ai_models_ensembles.plot_0d_distributions "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
python -m ai_models_ensembles.plot_1d_timeseries "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
python -m ai_models_ensembles.animate_2d_maps "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
python -m ai_models_ensembles.animate_3d_grids "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
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
- For GPU inference, match CUDA 12.2 drivers/toolkit to the environment
- If GraphCast local repo is missing and env install fails, clone `../ai-models-graphcast` or update `environment.yml`

## License

See `LICENSE`.
