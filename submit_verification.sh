#!/usr/bin/bash -l
#SBATCH --job-name=ai_verif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --output=logs/out_verif_%j.log
#SBATCH --error=logs/err_verif_%j.log
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --mem=200G

source ./config.sh

srun bash -c '
if ! test -d "${REGION_DIR}/png_${MODEL_NAME}"; then
    echo "Evaluating model and creating 0-dimensional figures"
    python -u -m ai_models_ensembles.plot_0d_distributions "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
        "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
    echo "Creating 1-dimensional figures (lineplots)"
    python -u -m ai_models_ensembles.plot_1d_timeseries "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
        "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
fi

if [ -z "$(find "${PERTURBATION_DIR}/${CROP_REGION}/0/animations/" -name '*gif' -print -quit 2>/dev/null)" ]; then
    echo "Generating 2-dimensional animations (maps)"
    python -u -m ai_models_ensembles.animate_2d_maps "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
            "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
    echo "Generating 3-dimensional animations (hypercubes)"
    python -u -m ai_models_ensembles.animate_3d_grids "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
            "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
fi

echo "Cleaning up GRIB files"
if command -v fd &>/dev/null; then
    fd -IH --type f ".grib" "${REGION_DIR}" -x rm {}
else
    find "${REGION_DIR}" -type f -name "*.grib" -delete
fi
echo "*****DONE*****"
'