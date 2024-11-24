#!/usr/bin/bash -l
#SBATCH --job-name=ai_verif
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --partition=pp-long
#SBATCH --account=s83
#SBATCH --output=logs/out_verif_%j.log
#SBATCH --error=logs/err_verif_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

source ./config.sh

export job='create_dir_if_not_exists "$REGION_DIR"
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

run_jobs() {
    local latents=("$@")
    for latent in "${latents[@]}"; do
        export PERTURBATION_LATENT=$latent
        export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
        export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"
        srun -N1 -n1 -c128 --mem=222 --output=logs/out_verif${latent}_%j.log \
             --error=logs/err_verif${latent}_%j.log \
             bash -c "$job" &
    done
    wait
}

# Run with different perturbation values
run_jobs 0.0 0.01
# run_jobs 0.02 0.03
# run_jobs 0.04 0.05
# run_jobs 0.06 0.07
# run_jobs 0.08 0.09
# run_jobs 0.008 0.006
# run_jobs 0.004 0.002

# # Run with different perturbation layers
# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
#     export LAYER=$layer
#     export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
#     export job
#     srun -N1 -n1 -c32 \
#          --output=logs/out_verif${layer}_%j.log \
#          --error=logs/err_verif${layer}_%j.log \
#          bash -c "$job"
# done
# wait
