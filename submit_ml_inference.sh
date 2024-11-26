#!/usr/bin/bash -l
#SBATCH --job-name=ai_inf
#SBATCH --nodes=4
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --account=s83
#SBATCH --output=logs/out_ml_%j.log
#SBATCH --error=logs/err_ml_%j.log
#SBATCH --time=18:00:00
#SBATCH --no-requeue

source ./config.sh

export job1='echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and initial \
perturbation $PERTURBATION_INIT, latent perturbation $PERTURBATION_LATENT"
echo "This will generate roughly $((NUM_MEMBERS * 7))GB of data"

proceed_if_not_exists "${MODEL_DIR}/${MODEL_NAME}.grib" "pushd ${MODEL_DIR} && \
    ai-models --input file --file init_field.grib --lead-time ${LEAD_TIME} \
    --download-assets $MODEL_NAME && popd"'

export job2='create_dir_if_not_exists "$PERTURBATION_DIR"

for MEMBER in $(seq 0 $((NUM_MEMBERS - 1))); do
    MEMBER_DIR="${PERTURBATION_DIR}/${MEMBER}"
    create_dir_if_not_exists "$MEMBER_DIR"
    if [ "$(echo "$PERTURBATION_INIT > 0.0" | bc -l)" -eq 1 ]; then
        proceed_if_not_exists "${MEMBER_DIR}/init_field.grib" \
            "python -u -m ai_models_ensembles.perturb_era5 $OUTPUT_DIR $DATE_TIME $MODEL_NAME \
            $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER"
    else
        ln -sf "${MODEL_DIR}/init_field.grib" "${MEMBER_DIR}/init_field.grib"
    fi

    if [ "$(echo "$PERTURBATION_LATENT > 0.0" | bc -l)" -eq 1 ]; then
        if [ "$MODEL_NAME" = "graphcast" ]; then
            if [ ! -d "${MEMBER_DIR}/params" ]; then
                python -u -m ai_models_ensembles.perturb_graphcast_weights $OUTPUT_DIR $DATE_TIME $MODEL_NAME \
                    $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER $LAYER
            fi
        else
            proceed_if_not_exists "${MEMBER_DIR}/weights.tar" \
                "python -u -m ai_models_ensembles.perturb_fourcastnet_weights $OUTPUT_DIR $DATE_TIME $MODEL_NAME \
                    $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER $LAYER"
        fi
    else
        if [ "$MODEL_NAME" = "graphcast" ]; then
            ln -sf "${MODEL_DIR}/params" "${MEMBER_DIR}"
        else
            ln -sf "${MODEL_DIR}/weights.tar" "${MEMBER_DIR}"
        fi
    fi
    if [ "$MODEL_NAME" = "graphcast" ]; then
        ln -sf "${MODEL_DIR}/stats" "${MEMBER_DIR}"
    else
        ln -sf "${MODEL_DIR}/global_means.npy" "${MEMBER_DIR}"
        ln -sf "${MODEL_DIR}/global_stds.npy" "${MEMBER_DIR}"
    fi
    # Run the model from a local GRIB-file
    proceed_if_not_exists "${PERTURBATION_DIR}/forecast.zarr/member/${MEMBER}" \
            "pushd ${MEMBER_DIR} &&  ai-models --input file --file \
    ${MEMBER_DIR}/init_field.grib --lead-time ${LEAD_TIME} $MODEL_NAME && popd"
done
echo "*****DONE*****"'

srun -N1 -n1 -c32 --mem=96 --output=logs/out_ml0_%j.log \
     --error=logs/err_ml0_%j.log \
     bash -c "$job1"

# Run with different perturbation values
for latent in 0.0 0.002 0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do
    export PERTURBATION_LATENT=$latent
    export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
    export job2
    srun -N1 -n1 -c32 --gres=gpu:1 \
         --output=logs/out_ml${latent}_%j.log \
         --error=logs/err_ml${latent}_%j.log \
         bash -c "$job2" &
done
wait

# # Run with different perturbation layers
# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
#     export LAYER=$layer
#     export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
#     export job2
#     srun -N1 -n1 -c32 --gres=gpu:1 \
#          --output=logs/out_ml${layer}_%j.log \
#          --error=logs/err_ml${layer}_%j.log \
#          bash -c "$job2"
# done
# wait
