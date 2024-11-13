#!/usr/bin/bash -l
#SBATCH --job-name=ai_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=logs/out_ml_%j.log
#SBATCH --error=logs/err_ml_%j.log
#SBATCH --time=18:00:00
#SBATCH --mem=444G
#SBATCH --no-requeue

source ./config.sh

srun bash -c '
echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and initial \
perturbation $PERTURBATION_INIT, latent perturbation $PERTURBATION_LATENT"
echo "This will generate roughly $((NUM_MEMBERS * 7))GB of data"

proceed_if_not_exists "${MODEL_DIR}/${MODEL_NAME}.grib" "pushd ${MODEL_DIR} && \
    ai-models --input file --file era5_init.grib --lead-time ${LEAD_TIME} \
    --download-assets $MODEL_NAME && popd"

create_dir_if_not_exists "$PERTURBATION_DIR"
create_dir_if_not_exists "$REGION_DIR"

for MEMBER in $(seq 0 $((NUM_MEMBERS - 1))); do
    MEMBER_DIR="${PERTURBATION_DIR}/${MEMBER}"
    create_dir_if_not_exists "$MEMBER_DIR"
    if [ "$(echo "$PERTURBATION_INIT > 0.0" | bc -l)" -eq 1 ]; then
        proceed_if_not_exists "${MEMBER_DIR}/era5_init.grib" \
            "python -u -m ai_models_ensembles.perturb_era5 $OUTPUT_DIR $DATE_TIME $MODEL_NAME \
            $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER"
    else
        ln -sf "${MODEL_DIR}/era5_init.grib" "${MEMBER_DIR}/era5_init.grib"
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
    ${MEMBER_DIR}/era5_init.grib --lead-time ${LEAD_TIME} $MODEL_NAME && popd"

done
echo "*****DONE*****"
'
