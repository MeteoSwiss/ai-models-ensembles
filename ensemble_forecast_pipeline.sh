#!/usr/bin/env bash
set -e
set -u

create_dir_if_not_exists() {
    if ! test -d "$1"; then
        echo "Creating directory $1"
        mkdir -p "$1"
    fi
}

proceed_if_not_exists() {
    if ! test -f "$1"; then
        echo "Executing command: $2"
        eval "$2"
    fi
}

echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and initial \
perturbation $PERTURBATION_INIT, latent perturbation $PERTURBATION_LATENT"
echo "This will generate roughly $((NUM_MEMBERS * 6 * 2))GB of data"

create_dir_if_not_exists "$OUTPUT_DIR/source_files"
cp -r "$SRC_DIR"/* "$OUTPUT_DIR/source_files"

create_dir_if_not_exists "$DATE_DIR"
create_dir_if_not_exists "$MODEL_DIR"

proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields $MODEL_NAME > ${MODEL_DIR}/fields.txt"

proceed_if_not_exists "${MODEL_DIR}/era5_init.grib" "python -m ai_models_ensembles.download_era5 \
    $OUTPUT_DIR $DATE_TIME $END_DATE_TIME $INTERVAL $MODEL_NAME"
proceed_if_not_exists "${MODEL_DIR}/ifs_ens.zarr/.zmetadata" "python -m ai_models_ensembles.download_ifs_ensemble \
    $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME"
proceed_if_not_exists "${MODEL_DIR}/ifs_control.zarr/.zmetadata" "python -m ai_models_ensembles.download_ifs_control \
    $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME"

proceed_if_not_exists "${MODEL_DIR}/${MODEL_NAME}.grib" "pushd ${MODEL_DIR} && \
    ai-models --input file --file era5_init.grib --lead-time ${LEAD_TIME} \
    --download-assets $MODEL_NAME && popd"
python -u -m ai_models_ensembles.convert_grib_to_zarr "$MODEL_DIR"

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

python -u -m ai_models_ensembles.convert_grib_to_zarr "$PERTURBATION_DIR" --subdir_search True

if ! test -d "${REGION_DIR}/png_${MODEL_NAME}"; then
    echo "Evaluating model and generating figures"
    python -u -m ai_models_ensembles.plot_0d_distributions "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
        "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
    python -u -m ai_models_ensembles.plot_1d_timeseries "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
        "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
fi

# if [ -z "$(find "${PERTURBATION_DIR}/${CROP_REGION}/0/animations/" -name '*gif' -print -quit 2>/dev/null)" ]; then
    echo "Generating Animations"
    python -u -m ai_models_ensembles.animate_2d_maps "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
            "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
    # python -u -m ai_models_ensembles.animate_3d_grids "$OUTPUT_DIR" "$DATE_TIME" "$MODEL_NAME" "$PERTURBATION_INIT" \
    #         "$PERTURBATION_LATENT" "$LAYER" "$NUM_MEMBERS" "$CROP_REGION"
# fi

echo "Cleaning up GRIB files"
if command -v fd &>/dev/null; then
    fd -IH --type f ".grib" "${REGION_DIR}" -x rm {}
else
    find "${REGION_DIR}" -type f -name "*.grib" -delete
fi

echo "*****DONE*****"
