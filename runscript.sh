#!/usr/bin/zsh
set -e
set -u

# Some paths to avoid using popd and pushd + relative paths
BASE_DIR=$PWD
DATE_DIR="${BASE_DIR}/${DATE_TIME}"
MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}"

# All AI-models produce 10 days of forecasts with 6-hourly intervals
NUM_DAYS=10
END_DATE_TIME=$(date -d "${DATE_TIME:0:8} + $(($NUM_DAYS)) days" +%Y%m%d)0000
INTERVAL=6

create_dir_if_not_exists() {
        if [[ ! -d $1 ]]; then
                echo "Creating directory $1"
                mkdir -p $1
        fi
}

proceed_if_not_exists() {
        if [[ ! -f $1 ]]; then
                echo "Executing command: $2"
                eval $2
        fi
}

echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and initial \
perturbation $PERTURBATION_INIT, latent perturbation $PERTURBATION_LATENT"
echo "This wil generate roughly $((NUM_MEMBERS * 6 * 2))GB of data"

create_dir_if_not_exists $DATE_DIR
create_dir_if_not_exists $MODEL_DIR

proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields \
    $MODEL_NAME >${MODEL_DIR}/fields.txt"

proceed_if_not_exists "${MODEL_DIR}/era5_init.grib" "python download_era5.py \
    $DATE_TIME $END_DATE_TIME $INTERVAL $MODEL_NAME"
proceed_if_not_exists "${MODEL_DIR}/ifs_ens.zarr/.zmetadata" "python download_ifs.py \
    $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME"

ln -sf ${DATE_DIR}/era5_init.grib ${MODEL_DIR}
proceed_if_not_exists "${MODEL_DIR}/weights.tar" "pushd ${MODEL_DIR} && \
    ai-models --input file --file era5_init.grib --download-assets $MODEL_NAME && popd"
python -u create_zarr.py $MODEL_DIR

create_dir_if_not_exists $PERTURBATION_DIR

for MEMBER in $(seq 0 $((NUM_MEMBERS - 1))); do
        MEMBER_DIR="${PERTURBATION_DIR}/${MEMBER}"
        create_dir_if_not_exists $MEMBER_DIR
        if [[ $(echo "$PERTURBATION_INIT > 0.0" | bc -l) -eq 1 ]]; then
                proceed_if_not_exists "${MEMBER_DIR}/era5_init.grib" \
                        proceed_if_not_exists "${MEMBER_DIR}/era5_init.grib" \
                        "python -u ${BASE_DIR}/perturb_era5.py $DATE_TIME $MODEL_NAME \
            $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER"
        else
                ln -sf ${DATE_DIR}/era5_init.grib ${MEMBER_DIR}/era5_init.grib
        fi

        if [[ $(echo "$PERTURBATION_LATENT > 0.0" | bc -l) -eq 1 ]]; then
                proceed_if_not_exists "${MEMBER_DIR}/weights.tar" \
                        "python -u ${BASE_DIR}/perturb_fourcastnet.py $DATE_TIME $MODEL_NAME \
            $PERTURBATION_INIT $PERTURBATION_LATENT $MEMBER"
        else
                ln -sf ${MODEL_DIR}/weights.tar ${MEMBER_DIR}/weights.tar
        fi
        ln -sf ${MODEL_DIR}/global_means.npy ${MEMBER_DIR}/global_means.npy
        ln -sf ${MODEL_DIR}/global_stds.npy ${MEMBER_DIR}/global_stds.npy
        # Run the model from a local GRIB-file
        proceed_if_not_exists "${PERTURBATION_DIR}/forecast.zarr/member/${MEMBER}" \
                "pushd ${MEMBER_DIR} &&  ai-models --input file --file \
        ${MEMBER_DIR}/era5_init.grib $MODEL_NAME && popd"

        create_dir_if_not_exists "${MEMBER_DIR}/animations"
done

python -u create_zarr.py $PERTURBATION_DIR --subdir_search True

if [ -z "$(find ${PERTURBATION_DIR} -name 'rmse_v100.png' -print -quit)" ]; then
        echo "Evaluating model and generating figures"
        python -u evaluation.py $DATE_TIME $MODEL_NAME $PERTURBATION_INIT \
                $PERTURBATION_LATENT $NUM_MEMBERS
fi

#BUG: should work for all members
if [ -z "$(find ${PERTURBATION_DIR}/0/animations/ -name '*gif' -print -quit)" ]; then
        echo "Generating Animations"
        python -u animator.py $DATE_TIME $MODEL_NAME $PERTURBATION_INIT $PERTURBATION_LATENT
        python -u animator_3d.py $DATE_TIME $MODEL_NAME $PERTURBATION_INIT $PERTURBATION_LATENT
fi

echo "Cleaning up GRIB files"
fd -IH --type f ".grib" "${PERTURBATION_DIR}" -x rm {}

echo "*****DONE*****"