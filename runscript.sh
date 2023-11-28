#!/usr/bin/zsh
set -e
set -u

# Some paths to avoid using popd and pushd + relative paths
BASE_DIR=$PWD
MODEL_DIR="${BASE_DIR}/${MODEL_NAME}"
DATE_DIR="${MODEL_DIR}/${DATE_TIME}"
PERTURBATION_DIR="${DATE_DIR}/${PERTURBATION}"

echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and perturbation $PERTURBATION"
echo "This wil generate roughly $((NUM_MEMBERS * 6 * 2))GB of data"

create_dir_if_not_exists() {
    if [[ ! -d $1 ]]; then
        mkdir -p $1
    fi
}

proceed_if_not_exists() {
    if [[ ! -f $1 ]]; then
        eval $2
    fi
}

echo "Retrieving required variables and levels"
proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields \
    $MODEL_NAME >${MODEL_DIR}/fields.txt"

echo "Retrieving initial conditions from ERA5"
proceed_if_not_exists "${BASE_DIR}/era5.grib" "python download_era5.py $MODEL_NAME $DATE_TIME"

echo "Perturbing initial conditions"
create_dir_if_not_exists $MODEL_DIR

echo "Downloading model checkpoint and statistics"
proceed_if_not_exists "${MODEL_DIR}/weights.tar" "ai-models --input cds --date \
    ${DATE_TIME:0:8} --time ${DATE_TIME:8:4} --lead-time 6 --download-assets $MODEL_NAME"

echo "Downloading global means and standard deviations"
create_dir_if_not_exists $DATE_DIR

create_dir_if_not_exists $PERTURBATION_DIR

echo "Generating ensemble forecast with $NUM_MEMBERS members"
for MEMBER in $(seq 0 $((NUM_MEMBERS - 1))); do
    MEMBER_DIR="${PERTURBATION_DIR}/${MEMBER}"
    create_dir_if_not_exists $MEMBER_DIR
    # Perturb the initial data
    proceed_if_not_exists "${MEMBER_DIR}/era5_init.grib" \
        "python -u ${BASE_DIR}/perturb_era5.py $MODEL_NAME $DATE_TIME $PERTURBATION $MEMBER"
    proceed_if_not_exists "${MEMBER_DIR}/weights.tar" \
        "python -u ${BASE_DIR}/perturb_fourcastnet.py $MODEL_NAME $DATE_TIME $PERTURBATION $MEMBER"
    ln -sf ${MODEL_DIR}/global_means.npy ${MEMBER_DIR}/global_means.npy
    ln -sf ${MODEL_DIR}/global_stds.npy ${MEMBER_DIR}/global_stds.npy
    # Run the model from a local GRIB-file
    #TODO: also perturb the model weights
    proceed_if_not_exists "${MEMBER_DIR}/${MODEL_NAME}.grib" "pushd ${MEMBER_DIR} && \
        ai-models --input file --file ${MEMBER_DIR}/era5_init.grib $MODEL_NAME && popd"

    create_dir_if_not_exists "${MEMBER_DIR}/animations"
done

echo "Creating zarr-file for the ensemble forecast"
python -u create_zarr.py $MODEL_NAME/$DATE_TIME/$PERTURBATION

echo "Evaluating model and generate figures"
if [ -z "$(find ${PERTURBATION_DIR} -name 'spread_skill_ratio*' -print -quit)" ]; then
    python -u evaluation.py $MODEL_NAME/$DATE_TIME/$PERTURBATION
fi

#BUG: should work for all members
echo "Generate Animations"
if [ -z "$(find ${PERTURBATION_DIR}/0/animations/ -name '*gif' -print -quit)" ]; then
    python -u animator.py $MODEL_NAME/$DATE_TIME/$PERTURBATION
fi

echo "*****DONE*****"
