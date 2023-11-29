#!/usr/bin/zsh
set -e
set -u

# Some paths to avoid using popd and pushd + relative paths
BASE_DIR=$PWD
MODEL_DIR="${BASE_DIR}/${MODEL_NAME}"
DATE_DIR="${MODEL_DIR}/${DATE_TIME}"
PERTURBATION_DIR="${DATE_DIR}/${PERTURBATION}"

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

echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and perturbation $PERTURBATION"
echo "This wil generate roughly $((NUM_MEMBERS * 6 * 2))GB of data"

proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields \
    $MODEL_NAME >${MODEL_DIR}/fields.txt"

proceed_if_not_exists "${BASE_DIR}/era5.grib" "python download_era5.py $MODEL_NAME $DATE_TIME"

create_dir_if_not_exists $MODEL_DIR

ln -sf ${BASE_DIR}/era5_init.grib ${MODEL_DIR}
proceed_if_not_exists "${MODEL_DIR}/weights.tar" "pushd ${MODEL_DIR} && \
    ai-models --input file --file era5_init.grib --download-assets $MODEL_NAME && popd"
python -u create_zarr.py $MODEL_DIR

create_dir_if_not_exists $DATE_DIR

create_dir_if_not_exists $PERTURBATION_DIR

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
    proceed_if_not_exists "${MEMBER_DIR}/${MODEL_NAME}.grib" "pushd ${MEMBER_DIR} && \
        ai-models --input file --file ${MEMBER_DIR}/era5_init.grib $MODEL_NAME && popd"

    create_dir_if_not_exists "${MEMBER_DIR}/animations"
done

python -u create_zarr.py $MODEL_NAME/$DATE_TIME/$PERTURBATION --subdir_search True

if [ -z "$(find ${PERTURBATION_DIR} -name 'spread_skill_ratio*' -print -quit)" ]; then
    echo "Evaluating model and generating figures"
    python -u evaluation.py $MODEL_NAME $DATE_TIME $PERTURBATION
fi

#BUG: should work for all members
if [ -z "$(find ${PERTURBATION_DIR}/$((NUM_MEMBERS - 1))/animations/ -name '*gif' -print -quit)" ]; then
    echo "Generating Animations"
    python -u animator.py $MODEL_NAME/$DATE_TIME/$PERTURBATION
fi

echo "*****DONE*****"