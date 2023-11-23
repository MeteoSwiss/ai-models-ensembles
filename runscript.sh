#!/usr/bin/zsh
set -e
set -u

srun -N1 -n1 --gres=gpu:4 --time=24:00:00 --partition=a100-80gb --account=s83 --pty zsh

MODEL_NAME=fourcastnetv2-small
DATE_TIME=199912250000 # Lothar
NUM_MEMBERS=5
PERTURBATION=0

# Some paths to avoid using popd and pushd + relative paths
BASE_DIR=$PWD
MODEL_DIR="${BASE_DIR}/${MODEL_NAME}"
DATE_DIR="${MODEL_DIR}/${DATE_TIME}"
PERTURBATION_DIR="${DATE_DIR}/${PERTURBATION}"

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

# if conda env ai-models does not exist then create it
if ! conda env list | grep -q ai-models; then
    conda env create -f environment.yml
fi
conda activate ai-models

# Get all variables and levels required by the model
proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields \
    $MODEL_NAME >${MODEL_DIR}/fields.txt"

# Retrieve the initial data from CDS
proceed_if_not_exists "${BASE_DIR}/era5.grib" "python download_era5.py $MODEL_NAME $DATE_TIME"

# Create the model-folder if it doesn't exist
create_dir_if_not_exists $MODEL_DIR

# download the latest model checkpoint and statistics
proceed_if_not_exists "${MODEL_DIR}/weights.tar" "ai-models --input cds --date \
    ${DATE_TIME:0:8} --time ${DATE_TIME:8:4} --lead-time 6 --download-assets $MODEL_NAME"

# Create a folder for DATE_TIME
create_dir_if_not_exists $DATE_DIR

# Create a folder for the initial perturbation
create_dir_if_not_exists $PERTURBATION_DIR

# Create a folder for each ensemble member
for MEMBER in $(seq 1 $NUM_MEMBERS); do
    MEMBER_DIR="${PERTURBATION_DIR}/${MEMBER}"
    create_dir_if_not_exists $MEMBER_DIR
    # Perturb the initial data
    proceed_if_not_exists "${MEMBER_DIR}/era5_init.grib" \
        "python -u ${BASE_DIR}/perturb_era5.py $MODEL_NAME $DATE_TIME $MEMBER $PERTURBATION"
    ln -sf ${MODEL_DIR}/weights.tar ${MEMBER_DIR}/weights.tar
    ln -sf ${MODEL_DIR}/global_means.npy ${MEMBER_DIR}/global_means.npy
    ln -sf ${MODEL_DIR}/global_stds.npy ${MEMBER_DIR}/global_stds.npy
    # Run the model from a local GRIB-file
    #TODO: also perturb the model weights
    proceed_if_not_exists "${MEMBER_DIR}/${MODEL_NAME}.grib" "pushd ${MEMBER_DIR} && \
        ai-models --input file --file ${MEMBER_DIR}/era5_init.grib $MODEL_NAME && popd"

    create_dir_if_not_exists "${MEMBER_DIR}/animations"
done

# Create a zarr-file for the model output mostly for performance reasons before running
# the evaluation
proceed_if_not_exists "${PERTURBATION_DIR}/forecast.zarr" "python -u create_zarr.py \
    $MODEL_NAME/$DATE_TIME/$PERTURBATION"

if ! ls ${PERTURBATION_DIR}/spread_skill_ratio* 1>/dev/null  2>&1; then
    echo "Evaluating model"
    python -u evaluation.py $MODEL_NAME/$DATE_TIME/$PERTURBATION
fi

if ! ls ${PERTURBATION_DIR}/${NUM_MEMBERS}/animations/* 1>/dev/null  2>&1; then
    python -u animator_mp.py $MODEL_NAME/$DATE_TIME/$PERTURBATION
fi
