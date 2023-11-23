#!/usr/bin/zsh
# set -e

MODEL_NAME=fourcastnetv2-small
DATE_TIME=199912250000 # Lothar
NUM_MEMBERS=10

# if conda env ai-models does not exist then create it
if ! conda env list | grep -q ai-models; then
    conda env create -f environment.yml
fi
conda activate ai-models

# Get all variables and levels required by the model
if [ ! -f fields.txt ]; then
    ai-models --fields $MODEL_NAME >fields.txt
fi

# Retrieve the initial data from CDS
if [ ! -f era5.grib ]; then
    python download_era5.py $DATE_TIME
fi

# Create the model-folder if it doesn't exist
mkdir -p $MODEL_NAME && pushd $MODEL_NAME

# download the latest model checkpoint and statistics
if [ ! -f weights.tar ]; then
    echo "Downloading model"
    ai-models --input cds --date ${DATE_TIME:0:8} --time ${DATE_TIME:8:4} --lead-time 6 --download-assets $MODEL_NAME
fi

# Create a folder for DATE_TIME
mkdir -p $DATE_TIME && pushd $DATE_TIME

# Create a folder for each ensemble member
for member in $(seq 1 $NUM_MEMBERS); do
    mkdir -p $member && pushd $member
    # Perturb the initial data
    if [ ! -f era5_init.grib ]; then
        echo "Perturbing member $member"
        python -u ../../../perturb_era5.py $member
    fi
    if [ ! -f weights.tar ]; then
        ln -s ../../../$MODEL_NAME/weights.tar
    fi
    if [ ! -f global_means.npy ]; then
        ln -s ../../../$MODEL_NAME/global_means.npy
    fi
    if [ ! -f global_stds.npy ]; then
        ln -s ../../../$MODEL_NAME/global_stds.npy
    fi
    # Run the model from a local GRIB-file
    #TODO: also perturb the model weights
    if [ ! -f ${MODEL_NAME}.grib ]; then
        ai-models --input file --file era5_init.grib $MODEL_NAME
    fi
    # Create forecast animations:
    if [ ! -f era5.grib ]; then
        ln -s ../../../era5.grib
    fi
    mkdir -p animations
    popd
done

popd
popd

# Create a zarr-file for the model output mostly for performance reasons before running
# the evaluation

srg python -u create_zarr.py $MODEL_NAME $DATE_TIME

python -u evaluation.py $MODEL_NAME $DATE_TIME

python -u animator.py $MODEL_NAME $DATE_TIME False
