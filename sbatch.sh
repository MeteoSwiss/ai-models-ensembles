#!/usr/bin/bash -l
#SBATCH --job-name=ai-models-ensembles
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --account=s83
#SBATCH --output=logs/out.log
#SBATCH --error=logs/err.log
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --mem=444G

export DATE_TIME=201801010000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=graphcast # fourcastnetv2-small, graphcast
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.005
export NUM_MEMBERS=2
export CROP_REGION=global # Crop to a specific region
export OUTPUT_DIR=/scratch/mch/sadamov/pyprojects_data/ai-models-ensembles

# Some paths to avoid using popd and pushd + relative paths
export SRC_DIR=$PWD
export BASE_DIR=$OUTPUT_DIR
export DATE_DIR="${BASE_DIR}/${DATE_TIME}"
export MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}"
export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"

# All AI-models produce 10 days of forecasts with 6-hourly intervals
export NUM_DAYS=10
export END_DATE_TIME=$(date -d "${DATE_TIME:0:8} + $((NUM_DAYS)) days" +%Y%m%d)0000
export INTERVAL=6
if MODEL_NAME="graphcast"; then
    export LEAD_TIME=246
else
    export LEAD_TIME=240
fi

# if conda env ai-models does not exist then create it
if ! mamba env list | grep -q ai-models; then
    mamba env create -n ai-models -f $SRC_DIR/environment.yml
    mamba activate ai-models
    pip install -r $SRC_DIR/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -e $SRC_DIR
fi

mamba activate ai-models

if [ ! -x runscript.sh ]; then
    chmod +x $SRC_DIR/runscript.sh
fi

srun $SRC_DIR/runscript.sh
