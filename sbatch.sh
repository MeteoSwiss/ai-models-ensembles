#!/usr/bin/bash -l
#SBATCH --job-name=ai-models-ensembles
#SBATCH --nodes=1
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --output=logs/logs%j.out
#SBATCH --error=logs/logs%j.err
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --mem=444G

export DATE_TIME=201801010000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=graphcast # fourcastnetv2-small, graphcast
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.01
export NUM_MEMBERS=49
export CROP_REGION=europe # Crop to a specific region

# Some paths to avoid using popd and pushd + relative paths
export BASE_DIR=$PWD
export DATE_DIR="${BASE_DIR}/${DATE_TIME}"
export MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}"
export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"

# All AI-models produce 10 days of forecasts with 6-hourly intervals
export NUM_DAYS=10
export END_DATE_TIME=$(date -d "${DATE_TIME:0:8} + $((NUM_DAYS)) days" +%Y%m%d)0000
export INTERVAL=6

# if conda env ai-models does not exist then create it
if ! mamba env list | grep -q ai-models; then
    mamba env create -n ai-models -f environment.yml
    mamba activate ai-models
    pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

mamba activate ai-models

if [ ! -x runscript.sh ]; then
    chmod +x runscript.sh
fi

srun runscript.sh
