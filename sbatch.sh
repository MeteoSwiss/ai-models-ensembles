#!/usr/bin/bash -l
#SBATCH --job-name=ai-models-ensembles
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=logs%j.out
#SBATCH --error=logs%j.err
#SBATCH --time=23:59:00
#SBATCH --no-requeue
#SBATCH --mem=444G

export DATE_TIME=201801010000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=fourcastnetv2-small # fourcastnetv2-small, graphcast
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.02
export NUM_MEMBERS=49

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
