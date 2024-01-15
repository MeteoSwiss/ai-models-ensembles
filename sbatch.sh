#!/bin/bash -l
#SBATCH --job-name=ai-models-ensembles
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH --time=23:59:00
#SBATCH --no-requeue
#SBATCH --mem=490G

export DATE_TIME=199912250000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=fourcastnetv2-small
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.01
export NUM_MEMBERS=100

# if conda env ai-models does not exist then create it
if ! conda env list | grep -q ai-models; then
    conda env create -f environment.yml
fi

conda activate ai-models

chmod +x runscript.sh

srun runscript.sh
