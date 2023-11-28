#!/bin/bash -l
#SBATCH --job-name=ai-models-ensembles
#SBATCH --nodes=1
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=logs.out
#SBATCH --error=logs.err

export MODEL_NAME=fourcastnetv2-small
export DATE_TIME=199912250000 # Lothar
export PERTURBATION=1.0
export NUM_MEMBERS=5

# if conda env ai-models does not exist then create it
if ! conda env list | grep -q ai-models; then
    conda env create -f environment.yml
fi

conda activate ai-models

chmod +x runscript.sh

srun runscript.sh
