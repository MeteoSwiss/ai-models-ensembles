#!/usr/bin/bash -l
set -euo pipefail
IFS=$'\n\t'
#SBATCH --job-name=${ZARR_JOB_NAME}
#SBATCH --nodes=${ZARR_NODES_SB}
#SBATCH --ntasks=${ZARR_NTASKS_SB}
#SBATCH --cpus-per-task=${ZARR_CPUS_PER_TASK_SB}
#SBATCH --mem-per-cpu=${ZARR_MEM_PER_CPU_SB}
#SBATCH --partition=${ZARR_PARTITION_SB}
#SBATCH --account=${ZARR_ACCOUNT_SB}
#SBATCH --output=${LOG_DIR}/out_zarr_%j.log
#SBATCH --error=${LOG_DIR}/err_zarr_%j.log
#SBATCH --time=${ZARR_TIME_SB}
#SBATCH --no-requeue

source ./config.sh
bash ./validate.sh

export job1='echo "Converting all grib files to zarr files for $MODEL_NAME and $DATE_TIME"
echo "Converting the unperturbed forecast."
python -u -m ai_models_ensembles.cli convert --path "$MODEL_DIR"'

export job2='echo "Converting the perturbed forecasts."
python -u -m ai_models_ensembles.cli convert --path "$PERTURBATION_DIR" --subdir-search
echo "*****DONE*****"
'

# Convert the unperturbed forecast
${DRY_RUN:+echo} srun -N1 -n1 -c"${ZARR_CPUS}" --mem "${ZARR_MEM}" \
     --output="${LOG_DIR}/out_zarr0_%j.log" \
     --error="${LOG_DIR}/err_zarr0_%j.log" \
     bash -c "$job1"

wait

# Run with different perturbation values
for latent in ${PERTURBATION_LATENTS}; do
    export PERTURBATION_LATENT=$latent
    export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
    export job2
     ${DRY_RUN:+echo} srun -N1 -n1 -c"${ZARR_CPUS}" --mem "${ZARR_MEM}" \
           --output="${LOG_DIR}/out_zarr${latent}_%j.log" \
           --error="${LOG_DIR}/err_zarr${latent}_%j.log" \
           bash -c "$job2" &
done
wait
