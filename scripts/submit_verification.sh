#!/usr/bin/bash -l
set -euo pipefail
IFS=$'\n\t'
#SBATCH --job-name=${VERIF_JOB_NAME}
#SBATCH --nodes=${VERIF_NODES_SB}
#SBATCH --ntasks=${VERIF_NTASKS_SB}
#SBATCH --partition=${VERIF_PARTITION_SB}
#SBATCH --account=${VERIF_ACCOUNT_SB}
#SBATCH --output=${LOG_DIR}/out_verif_%j.log
#SBATCH --error=${LOG_DIR}/err_verif_%j.log
#SBATCH --time=${VERIF_TIME_SB}
#SBATCH --no-requeue
#SBATCH --exclusive=${VERIF_EXCLUSIVE}

# Change to repository root
cd "$(dirname "$0")/.." || exit 1

source ./scripts/config.sh
bash ./tools/validate.sh

export job='python -u -m ai_models_ensembles.cli verify'

run_jobs() {
    local latents=("$@")
    for latent in "${latents[@]}"; do
        export PERTURBATION_LATENT=$latent
        export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
        export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"
           ${DRY_RUN:+echo} srun -N1 -n1 -c"${VERIF_CPUS}" --mem "${VERIF_MEM}" \
               --output="${LOG_DIR}/out_verif${latent}_%j.log" \
               --error="${LOG_DIR}/err_verif${latent}_%j.log" \
               bash -c "$job" &
    done
    wait
}

# Run with different perturbation values
run_jobs ${PERTURBATION_LATENTS}
# run_jobs 0.02 0.03
# run_jobs 0.04 0.05
# run_jobs 0.06 0.07
# run_jobs 0.08 0.09
# run_jobs 0.008 0.006
# run_jobs 0.004 0.002

# # Run with different perturbation layers
# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
#     export LAYER=$layer
#     export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
#     export job
#     srun -N1 -n1 -c32 \
#          --output=logs/out_verif${layer}_%j.log \
#          --error=logs/err_verif${layer}_%j.log \
#          bash -c "$job"
# done
# wait
