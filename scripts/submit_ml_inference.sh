#!/usr/bin/bash -l
set -euo pipefail
IFS=$'\n\t'
#SBATCH --job-name=${INF_JOB_NAME}
#SBATCH --nodes=${INF_NODES_SB}
#SBATCH --ntasks=${INF_NTASKS_SB}
#SBATCH --cpus-per-task=${INF_CPUS_PER_TASK_SB}
#SBATCH --mem-per-cpu=${INF_MEM_PER_CPU_SB}
#SBATCH --partition=${INF_PARTITION_SB}
#SBATCH --gres=${INF_GRES_SB}
#SBATCH --account=${INF_ACCOUNT_SB}
#SBATCH --output=${LOG_DIR}/out_ml_%A_%a.log
#SBATCH --error=${LOG_DIR}/err_ml_%A_%a.log
#SBATCH --time=${INF_TIME_SB}
#SBATCH --no-requeue

# Change to repository root
cd "$(dirname "$0")/.." || exit 1

source ./scripts/config.sh
bash ./tools/validate.sh

# One srun per perturbation latent. Each srun runs `ai-ens infer` inside the
# enroot container, which loads the e2s model, builds the IC source, and
# rolls $NUM_MEMBERS members into $PERTURBATION_DIR/forecast.zarr.
run_one() {
    local latent="$1"
    export PERTURBATION_LATENT="$latent"
    export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
    mkdir -p "${PERTURBATION_DIR}"

    local container_args=()
    if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
        container_args=(--container-image="${CONTAINER_IMAGE}")
    fi

    ${DRY_RUN:+echo} srun -N1 -n1 -c"${INF_CPUS}" --mem "${INF_MEM}" \
        --gres=gpu:"${INF_GPUS}" \
        "${container_args[@]}" \
        --output="${LOG_DIR}/out_ml${latent}_%j.log" \
        --error="${LOG_DIR}/err_ml${latent}_%j.log" \
        ai-ens infer \
            --model "${MODEL_NAME}" \
            --init "${INIT_ISO}" \
            --lead-hours "${LEAD_TIME}" \
            --members "${NUM_MEMBERS}" \
            --ic-magnitude "${PERTURBATION_INIT}" \
            --weight-magnitude "${PERTURBATION_LATENT}" \
            --layer "${LAYER}" \
            --data-source "${DATA_SOURCE:-arco}" \
            --output-levels "${OUTPUT_LEVELS}" \
            --output "${PERTURBATION_DIR}/forecast.zarr" &
}

# Convert YYYYMMDDHHMM to ISO-8601 once.
INIT_ISO="${DATE_TIME:0:4}-${DATE_TIME:4:2}-${DATE_TIME:6:2}T${DATE_TIME:8:2}:${DATE_TIME:10:2}"
export INIT_ISO

for latent in ${PERTURBATION_LATENTS}; do
    run_one "$latent"
done
wait
