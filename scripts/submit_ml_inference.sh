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

export job1='echo "Running $MODEL_NAME for $DATE_TIME with $NUM_MEMBERS members and initial \
perturbation $PERTURBATION_INIT, latent perturbation $PERTURBATION_LATENT"
echo "This will generate roughly $((NUM_MEMBERS * 7))GB of data"

proceed_if_not_exists "${MODEL_DIR}/${MODEL_NAME}.grib" "pushd ${MODEL_DIR} && \
    ai-models --input file --file init_field.grib --lead-time ${LEAD_TIME} \
    --download-assets $MODEL_NAME && popd"'

export job2='python -u -m ai_models_ensembles.cli infer'

${DRY_RUN:+echo} srun -N1 -n1 -c"${INF_CPUS}" --mem "${INF_MEM}" \
    --output="${LOG_DIR}/out_ml0_%j.log" \
    --error="${LOG_DIR}/err_ml0_%j.log" \
    bash -c "$job1"

# Array mode: if SLURM_ARRAY_TASK_ID is present, run a single member for the current env (set PERTURBATION_LATENT before sbatch)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MEMBER_ID="${SLURM_ARRAY_TASK_ID}"
    python -u -m ai_models_ensembles.cli infer --member "$MEMBER_ID"
else
    # Non-array mode: loop over perturbation values and run full member loops via CLI
    for latent in ${PERTURBATION_LATENTS}; do
        export PERTURBATION_LATENT=$latent
        export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
        export job2
        ${DRY_RUN:+echo} srun -N1 -n1 -c"${INF_CPUS}" --mem "${INF_MEM}" --gres=gpu:"${INF_GPUS}" \
             --output="${LOG_DIR}/out_ml${latent}_%j.log" \
             --error="${LOG_DIR}/err_ml${latent}_%j.log" \
             bash -c "$job2" &
    done
    wait
fi

# # Run with different perturbation layers
# for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
#     export LAYER=$layer
#     export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
#     export job2
#     srun -N1 -n1 -c32 --gres=gpu:1 \
#          --output=logs/out_ml${layer}_%j.log \
#          --error=logs/err_ml${layer}_%j.log \
#          bash -c "$job2"
# done
# wait
