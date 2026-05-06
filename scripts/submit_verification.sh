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

require_cmd envsubst

# Variables expanded inside the YAML templates. Anything else stays as-is so
# downstream re-rendering still works.
SWISSCLIM_PLACEHOLDERS='${MODEL_DIR} ${PERTURBATION_DIR} ${REGION_DIR} ${MODEL_NAME} ${DATE_TIME} ${CROP_REGION} ${OUTPUT_DIR} ${TARGET_PATH} ${IFS_ENS_PATH}'

render_swissclim_config() {
    local template="$1"
    local rendered="$2"
    mkdir -p "$(dirname "$rendered")"
    envsubst "${SWISSCLIM_PLACEHOLDERS}" < "$template" > "$rendered"
}

container_args=()
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    container_args=(--container-image="${CONTAINER_IMAGE}")
fi

# --- 1. Verify each AI-model perturbation run ------------------------------
run_ai_models() {
    local latents=("$@")
    for latent in "${latents[@]}"; do
        export PERTURBATION_LATENT=$latent
        export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
        export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"

        local rendered="${REGION_DIR}/swissclim_eval.yaml"
        if [[ -n "${SWISSCLIM_CONFIG:-}" ]]; then
            rendered="${SWISSCLIM_CONFIG}"
        else
            ${DRY_RUN:+echo} render_swissclim_config "${SWISSCLIM_CONFIG_TEMPLATE}" "${rendered}"
        fi

        ${DRY_RUN:+echo} srun -N1 -n1 -c"${VERIF_CPUS}" --mem "${VERIF_MEM}" \
            "${container_args[@]}" \
            --output="${LOG_DIR}/out_verif${latent}_%j.log" \
            --error="${LOG_DIR}/err_verif${latent}_%j.log" \
            ai-ens verify --config "${rendered}" &
    done
    wait
}

# --- 2. Verify the on-disk IFS ENS forecast --------------------------------
run_ifs_ens() {
    local template="config/swissclim_ifs_ens.yaml.template"
    local out_dir="${OUTPUT_DIR}/${DATE_TIME}/_ifs_ens/${CROP_REGION}"
    local rendered="${out_dir}/swissclim_ifs_ens.yaml"

    [[ -f "$template" ]] || { echo "Missing template: $template" >&2; return 1; }
    mkdir -p "${out_dir}"
    ${DRY_RUN:+echo} render_swissclim_config "${template}" "${rendered}"

    ${DRY_RUN:+echo} srun -N1 -n1 -c"${VERIF_CPUS}" --mem "${VERIF_MEM}" \
        "${container_args[@]}" \
        --output="${LOG_DIR}/out_verif_ifs_ens_%j.log" \
        --error="${LOG_DIR}/err_verif_ifs_ens_%j.log" \
        ai-ens verify --config "${rendered}" &
}

# --- Run --------------------------------------------------------------------
run_ai_models ${PERTURBATION_LATENTS}

if [[ "${VERIFY_IFS_ENS:-1}" == "1" && "${IFS_ENS_PATH:-}" != "/path/to/ifs_ens.zarr" ]]; then
    run_ifs_ens
fi
wait
