################### DEFINE YOUR ENVIRONMENT VARIABLES HERE ###################
set -euo pipefail
IFS=$'\n\t'

export EARTHKIT_CACHE_DIR=$SCRATCH/earthkit_cache

# Logs
export LOG_DIR=${LOG_DIR:-logs}
export DRY_RUN=${DRY_RUN:-}
export DATE_TIME=201801010000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=graphcast # fourcastnetv2-small, graphcast
export LAYER=13 #0-13 for graphcast, 0-11 for fourcastnetv2-small
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.01
export NUM_MEMBERS=50
export CROP_REGION=europe # Crop to a specific region (europe/global)
export OUTPUT_DIR=$STORE/sadamov/ai-models-ensembles

# These paths are used in the pipeline
export SRC_DIR=$PWD
export BASE_DIR=$OUTPUT_DIR
export DATE_DIR="${BASE_DIR}/${DATE_TIME}"
export MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"

# Optional sweep values (space-separated) for perturbation latents
export PERTURBATION_LATENTS=${PERTURBATION_LATENTS:-"0.0 0.002 0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1"}

# ecCodes library setup (required by earthkit/eccodes)
# Try to auto-detect an existing ecCodes installation (e.g., in Miniforge/Conda 'apps' env)
if [ -z "${ECCODES_DIR:-}" ]; then
    for CAND in \
        "$HOME/miniforge3/envs/apps" \
        "$HOME/miniconda3/envs/apps" \
        "/users/$USER/miniforge3/envs/apps" \
        "/users/$USER/miniconda3/envs/apps"; do
        if [ -f "$CAND/lib/libeccodes.so" ]; then
            export ECCODES_DIR="$CAND"
            break
        fi
    done
fi

if [ -n "${ECCODES_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${ECCODES_DIR}/lib:${LD_LIBRARY_PATH:-}"
    if [ -d "${ECCODES_DIR}/share/eccodes/definitions" ]; then
        export ECCODES_DEFINITION_PATH="${ECCODES_DIR}/share/eccodes/definitions"
    fi
    if [ -d "${ECCODES_DIR}/share/eccodes/samples" ]; then
        export ECCODES_SAMPLES_PATH="${ECCODES_DIR}/share/eccodes/samples"
    fi
fi

# Slurm resource knobs (override per cluster)
export ZARR_CPUS=${ZARR_CPUS:-32}
export ZARR_MEM=${ZARR_MEM:-64G}
export INF_CPUS=${INF_CPUS:-32}
export INF_MEM=${INF_MEM:-96G}
export INF_GPUS=${INF_GPUS:-1}
export VERIF_CPUS=${VERIF_CPUS:-128}
export VERIF_MEM=${VERIF_MEM:-222G}

# SBATCH defaults per stage (centralized)
# Download stage
export DL_JOB_NAME=${DL_JOB_NAME:-ai_dl}
export DL_NODES=${DL_NODES:-1}
export DL_NTASKS=${DL_NTASKS:-1}
export DL_CPUS_PER_TASK=${DL_CPUS_PER_TASK:-8}
export DL_MEM_PER_CPU=${DL_MEM_PER_CPU:-8G}
export DL_PARTITION=${DL_PARTITION:-pp-long}
export DL_ACCOUNT=${DL_ACCOUNT:-s83}
export DL_TIME=${DL_TIME:-5-00:00:00}
export DL_NO_REQUEUE=${DL_NO_REQUEUE:-1}

# Inference stage
export INF_JOB_NAME=${INF_JOB_NAME:-ai_inf}
export INF_NODES_SB=${INF_NODES_SB:-4}
export INF_NTASKS_SB=${INF_NTASKS_SB:-15}
export INF_CPUS_PER_TASK_SB=${INF_CPUS_PER_TASK_SB:-32}
export INF_MEM_PER_CPU_SB=${INF_MEM_PER_CPU_SB:-3G}
export INF_PARTITION_SB=${INF_PARTITION_SB:-normal}
export INF_GRES_SB=${INF_GRES_SB:-gpu:4}
export INF_ACCOUNT_SB=${INF_ACCOUNT_SB:-s83}
export INF_TIME_SB=${INF_TIME_SB:-18:00:00}
export INF_NO_REQUEUE=${INF_NO_REQUEUE:-1}

# Zarr conversion stage
export ZARR_JOB_NAME=${ZARR_JOB_NAME:-ai_zarr}
export ZARR_NODES_SB=${ZARR_NODES_SB:-3}
export ZARR_NTASKS_SB=${ZARR_NTASKS_SB:-15}
export ZARR_CPUS_PER_TASK_SB=${ZARR_CPUS_PER_TASK_SB:-32}
export ZARR_MEM_PER_CPU_SB=${ZARR_MEM_PER_CPU_SB:-2G}
export ZARR_PARTITION_SB=${ZARR_PARTITION_SB:-postproc}
export ZARR_ACCOUNT_SB=${ZARR_ACCOUNT_SB:-s83}
export ZARR_TIME_SB=${ZARR_TIME_SB:-24:00:00}
export ZARR_NO_REQUEUE=${ZARR_NO_REQUEUE:-1}

# Verification stage
export VERIF_JOB_NAME=${VERIF_JOB_NAME:-ai_verif}
export VERIF_NODES_SB=${VERIF_NODES_SB:-2}
export VERIF_NTASKS_SB=${VERIF_NTASKS_SB:-2}
export VERIF_PARTITION_SB=${VERIF_PARTITION_SB:-pp-long}
export VERIF_ACCOUNT_SB=${VERIF_ACCOUNT_SB:-s83}
export VERIF_TIME_SB=${VERIF_TIME_SB:-5-00:00:00}
export VERIF_NO_REQUEUE=${VERIF_NO_REQUEUE:-1}
export VERIF_EXCLUSIVE=${VERIF_EXCLUSIVE:-1}
# These functions are used in the pipeline
create_dir_if_not_exists() {
    if ! test -d "$1"; then
        echo "Creating directory $1"
        mkdir -p "$1"
    fi
}
export -f create_dir_if_not_exists
proceed_if_not_exists() {
    if ! test -f "$1"; then
        echo "Executing command: $2"
        eval "$2"
    fi
}
export -f proceed_if_not_exists

# Utilities
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}
export -f require_cmd

##################THIS IS CURRENTLY HARDCODED IN THE EXPERIMENTS##################
# All AI-models produce 10 days of forecasts with 6-hourly intervals
export NUM_DAYS=10
export END_DATE_TIME=$(date -d "${DATE_TIME:0:8} + $((NUM_DAYS)) days" +%Y%m%d)0000
export INTERVAL=6
export LEAD_TIME=240

# Activate local virtual environment if present (uv + venv)
if [[ -d "$SRC_DIR/.venv" ]]; then
    # shellcheck disable=SC1091
    source "$SRC_DIR/.venv/bin/activate"
else
    echo "Warning: .venv not found in $SRC_DIR. Run scripts/setup_uv.sh to create the environment." >&2
fi
