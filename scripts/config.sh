################### DEFINE YOUR ENVIRONMENT VARIABLES HERE ###################
set -euo pipefail
IFS=$'\n\t'

# ---- Run-level inputs -------------------------------------------------------
export LOG_DIR=${LOG_DIR:-logs}
export DRY_RUN=${DRY_RUN:-}
export DATE_TIME=201801010000             # YYYYMMDDHHMM (Burglind: 201801010000)
export MODEL_NAME=graphcast_operational   # see `ai-ens models` for the full list
export NUM_MEMBERS=10                     # ensemble members per perturbation
                                          # (matched by IFS ENS subsample in
                                          # swissclim_ifs_ens.yaml.template)
export CROP_REGION=europe                 # europe | global (used by SwissClim YAML)
export OUTPUT_DIR=$STORE/sadamov/ai-models-ensembles
export LEAD_TIME=$((14 * 24))             # 336 hours = 2 weeks; must be a multiple
                                          # of the model step.
export OUTPUT_LEVELS=${OUTPUT_LEVELS:-"100,500,850"}  # pressure levels (hPa) kept
                                                      # in forecast.zarr; 'all' to keep every level

# ---- IFS reference data (on-disk SwissClim-format zarr) --------------------
# IFS ENS forecast (50 members, full lead time). Acts as the physical-model
# probabilistic baseline fed directly to SwissClim verification.
export IFS_ENS_PATH=${IFS_ENS_PATH:-/path/to/ifs_ens.zarr}

# ---- ERA5 verification target ----------------------------------------------
# SwissClim verification compares predictions against this zarr.
export TARGET_PATH=${TARGET_PATH:-/path/to/era5.zarr}

# ---- earth2studio data source ----------------------------------------------
# arco | cds | gfs | ifs | ifs_ens | wb2 | file:/path | ifs_analysis:/path
# Default: ARCO ERA5. All five AI models were trained on ERA5; ARCO has
# every variable and level any of them needs. The IFS ENS forecast on disk
# is plugged in as a physical-model baseline via the swissclim_ifs_ens
# template, not as an IC source.
export DATA_SOURCE=${DATA_SOURCE:-arco}

# ---- Perturbations ----------------------------------------------------------
export PERTURBATION_INIT=0.0              # multiplicative IC noise sigma (0 disables)
export PERTURBATION_LATENT=0.01           # multiplicative weight noise sigma (0 disables)
export LAYER=                             # int weight-tensor index, or empty for all
export PERTURBATION_LATENTS=${PERTURBATION_LATENTS:-"0.0 0.002 0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1"}

# ---- Container / SwissClim --------------------------------------------------
# Path to the .sqsh image produced by `containers/build.sh`. Empty = run on
# the host environment (uv-managed .venv).
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
# Optional: explicit SwissClim Evaluations YAML; if unset, submit_verification.sh
# renders config/swissclim_eval.yaml.template.
export SWISSCLIM_CONFIG_TEMPLATE=${SWISSCLIM_CONFIG_TEMPLATE:-config/swissclim_eval.yaml.template}
export SWISSCLIM_CONFIG=${SWISSCLIM_CONFIG:-}

# ---- Derived paths ----------------------------------------------------------
export SRC_DIR=$PWD
export BASE_DIR=$OUTPUT_DIR
export DATE_DIR="${BASE_DIR}/${DATE_TIME}"
export MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"

# ---- Slurm resource knobs ---------------------------------------------------
export INF_CPUS=${INF_CPUS:-32}
export INF_MEM=${INF_MEM:-96G}
export INF_GPUS=${INF_GPUS:-1}
export VERIF_CPUS=${VERIF_CPUS:-128}
export VERIF_MEM=${VERIF_MEM:-222G}

# Inference SBATCH defaults
export INF_JOB_NAME=${INF_JOB_NAME:-ai_inf}
export INF_NODES_SB=${INF_NODES_SB:-4}
export INF_NTASKS_SB=${INF_NTASKS_SB:-15}
export INF_CPUS_PER_TASK_SB=${INF_CPUS_PER_TASK_SB:-32}
export INF_MEM_PER_CPU_SB=${INF_MEM_PER_CPU_SB:-3G}
export INF_PARTITION_SB=${INF_PARTITION_SB:-normal}
export INF_GRES_SB=${INF_GRES_SB:-gpu:4}
export INF_ACCOUNT_SB=${INF_ACCOUNT_SB:-a122}
export INF_TIME_SB=${INF_TIME_SB:-18:00:00}
export INF_NO_REQUEUE=${INF_NO_REQUEUE:-1}

# Verification SBATCH defaults
export VERIF_JOB_NAME=${VERIF_JOB_NAME:-ai_verif}
export VERIF_NODES_SB=${VERIF_NODES_SB:-2}
export VERIF_NTASKS_SB=${VERIF_NTASKS_SB:-2}
export VERIF_PARTITION_SB=${VERIF_PARTITION_SB:-pp-long}
export VERIF_ACCOUNT_SB=${VERIF_ACCOUNT_SB:-a122}
export VERIF_TIME_SB=${VERIF_TIME_SB:-5-00:00:00}
export VERIF_NO_REQUEUE=${VERIF_NO_REQUEUE:-1}
export VERIF_EXCLUSIVE=${VERIF_EXCLUSIVE:-1}

# ---- Helpers ----------------------------------------------------------------
create_dir_if_not_exists() {
    if ! test -d "$1"; then
        echo "Creating directory $1"
        mkdir -p "$1"
    fi
}
export -f create_dir_if_not_exists

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}
export -f require_cmd

# Activate the host venv if present and no container is in use.
if [[ -z "${CONTAINER_IMAGE:-}" && -d "$SRC_DIR/.venv" ]]; then
    # shellcheck disable=SC1091
    source "$SRC_DIR/.venv/bin/activate"
fi
