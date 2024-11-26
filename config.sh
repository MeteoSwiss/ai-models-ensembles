###################DEFINE YOUR ENVIRONMENT VARIABLES HERE###################
export DATE_TIME=201801010000 # Lothar: 199912250000, Burglind: 201801010000
export MODEL_NAME=graphcast # fourcastnetv2-small, graphcast
export LAYER=13 #0-13 for graphcast, 0-11 for fourcastnetv2-small
export PERTURBATION_INIT=0.0
export PERTURBATION_LATENT=0.01
export NUM_MEMBERS=50
export CROP_REGION=europe # Crop to a specific region (europe/global)
export OUTPUT_DIR=/scratch/mch/sadamov/pyprojects_data/ai-models-ensembles

# These paths are used in the pipeline
export SRC_DIR=$PWD
export BASE_DIR=$OUTPUT_DIR
export DATE_DIR="${BASE_DIR}/${DATE_TIME}"
export MODEL_DIR="${DATE_DIR}/${MODEL_NAME}"
export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
export REGION_DIR="${PERTURBATION_DIR}/${CROP_REGION}"
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

##################THIS IS CURRENTLY HARDCODED IN THE EXPERIMENTS##################
# All AI-models produce 10 days of forecasts with 6-hourly intervals
export NUM_DAYS=10
export END_DATE_TIME=$(date -d "${DATE_TIME:0:8} + $((NUM_DAYS)) days" +%Y%m%d)0000
export INTERVAL=6
export LEAD_TIME=240

# if conda env ai-models does not exist then create it
if ! conda env list | grep -q ai_models_ens; then
    mamba env create -n ai_models_ens -f $SRC_DIR/environment.yml
fi
conda activate ai_models_ens
