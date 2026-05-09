#!/usr/bin/env bash
# Submit inference jobs for all 6 models via sbatch.
# Each model gets its own job with 4 GPUs, 444G memory, proper log capture.
#
# Usage:
#   bash scripts/submit_all_inference.sh              # all 6 models
#   bash scripts/submit_all_inference.sh aurora sfno   # specific models
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/logs"
WORKDIR=/workspace/ai-models-ensembles

INIT_TIME="2018-01-01T00:00"
LEAD_HOURS=336
NUM_MEMBERS=10
OUTPUT_LEVELS="100,500,850"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="06:00:00"

mkdir -p "$LOG_DIR"

# Model -> (model_id, container_name, weight_magnitude, data_source)
declare -A MODEL_IDS=( [aurora]=aurora [graphcast]=graphcast_operational [sfno]=sfno [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens )
declare -A WEIGHT_MAG=( [aurora]=0.01 [graphcast]=0.01 [sfno]=0.01 [fcn3]=0.0 [atlas]=0.0 [aifsens]=0.0 )
declare -A DATA_SRC=( [aurora]=arco [graphcast]=arco [sfno]=arco [fcn3]=arco [atlas]=arco [aifsens]=cds )
ALL_MODELS="aurora graphcast sfno fcn3 atlas aifsens"
REQUESTED="${@:-$ALL_MODELS}"

for model in $REQUESTED; do
    model_id="${MODEL_IDS[$model]}"
    container="$STORE/${model}.sqsh"
    wmag="${WEIGHT_MAG[$model]}"
    dsrc="${DATA_SRC[$model]}"

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        continue
    fi

    # Output path
    out_dir="$STORE/201801010000/${model_id}/init_0.0_latent_${wmag}"
    out_zarr="$out_dir/forecast.zarr"

    # Clean up old partial outputs
    if [[ -d "$out_dir" ]]; then
        echo "Cleaning old output: $out_dir"
        rm -rf "$out_dir"
    fi
    mkdir -p "$out_dir"

    # Container mounts
    mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    echo "Submitting $model (model_id=$model_id, weight_mag=$wmag, data=$dsrc)"

    sbatch --parsable \
        --job-name="inf_${model}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --mem=444G \
        --gres=gpu:4 \
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/inf_${model}_%j.out" \
        --error="$LOG_DIR/inf_${model}_%j.err" \
        --container-image="$container" \
        --container-mounts="$mounts" \
        --container-workdir="$WORKDIR" \
        --wrap="python -m ai_models_ensembles.cli infer \
            --model $model_id \
            --init '$INIT_TIME' \
            --lead-hours $LEAD_HOURS \
            --members $NUM_MEMBERS \
            --weight-magnitude $wmag \
            --data-source $dsrc \
            --output-levels '$OUTPUT_LEVELS' \
            --output '$out_zarr'"
done

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOG_DIR/inf_*.{out,err}"
