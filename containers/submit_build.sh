#!/usr/bin/env bash
# Submit per-model container build(s) to a compute node via Slurm.
#
# Usage:
#   bash containers/submit_build.sh graphcast           # build one model (debug)
#   bash containers/submit_build.sh graphcast normal     # build one model (normal)
#   bash containers/submit_build.sh all                  # build all 5 models in parallel
#   bash containers/submit_build.sh all normal           # build all 5 on normal partition

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: submit_build.sh <model|all> [partition]}"
PARTITION="${2:-debug}"
ACCOUNT="${ACCOUNT:-a122}"
STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

ALL_MODELS=(aurora graphcast sfno fcn3 atlas aifsens)

submit_one() {
    local model=$1
    echo "Submitting build: $model (partition=$PARTITION)"
    OUTPUT="${STORE}/${model}.sqsh" srun -N1 -n1 \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --cpus-per-task=32 \
        --mem=444G \
        --time=01:00:00 \
        --job-name="build_${model}" \
        --output="${LOG_DIR}/build_${model}_%j.log" \
        --error="${LOG_DIR}/build_${model}_%j.log" \
        bash containers/build.sh "$model" &
}

if [[ "$MODEL" == "all" ]]; then
    for m in "${ALL_MODELS[@]}"; do
        submit_one "$m"
    done
    echo "Waiting for all builds..."
    wait
    echo "All builds complete. Check logs: ${LOG_DIR}/build_*.log"
else
    submit_one "$MODEL"
    wait
    echo "Done. Check logs: ${LOG_DIR}/build_${MODEL}_*.log"
    echo "Container: ${STORE}/${MODEL}.sqsh"
fi
