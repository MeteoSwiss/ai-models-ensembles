#!/usr/bin/env bash
# Submit the E2E debug script to a GPU compute node via Slurm.
#
# Usage:
#   bash tools/submit_debug.sh                      # default: container mode
#   bash tools/submit_debug.sh --with-infer          # also run real 1-step inference
#   CONTAINER_IMAGE=/path/to/ai-ens.sqsh bash tools/submit_debug.sh
#
# Without CONTAINER_IMAGE, runs directly on the node (requires deps in PATH).
# Logs go to logs/debug_e2e_<jobid>.log.

set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "$0")/.."
source ./scripts/config.sh

DEBUG_ARGS="${*:---skip-gpu}"  # default to --skip-gpu if nothing passed
# But if anything was passed, use that instead
if [[ $# -gt 0 ]]; then
    DEBUG_ARGS="$*"
fi

mkdir -p "$LOG_DIR"

CONTAINER_ARGS=()
if [[ -n "${CONTAINER_IMAGE:-}" && -f "${CONTAINER_IMAGE}" ]]; then
    CONTAINER_ARGS=(
        "--container-image=${CONTAINER_IMAGE}"
        "--container-mounts=${PWD}:/workspace/ai-models-ensembles"
    )
    echo "Using container: ${CONTAINER_IMAGE}"
    echo "  mounting $PWD -> /workspace/ai-models-ensembles"
else
    echo "No CONTAINER_IMAGE set or file not found. Running on host env."
fi

echo "Submitting debug job..."
echo "  args: python tools/debug_e2e.py ${DEBUG_ARGS}"

srun -N1 -n1 \
    --partition="${INF_PARTITION_SB}" \
    --account="${INF_ACCOUNT_SB}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=00:30:00 \
    --job-name=ai_debug \
    --output="${LOG_DIR}/debug_e2e_%j.log" \
    --error="${LOG_DIR}/debug_e2e_%j.log" \
    "${CONTAINER_ARGS[@]}" \
    python tools/debug_e2e.py ${DEBUG_ARGS}

echo "Done. Check logs: ${LOG_DIR}/debug_e2e_*.log"
