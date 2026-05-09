#!/usr/bin/env bash
# Submit parallel inference tests for all 5 models.
# Each model runs on its own GPU node.
#
# Usage:
#   bash tools/submit_model_tests.sh

set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "$0")/.."
source ./scripts/config.sh

CONTAINER_IMAGE=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles/ai-ens.sqsh
MODELS=("graphcast_operational" "sfno" "aurora" "fcn3" "atlas")
STORE=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles
OUTPUT_DIR="${STORE}/model_test_logs"

mkdir -p "$OUTPUT_DIR"

echo "Submitting 5 parallel model inference tests..."
echo "Container: $CONTAINER_IMAGE"
echo "Logs: $OUTPUT_DIR"
echo ""

for model in "${MODELS[@]}"; do
    echo "  -> $model"
    srun -N1 -n1 \
        --partition="${INF_PARTITION_SB}" \
        --account="${INF_ACCOUNT_SB}" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=00:30:00 \
        --job-name="test_${model}" \
        --output="${OUTPUT_DIR}/${model}_%j.log" \
        --error="${OUTPUT_DIR}/${model}_%j.log" \
        --container-image="${CONTAINER_IMAGE}" \
        python tools/test_model.py "$model" &
done

echo ""
echo "Waiting for all background jobs to complete..."
wait

echo ""
echo "All jobs complete. Results:"
for model in "${MODELS[@]}"; do
    log_file=$(ls -t "$OUTPUT_DIR"/${model}_*.log 2>/dev/null | head -1)
    if [[ -n "$log_file" ]]; then
        status="PASS"
        grep -q "Status: FAIL" "$log_file" && status="FAIL"
        printf "  %-25s %s\n" "$model" "$status"
    fi
done

echo ""
echo "Full logs in: $OUTPUT_DIR"
