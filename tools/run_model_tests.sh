#!/usr/bin/env bash
# Test all 5 models using per-model containers.
# Each model runs in its own srun with its own container image.
#
# Usage:
#   bash tools/run_model_tests.sh              # test all 5
#   bash tools/run_model_tests.sh graphcast    # test one model

set -euo pipefail

cd "$(dirname "$0")/.."

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
LOG_DIR=$STORE/model_test_logs
WORKDIR=/workspace/ai-models-ensembles

mkdir -p "$LOG_DIR"

# Model name -> e2s model id mapping
declare -A MODEL_IDS=(
  [aurora]=aurora
  [graphcast]=graphcast_operational
  [sfno]=sfno
  [fcn3]=fcn3
  [atlas]=atlas
  [aifsens]=aifsens
)

run_test() {
    local model=$1
    local model_id=${MODEL_IDS[$model]}
    local container=$STORE/${model}.sqsh

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found (build it first)"
        return 1
    fi

    # Mount CDS/ECMWF API credentials for models that need CDS data (e.g. aifsens)
    local mounts="${PWD}:${WORKDIR},${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    echo "Testing $model (model_id=$model_id)"
    srun -N1 -n1 \
        --partition=debug \
        --account=a122 \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=00:20:00 \
        --job-name="test_${model}" \
        --container-image="$container" \
        --container-mounts="${mounts}" \
        --container-workdir="${WORKDIR}" \
        python tools/test_model.py "$model_id" 2>&1 | tee "$LOG_DIR/${model}.log" &
}

REQUESTED="${1:-all}"

if [[ "$REQUESTED" == "all" ]]; then
    for model in aurora graphcast sfno fcn3 atlas aifsens; do
        run_test "$model" || true
    done
else
    run_test "$REQUESTED"
fi

echo "Waiting for all tests..."
wait

echo ""
echo "=== Test Results ==="
for model in aurora graphcast sfno fcn3 atlas aifsens; do
    log="$LOG_DIR/${model}.log"
    if [[ -f "$log" ]]; then
        status=$(grep -o "Status: [A-Z]*" "$log" | tail -1 | cut -d" " -f2)
        printf "%-25s %s\n" "$model" "${status:-UNKNOWN}"
    fi
done
echo ""
echo "Full logs: $LOG_DIR"
