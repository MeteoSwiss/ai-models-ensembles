#!/usr/bin/env bash
# Submit resharding job to convert existing zarr stores to sharded zarr v3.
# Also cleans up leftover _e2s_work directories.
#
# Usage:
#   bash tools/submit_reshard.sh                 # reshard everything
#   bash tools/submit_reshard.sh ablation        # ablation only
#   bash tools/submit_reshard.sh production      # production only
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR=/workspace/ai-models-ensembles
CONTAINER="$STORE/aurora.sqsh"  # any container with zarr+xarray works
PARTITION="${PARTITION:-normal}"

TARGET="${1:-all}"

case "$TARGET" in
    ablation)   PATHS="$STORE/ablation" ;;
    production) PATHS="$STORE/201801010000" ;;
    all)        PATHS="$STORE/ablation $STORE/201801010000" ;;
    *)          echo "Usage: $0 [ablation|production|all]"; exit 1 ;;
esac

MOUNTS="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"

sbatch --parsable \
    --account=a122 \
    --partition="$PARTITION" \
    --job-name="reshard_${TARGET}" \
    --time=12:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=256G \
    --output="$STORE/logs/reshard_%j.out" \
    --error="$STORE/logs/reshard_%j.err" \
    --container-image="$CONTAINER" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$WORKDIR" \
    --wrap="
set -euo pipefail

echo '=== Cleaning up _e2s_work directories ==='
find $STORE -type d -name '_e2s_work' -exec rm -rf {} + 2>/dev/null || true
find $STORE -type d -name '_seq_members' -exec rm -rf {} + 2>/dev/null || true
find $STORE -type d -name '_par_members' -exec rm -rf {} + 2>/dev/null || true
echo 'Cleanup done.'

echo '=== Resharding zarr stores ==='
for p in $PATHS; do
    python tools/reshard_zarr.py \"\$p\"
done
echo '=== All done ==='
"
