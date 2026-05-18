#!/usr/bin/env bash
#SBATCH --job-name=graphcast_keys
#SBATCH --partition=debug
#SBATCH --account=a122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles/model_test_logs/graphcast_keys.log
#SBATCH --error=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles/model_test_logs/graphcast_keys.log

set -euo pipefail

STORE=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles
REPO=/users/sadamov/pyprojects/ai-models-ensembles
OUT_HOST=$STORE/model_test_logs
# Inside the container the repo is at /workspace/ai-models-ensembles; STORE is bind-mounted at its real path.
OUT_IN_CONTAINER=$STORE/model_test_logs

mkdir -p "$OUT_HOST"

srun \
  --container-image="$STORE/graphcast.sqsh" \
  --container-mounts="$REPO:/workspace/ai-models-ensembles,$STORE:$STORE" \
  --container-workdir=/workspace/ai-models-ensembles \
  python tools/dump_graphcast_keys.py "$OUT_IN_CONTAINER"
