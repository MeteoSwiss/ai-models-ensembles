#!/usr/bin/env bash
#SBATCH --job-name=sfno_keys
#SBATCH --partition=debug
#SBATCH --account=a122
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/model_test_logs/sfno_keys.log
#SBATCH --error=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/model_test_logs/sfno_keys.log

set -euo pipefail

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
REPO=/users/sadamov/pyprojects/ai-models-ensembles

srun -N1 -n1 \
  --container-image="$STORE/sfno.sqsh" \
  --container-mounts="$REPO:/workspace/ai-models-ensembles,$STORE:$STORE" \
  --container-workdir=/workspace/ai-models-ensembles \
  python tools/dump_sfno_keys.py
