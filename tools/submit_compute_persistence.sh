#!/usr/bin/env bash
# Submit the 112-init persistence-MAE recompute to Clariden's normal queue.
# WB2 truth source, production 112-init grid, 6 h walltime (was 2 h in the
# first attempt; that TIMEOUT'd before the JSON was written due to buffered
# stdout + heavy WB2 disk I/O). This version forces unbuffered output via
# `python -u` and the script's own sys.stdout.reconfigure(line_buffering=True)
# so progress prints land in the log immediately.
#
# Usage:
#   sbatch tools/submit_compute_persistence.sh
#
# Output:
#   /iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=persistence_112inits
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/persistence_112inits_%j.log

set -euo pipefail

PY=/iopsstor/scratch/cscs/sadamov/venvs/ai-models-ensembles/bin/python
export SSL_CERT_FILE=$($PY -c 'import certifi; print(certifi.where())')
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

$PY -u tools/compute_persistence_mae.py
