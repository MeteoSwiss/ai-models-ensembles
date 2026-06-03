#!/usr/bin/env bash
# Submit the 30-year WB2 climatology recompute to Clariden's normal queue.
# 12h walltime, single CPU node, 800G RAM (per project rule).  The script
# itself drives HTTPS reads with a thread-pool so a single task is enough.
#
# Usage:
#   sbatch tools/submit_compute_climatology.sh
#
# Outputs land in /iopsstor/scratch/cscs/sadamov/:
#   sigma_clim_1990_2019.json
#   empirical_crps_clim_1990_2019.json
#   climatology_1990_2019_provenance.json
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=clim1990
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/clim1990_%j.log

set -euo pipefail

PY=/iopsstor/scratch/cscs/sadamov/venvs/ai-models-ensembles/bin/python
export SSL_CERT_FILE=$($PY -c 'import certifi; print(certifi.where())')
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export UV_CACHE_DIR=/capstor/scratch/cscs/sadamov/uv_cache
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

$PY tools/compute_climatology_1990_2019.py \
    --year-start 1990 --year-end 2019 \
    --hours 0 6 12 18 \
    --workers 48
