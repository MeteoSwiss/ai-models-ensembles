#!/usr/bin/env bash
# Exact-WB2 probabilistic-climatology CRPS denominator (30 members = years
# 1990-2019, fair CRPS vs the actual 2023/2024 eval truth at every valid time).
# Replaces the analytical-Gaussian CRPS_clim = sigma/sqrt(pi) workaround.
#
# Memory: ~43 GB clim ensemble + ~3 GB truth per field; --parallel 2 -> ~100 GB
# peak. Lighter than the full-clim LOO job, so 12 h on normal is ample.
#
# Usage:
#   sbatch tools/submit_compute_climatology_vs_eval.sh [production|ablation]
# (default: production -- the 112-init headline grid; ablation = 4 mid-season
#  inits at 240 h for the calibration table)
#
# Outputs in /iopsstor/scratch/cscs/sadamov/:
#   crps_clim_eval_<tag>.json
#   crps_clim_eval_<tag>_provenance.json
# plus a versioned copy at tools/data/crps_clim_eval_<tag>.json
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=crps_clim_eval
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/crps_clim_eval_%j.log

set -euo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export UV_CACHE_DIR=/capstor/scratch/cscs/sadamov/uv_cache
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

GRID="${1:-production}"
$PY -u tools/compute_climatology_crps_vs_eval.py \
    --grid "$GRID" \
    --workers 16 \
    --parallel 2
