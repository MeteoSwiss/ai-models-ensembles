#!/usr/bin/env bash
# Raw member values at one gridpoint, all 112 inits (tools/compute_point_ensemble.py).
# CPU sbatch: every chunk read is a full global field (~0.5 TB in total).
#
# Usage:
#   sbatch tools/submit_point_ensemble.sh                                  # all 8 default points
#   sbatch tools/submit_point_ensemble.sh --points eq,0,180 nya,79,12      # other points, one pass
#
# Output: $STORE/analysis/ens_anom_gaussianity/point_<name>.npz
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=point_ens
#SBATCH --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baseline_eval_logs/point_ens_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}/tools

$PY -u compute_point_ensemble.py "$@"
