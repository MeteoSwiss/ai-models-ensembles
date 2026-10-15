#!/usr/bin/env bash
# Standardized ensemble anomalies for the marginal-Gaussianity check
# (tools/compute_ens_anomalies.py). CPU sbatch: reads ~1.4 TB of production zarrs.
#
# Usage:
#   sbatch tools/submit_ens_anomalies.sh                       # all 7 baselines
#   sbatch tools/submit_ens_anomalies.sh aurora_encoder fcn3   # a subset
#
# Output: $STORE/analysis/ens_anom_gaussianity/<baseline>.npz
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ens_anom
#SBATCH --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baseline_eval_logs/ens_anom_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}/tools

BASELINES=("${@:-all}")
$PY -u compute_ens_anomalies.py --baselines "${BASELINES[@]}" --workers 28
