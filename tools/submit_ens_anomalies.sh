#!/usr/bin/env bash
# Standardized ensemble anomalies for the marginal-Gaussianity check
# (tools/compute_ens_anomalies.py). CPU sbatch: reads ~1.4 TB of production zarrs.
#
# Usage:
#   sbatch tools/submit_ens_anomalies.sh                                   # all 7 baselines
#   sbatch tools/submit_ens_anomalies.sh --baselines aurora_encoder fcn3   # a subset
#   sbatch tools/submit_ens_anomalies.sh --baselines all era5_clim ifs_ens --init-stride 1 \
#       --n-points 8192 --vars 2m_temperature specific_humidity_850 geopotential_500 \
#       --tag sw --p-only                       # Shapiro-Wilk sample (member_shapiro_rate)
#
# Output: $STORE/analysis/ens_anom_gaussianity/<baseline>[_<tag>].npz
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

if [[ $# -eq 0 ]]; then set -- --baselines all; fi
$PY -u compute_ens_anomalies.py --workers 28 "$@"
