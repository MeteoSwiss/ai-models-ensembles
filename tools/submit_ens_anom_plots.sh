#!/usr/bin/env bash
# Member-distribution figures in figures/ens_anom/: the large 3-panel values and
# detrended Q-Q figures (tools/plot_ens_anom_gaussianity.py) and the point and
# latitude-band KDE grids (tools/plot_point_ensemble.py). Run after
# tools/submit_ens_anomalies.sh and tools/submit_point_ensemble.sh.
# CPU sbatch: loads the ~0.5-1.5 GB per-baseline npz files; the login node kills it.
#
# Usage: sbatch tools/submit_ens_anom_plots.sh [--lead 240]
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ens_anom_plots
#SBATCH --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baseline_eval_logs/ens_anom_plots_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}/tools

$PY -u plot_ens_anom_gaussianity.py "$@"
$PY -u plot_point_ensemble.py "$@"
