#!/usr/bin/env bash
# Exact member-0 bug impact on the paper's delta spectrogram, on a lead subset.
#
# Usage: sbatch tools/submit_spectrogram_bug_impact.sh [lead_stride]
#
#SBATCH --account=ab016
#SBATCH --partition=debug
#SBATCH --time=01:25:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=spec_impact
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/spec_impact_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

$PY -u tools/compute_spectrogram_bug_impact.py --lead-stride "${1:-4}" --workers 12
