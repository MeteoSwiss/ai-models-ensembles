#!/usr/bin/env bash
# IC-versus-weight spread attribution (Fuhrer review item 1): scores the
# weight-only / IC-only / weight+IC arms of each backbone on strictly paired
# initialisations. Aborts unless every arm has the full 112 (--require-inits).
#
# Usage: sbatch tools/submit_ic_weight_decomposition.sh
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ic_decomp
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/ic_decomp_%j.log

set -euo pipefail
PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"
cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

$PY -u tools/ic_weight_decomposition.py --backbones all \
    --leads 6 24 72 120 240 --workers 12 "$@"
