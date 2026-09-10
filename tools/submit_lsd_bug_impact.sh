#!/usr/bin/env bash
# Exact impact of the member-0 align bug on the paper's LSD column.
# 7 production baselines x 5 3D variables x 2 levels x leads 120/240 x 112 inits.
#
# Usage: sbatch tools/submit_lsd_bug_impact.sh
#
#SBATCH --account=ab016
#SBATCH --partition=debug
#SBATCH --time=01:25:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=lsd_impact
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/lsd_impact_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

$PY -u tools/compute_lsd_bug_impact.py --workers 12 --out tools/data/lsd_bug_impact.csv
