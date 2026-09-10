#!/usr/bin/env bash
# Back-fill the banded-LSD rows the SwissClim eval never produced.
#
# The AIFS Phase-1 unperturbed run has forecast.zarr for all four inits but no
# eval/mag_0_layer_all, so it is missing from the intercomparison CSV. First
# validates the method against an already-evaluated AIFS run, then computes the
# missing one.
#
# Usage: sbatch tools/submit_banded_lsd_gapfill.sh
#
#SBATCH --account=ab016
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=lsd_gapfill
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/lsd_gapfill_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

$PY -u tools/compute_banded_lsd.py --model aifs --run mag_0.01_layer_all --validate
$PY -u tools/compute_banded_lsd.py --model aifs --run mag_0_layer_all
