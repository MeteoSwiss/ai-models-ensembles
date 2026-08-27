#!/usr/bin/env bash
# Spread-error relationship (state-dependent uncertainty check, reviewer item 5).
# Streams every production baseline and accumulates a weighted spread histogram
# with the matching ensemble-mean squared error.
#
# Usage:
#   sbatch tools/submit_spread_error.sh [lead ...]     # default 24 72 120 240
#
# Output: /iopsstor/scratch/cscs/sadamov/spread_error_binned.csv
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=spread_error
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/spread_error_%j.log

set -euo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

LEADS=("${@:-}")
[[ -z "${LEADS[*]}" ]] && LEADS=(24 72 120 240)

$PY -u tools/spread_error_binned.py --baselines all --leads "${LEADS[@]}"
