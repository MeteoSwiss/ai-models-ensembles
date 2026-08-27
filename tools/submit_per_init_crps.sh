#!/usr/bin/env bash
# Per-init fair CRPS for every baseline (numerator of the CRPSS tables), one
# scalar per (baseline, init, variable, level, lead). Feeds the paired block
# bootstrap and the common-sample (IFS-ENS-valid) CRPSS recomputation.
#
# Usage:
#   sbatch tools/submit_per_init_crps.sh [lead ...]      # default 24 72 120 240 360
#
# Output: /iopsstor/scratch/cscs/sadamov/per_init_crps_production.csv
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=per_init_crps
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/per_init_crps_%j.log

set -euo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

LEADS=("${@:-}")
[[ -z "${LEADS[*]}" ]] && LEADS=(24 72 120 240 360)

$PY -u tools/compute_per_init_crps.py --baselines all --leads "${LEADS[@]}"
