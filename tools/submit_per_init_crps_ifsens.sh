#!/usr/bin/env bash
# Per-init CRPS for IFS-ENS only, appended to the existing production CSV.
# IFS-ENS lives in one consolidated zarr (init_time dim, 50 members) rather than
# per-init directories, so it was silently skipped by the main pass; this fills
# it in and supplies the valid-case mask the common-sample CRPSS needs.
#
# Usage: sbatch tools/submit_per_init_crps_ifsens.sh
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=per_init_ifsens
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/per_init_ifsens_%j.log

set -euo pipefail
PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"
cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

OUT=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/per_init_crps_ifsens.csv
MAIN=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/per_init_crps_production.csv

$PY -u tools/compute_per_init_crps.py --baselines ifs_ens --leads 24 72 120 240 360 --out "$OUT"

# Append to the main CSV (header dropped) so the common-sample tool sees one file.
tail -n +2 "$OUT" >> "$MAIN"
echo "merged $(($(wc -l < "$OUT") - 1)) IFS-ENS rows into $MAIN"
