#!/usr/bin/env bash
# Zonal power spectra of an ablation sweep at one lead time.
# The routine eval runs on a 24 h stride, so +6 h is not in the NPZ bundles and
# has to come from forecast.zarr.
#
# Usage:
#   sbatch tools/submit_psd_lead.sh [lead_hours] [phase ...]   # default: 6 phase1
#   sbatch --partition=normal --time=02:00:00 tools/submit_psd_lead.sh 6 phase2 phase2b
#
# Output: tools/data/psd_<first phase>_lead<LLL>h.npz
#
#SBATCH --account=ab016
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=psd_lead
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/psd_lead_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

LEAD=${1:-6}
shift || true
PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(phase1)

$PY -u tools/compute_psd_lead.py --lead "$LEAD" --phases "${PHASES[@]}" \
    --out "tools/data/psd_${PHASES[0]}_lead$(printf '%03d' "$LEAD")h.npz"
