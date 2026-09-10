#!/usr/bin/env bash
# Ensemble-pooled banded LSD for every Phase-1 run, all four backbones.
#
# The SwissClim eval's own "enspooled" spectra are member-0 only: the target
# carries a singleton ensemble dim and _compute_spectra_pair's
# xr.align(..., join="inner") intersects the ensemble coordinate down to {0}.
# This recomputes them as a true mean of per-member spectra.
#
# Usage:
#   sbatch tools/submit_banded_lsd_sweep.sh a      # mass/thermodynamic group
#   sbatch tools/submit_banded_lsd_sweep.sh b      # wind group
#
#SBATCH --account=ab016
#SBATCH --partition=debug
#SBATCH --time=01:25:00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=lsd_sweep
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/lsd_sweep_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=$TMPDIR
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

GROUP=${1:-a}
case "$GROUP" in
  a) VARS=(geopotential@500 geopotential@850 temperature@500 temperature@850
           specific_humidity@500 specific_humidity@850 2m_temperature) ;;
  b) VARS=(u_component_of_wind@500 u_component_of_wind@850 v_component_of_wind@500
           v_component_of_wind@850 mean_sea_level_pressure 10m_u_component_of_wind
           10m_v_component_of_wind) ;;
  *) echo "unknown group '$GROUP' (expected a or b)" >&2; exit 1 ;;
esac

$PY -u tools/compute_banded_lsd.py --all --variables "${VARS[@]}" \
    --out "tools/data/lsd_bands_phase1_pooled_${GROUP}.csv"
