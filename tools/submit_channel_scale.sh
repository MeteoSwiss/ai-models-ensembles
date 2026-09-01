#!/usr/bin/env bash
# Fixed climatological per-channel scale for ES/VS/SIGK (reviewer item 2):
# removes the truth-dependent standardisation that costs those scores propriety.
#
# Usage: sbatch tools/submit_channel_scale.sh
# Output: tools/data/channel_scale_1990_2019.json
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=channel_scale
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/channel_scale_%j.log

set -euo pipefail

PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
export DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"

cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

# The first attempt (job 3189637) produced no output in 6 h at 36 MB RSS, i.e.
# it blocked before the first print. Stage the startup so a repeat says where.
echo "[$(date +%T)] node=$(hostname)"
$PY -c "print('[interpreter ok]')"
$PY -c "import xarray, numpy; print('[imports ok]', xarray.__version__)"
timeout 300 $PY -c "
import xarray as xr
ds = xr.open_zarr('/capstor/store/cscs/swissai/weatherbench/weatherbench2_original', consolidated=True)
print('[store ok]', ds.sizes.get('time'), 'timesteps')
" || { echo "[store OPEN BLOCKED - weatherbench2_original unreachable]"; exit 1; }

echo "[$(date +%T)] starting scale computation"
$PY -u tools/compute_channel_scale.py --stride-days 7 --max-times 100 --workers 16
