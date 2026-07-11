#!/usr/bin/env bash
# Spatial-mean SSR for a RESEED refresh grid (fig:phase6 number after the
# 2026-07-10 fresh-hook seeding fix). Mirrors tools/submit_sfno_p6c_ssr.sh but
# parameterised by model, pointing at $STORE/baselines/<model>_p6c_reseed.
# CPU-only sbatch (heavy zarr I/O, low mem). Usage: sbatch tools/submit_p6c_reseed_ssr.sh <sfno|aurora|aifs>
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=p6reseed_ssr
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/tmp/p6reseed_ssr_%j.log
set -uo pipefail
MODEL="${1:?usage: sbatch submit_p6c_reseed_ssr.sh <sfno|aurora|aifs>}"
PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"
cd /users/sadamov/pyprojects/ai-models-ensembles
STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
VARS="2m_temperature mean_sea_level_pressure geopotential temperature u_component_of_wind v_component_of_wind specific_humidity"
GRID="$STORE/baselines/${MODEL}_p6c_reseed"
OUT="$GRID/spatial_mean_ssr/spatial_ssr.csv"
ZARRS=$(ls -d "$GRID"/*/forecast.zarr 2>/dev/null)
N=$(echo "$ZARRS" | grep -c forecast.zarr || true)
echo "Found $N ${MODEL}_p6c_reseed forecast.zarr inits"
[[ "$N" -lt 1 ]] && { echo "no zarrs, aborting"; exit 1; }
# shellcheck disable=SC2086
$PY -u tools/spatial_mean_ssr.py \
    --forecast-zarrs $ZARRS \
    --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --leads 24 72 120 240 \
    --model-label "${MODEL}_p6c_reseed" --out-csv "$OUT"
echo "DONE -> $OUT"
