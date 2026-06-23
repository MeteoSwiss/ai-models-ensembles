#!/usr/bin/env bash
# Spatial-mean SSR for the 112-init SFNO refresh-every-20 run (sfno_p6c).
# Companion to tools/submit_sfno_p6c_production.sh: once all 112 inits land,
# this computes the single metric Fig. fig:phase6 needs (the field-averaged
# 7-variable SSR per lead) so the SFNO frozen+refresh pair sits on the same
# 112-init production grid as the Aurora/AIFS pairs.
#
# CPU-only, host venv, sequential per-(var,level,lead,init) slice loads (low mem,
# heavy zarr I/O). Output CSV schema matches the existing aurora_encoder /
# aifs_perturbed / *_p6c spatial_ssr.csv files (7 vars x 2 levels x 4 leads).
#
# Usage: sbatch tools/submit_sfno_p6c_ssr.sh
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sfno_p6c_ssr
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/sfno_p6c_ssr_%j.log

set -uo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
OUT="$STORE/baselines/sfno_p6c/spatial_mean_ssr/spatial_ssr.csv"

VARS="2m_temperature mean_sea_level_pressure geopotential temperature \
u_component_of_wind v_component_of_wind specific_humidity"

ZARRS=$(ls -d "$STORE"/baselines/sfno_p6c/*/forecast.zarr)
N=$(echo "$ZARRS" | wc -l)
echo "Found $N sfno_p6c forecast.zarr inits"

# shellcheck disable=SC2086
$PY -u tools/spatial_mean_ssr.py \
    --forecast-zarrs $ZARRS \
    --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS \
    --levels 500 850 \
    --leads 24 72 120 240 \
    --model-label sfno_p6c \
    --out-csv "$OUT"

echo "DONE -> $OUT"
