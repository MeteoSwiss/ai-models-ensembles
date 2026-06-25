#!/usr/bin/env bash
# Spatial-mean SSR refresh for an IC-augmented baseline on the full 112-init grid.
# Parameterized companion to tools/submit_sfno_p6c_ssr.sh.
#
# The four *_ic runs (aurora_encoder_ic, graphcast_all_ic, sfno_modes10_ic,
# aifs_perturbed_ic) had stale spatial_mean_ssr/spatial_ssr.csv caches written
# 2026-06-13 over only 84 of the 112 inits then on disk. This recomputes the
# Fortin spatial-mean SSR from scratch over all 112 forecast.zarr inits.
#
# CPU-only, host venv, sequential per-(var,level,lead,init) slice loads (low mem,
# heavy zarr I/O).
#
# Usage: sbatch tools/submit_ic_ssr.sh <model_label>
#   e.g. sbatch tools/submit_ic_ssr.sh aurora_encoder_ic
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ic_ssr
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/ic_ssr_%j.log

set -uo pipefail

MODEL="${1:?usage: sbatch tools/submit_ic_ssr.sh <model_label>}"

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
OUT="$STORE/baselines/$MODEL/spatial_mean_ssr/spatial_ssr.csv"

VARS="2m_temperature mean_sea_level_pressure geopotential temperature \
u_component_of_wind v_component_of_wind specific_humidity"

ZARRS=$(ls -d "$STORE"/baselines/"$MODEL"/*/forecast.zarr)
N=$(echo "$ZARRS" | wc -l)
echo "Found $N $MODEL forecast.zarr inits"

# shellcheck disable=SC2086
$PY -u tools/spatial_mean_ssr.py \
    --forecast-zarrs $ZARRS \
    --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS \
    --levels 500 850 \
    --leads 24 72 120 240 \
    --model-label "$MODEL" \
    --out-csv "$OUT"

echo "DONE -> $OUT"
