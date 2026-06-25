#!/usr/bin/env bash
# Per-pixel (Fortin) SSR refresh for one baseline on the full 112-init grid.
# Companion to tools/submit_ic_ssr.sh (spatial-mean SSR). Refreshes the
# ssr_line_<var>_by_lead per-pixel SSR to the current Fortin vintage over all
# 112 inits, for both the weight-only production baselines and their _ic
# variants, so the paper's "per-pixel SSR rises from X to Y" / "+gain across
# four backbones" numbers come from one consistent 112-init computation.
#
# CPU-only, host venv. Heavy zarr I/O (112 x 19 GB forecast.zarr) -> sbatch,
# never login node.
#
# Usage: sbatch tools/submit_perpixel_ssr.sh <model_label>
#   e.g. sbatch tools/submit_perpixel_ssr.sh aifs_perturbed_ic
#        sbatch tools/submit_perpixel_ssr.sh aifs_perturbed
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=pp_ssr
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/pp_ssr_%j.log

set -uo pipefail

MODEL="${1:?usage: sbatch tools/submit_perpixel_ssr.sh <model_label>}"

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
OUT="$STORE/baselines/$MODEL/per_pixel_ssr/per_pixel_ssr.csv"
mkdir -p "$(dirname "$OUT")"

# VARS / LEADS overridable via env. Default = full 7-var set; the paper only
# cites MSL per-pixel SSR @24h, so VARS="mean_sea_level_pressure 2m_temperature"
# runs ~14x fewer slab reads (minutes vs hours).
VARS="${VARS:-2m_temperature mean_sea_level_pressure geopotential temperature \
u_component_of_wind v_component_of_wind specific_humidity}"
LEADS="${LEADS:-6 24 72 120 240}"

ZARRS=$(ls -d "$STORE"/baselines/"$MODEL"/*/forecast.zarr)
N=$(echo "$ZARRS" | wc -l)
echo "Found $N $MODEL forecast.zarr inits; VARS=[$VARS] LEADS=[$LEADS]"

# shellcheck disable=SC2086
$PY -u tools/per_pixel_ssr.py \
    --forecast-zarrs $ZARRS \
    --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS \
    --levels 500 850 \
    --leads $LEADS \
    --model-label "$MODEL" \
    --out-csv "$OUT"

echo "DONE -> $OUT"
