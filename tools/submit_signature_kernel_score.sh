#!/usr/bin/env bash
# Signature Kernel Score over the lead-time trajectory for all 7 production
# baselines (the same set as Tab. esvs; IFS-ENS omitted for the same NaN-gap
# reason). One CPU node, the 7 baselines run in parallel, each writing a
# per-baseline CSV; tools/aggregate_signature_kernel_score.py then builds the
# combined table.
#
# Usage:
#   sbatch tools/submit_signature_kernel_score.sh
#
# Outputs in /iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/sigk/:
#   <baseline>.csv  (one row each) + sigk_combined.csv after aggregation.
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sigk_score
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/sigk_score_%j.log

set -euo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
# cap per-process threads so 7 parallel baselines (7x8=56) stay under 64 cpus
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
OUT=/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/sigk
mkdir -p "$OUT"

VARS="2m_temperature mean_sea_level_pressure geopotential temperature \
u_component_of_wind v_component_of_wind specific_humidity"
BASELINES=(aifsens atlas aifs_perturbed fcn3 graphcast_all aurora_encoder sfno_modes10)

pids=()
for b in "${BASELINES[@]}"; do
    ZARRS=$(ls -d "$STORE"/baselines/"$b"/*/forecast.zarr)
    # shellcheck disable=SC2086
    $PY -u tools/signature_kernel_score.py \
        --forecast-zarrs $ZARRS \
        --truth-zarrs "$WB2A" "$WB2B" \
        --variables $VARS \
        --levels 500 850 \
        --lead 240 --lead-stride 2 --n-pixels 128 \
        --sigma 1.0 --dyadic 1 --seed 42 \
        --model-label "$b" \
        --out-csv "$OUT/$b.csv" \
        > "$OUT/$b.log" 2>&1 &
    pids+=($!)
    echo "launched $b (pid $!)"
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "OK  ${BASELINES[$i]}"
    else
        echo "FAIL ${BASELINES[$i]}"
        rc=1
    fi
done

$PY -u tools/aggregate_signature_kernel_score.py \
    --in-dir "$OUT" --out-csv "$OUT/sigk_combined.csv"

exit $rc
