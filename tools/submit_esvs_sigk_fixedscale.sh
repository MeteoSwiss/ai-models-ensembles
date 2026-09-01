#!/usr/bin/env bash
# ES/VS/SIGK for the 4 production winners AND their 6 closest ablation rivals,
# computed BOTH ways: with the per-init truth std (as published) and with the
# fixed 1990-2019 climatological scale (reviewer item 2, propriety). Paired so
# the question "does a proper scaling change the tie-break?" is answerable
# directly from the two CSV sets.
#
# Requires tools/data/channel_scale_1990_2019.json (tools/submit_channel_scale.sh).
#
# Usage: sbatch --dependency=afterok:<channel_scale jobid> tools/submit_esvs_sigk_fixedscale.sh
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=esvs_fixedscale
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/esvs_fixedscale_%j.log
set -uo pipefail
PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20
export TMPDIR=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp DASK_TEMPORARY_DIRECTORY=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/tmp
mkdir -p "$TMPDIR"
cd ${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}

SCALE=tools/data/channel_scale_1990_2019.json
if [[ ! -f "$SCALE" ]]; then
    echo "missing $SCALE -- run tools/submit_channel_scale.sh first"
    exit 1
fi

STORE=${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}
A=$STORE/ablation
WB2A=${AIENS_WB2_22:-/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr}
WB2B=${AIENS_WB2_24:-/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr}
OUT=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/ai-models-ensembles/scratch/table_metrics_fixedscale
mkdir -p "$OUT"
VARS="2m_temperature mean_sea_level_pressure geopotential temperature u_component_of_wind v_component_of_wind specific_humidity"

# label -> ablation run glob. First four are the production winners, the rest
# are the runner-up cells inside the +-0.02 CRPSS band of Tab. calibration.
declare -A RUNS=(
  [win_aurora_enc]="$A/phase2b/aurora/*/mag_0.025_layer_encoder/forecast.zarr"
  [win_graphcast_all]="$A/phase1/graphcast_operational/*/mag_0.01_layer_all/forecast.zarr"
  [win_sfno_modes10]="$A/phase3/sfno/*/mag_0.25_modes10/forecast.zarr"
  [win_aifs_decoder]="$A/phase2/aifs/*/mag_0.027500_layer_decoder/forecast.zarr"
  [riv_aurora_enc_s044]="$A/phase2/aurora/*/mag_0.044176_layer_encoder/forecast.zarr"
  [riv_graphcast_m2g]="$A/phase2/graphcast_operational/*/mag_0.029665_layer_m2g/forecast.zarr"
  [riv_graphcast_g2m]="$A/phase2b/graphcast_operational/*/mag_0.014_layer_g2m/forecast.zarr"
  [riv_sfno_enc_s054]="$A/phase2/sfno/*/mag_0.053852_layer_encoder/forecast.zarr"
  [riv_sfno_enc_s035]="$A/phase2b/sfno/*/mag_0.035_layer_encoder/forecast.zarr"
  [riv_aifs_all_s010]="$A/phase1/aifs/*/mag_0.01_layer_all/forecast.zarr"
)

esvs () { local lbl=$1 L=$2 tag=$3 extra=$4; shift 4
  $PY -u tools/energy_variogram_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --model-label "$lbl" $extra \
    --out-csv "$OUT/esvs_${lbl}_${tag}_L${L}.csv" > "$OUT/esvs_${lbl}_${tag}_L${L}.log" 2>&1; }
sigk () { local lbl=$1 L=$2 tag=$3 extra=$4; shift 4
  $PY -u tools/signature_kernel_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --lead-stride 2 --n-pixels 128 --sigma 1.0 \
    --dyadic 1 --seed 42 --model-label "$lbl" $extra \
    --out-csv "$OUT/sigk_${lbl}_${tag}_L${L}.csv" > "$OUT/sigk_${lbl}_${tag}_L${L}.log" 2>&1; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done; }

for L in 120 240; do
  for lbl in "${!RUNS[@]}"; do
    Z=$(ls -d ${RUNS[$lbl]} 2>/dev/null)
    if [[ -z "$Z" ]]; then echo "MISSING runs for $lbl: ${RUNS[$lbl]}"; continue; fi
    # shellcheck disable=SC2086
    throttle; esvs "$lbl" "$L" truthstd "" $Z &
    # shellcheck disable=SC2086
    throttle; sigk "$lbl" "$L" truthstd "" $Z &
    # shellcheck disable=SC2086
    throttle; esvs "$lbl" "$L" fixed "--scale-json $SCALE" $Z &
    # shellcheck disable=SC2086
    throttle; sigk "$lbl" "$L" fixed "--scale-json $SCALE" $Z &
  done
done
wait
echo "FIXED-SCALE ES/VS/SIGK DONE"; ls -la "$OUT" | tail -20
