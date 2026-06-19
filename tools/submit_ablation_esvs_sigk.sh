#!/usr/bin/env bash
# ES/VS/SIGK for the 4 ablation winners at leads 120 and 240 -> Table 3
# (calibration). Only 4 inits each, so cheap; run ~6 concurrent.
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=abl_esvs
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/abl_esvs_%j.log
set -uo pipefail
PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"
cd /users/sadamov/pyprojects/ai-models-ensembles
STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
OUT=/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/table_metrics
mkdir -p "$OUT"
VARS="2m_temperature mean_sea_level_pressure geopotential temperature u_component_of_wind v_component_of_wind specific_humidity"
esvs () { local lbl=$1 L=$2; shift 2
  $PY -u tools/energy_variogram_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --model-label "$lbl" \
    --out-csv "$OUT/esvs_${lbl}_L${L}.csv" > "$OUT/esvs_${lbl}_L${L}.log" 2>&1; }
sigk () { local lbl=$1 L=$2; shift 2
  $PY -u tools/signature_kernel_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --lead-stride 2 --n-pixels 128 --sigma 1.0 --dyadic 1 --seed 42 \
    --model-label "$lbl" --out-csv "$OUT/sigk_${lbl}_L${L}.csv" > "$OUT/sigk_${lbl}_L${L}.log" 2>&1; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done; }
declare -A ABL=(
  [aurora_enc]="$STORE/ablation/phase2b/aurora/*/mag_0.025_layer_encoder/forecast.zarr"
  [graphcast_all]="$STORE/ablation/phase1/graphcast_operational/*/mag_0.01_layer_all/forecast.zarr"
  [sfno_modes10]="$STORE/ablation/phase3/sfno/*/mag_0.25_modes10/forecast.zarr"
  [aifs_decoder]="$STORE/ablation/phase2/aifs/*/mag_0.027500_layer_decoder/forecast.zarr" )
for L in 120 240; do
  for lbl in "${!ABL[@]}"; do
    Z=$(ls -d ${ABL[$lbl]})
    # shellcheck disable=SC2086
    throttle; esvs "abl_$lbl" "$L" $Z &
    # shellcheck disable=SC2086
    throttle; sigk "abl_$lbl" "$L" $Z &
  done
done
wait
echo "ABLATION ES/VS/SIGK DONE"; ls -la "$OUT"/*abl*.csv
