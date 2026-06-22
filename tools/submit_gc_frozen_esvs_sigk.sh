#!/usr/bin/env bash
# ES/VS/SIGK for the two FROZEN GraphCast Phase-3 calibration rows at leads 120
# and 240 -> Table 3 (calibration) frozen recompute. Mirrors
# submit_ablation_esvs_sigk.sh exactly; only the run set differs (the frozen
# sweep superseded the fresh one in 6138906, so the published p3/p3b ES/VS/SIGK
# are the stale fresh values). 4 inits each, cheap.
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gcfrz_esvs
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/gcfrz_esvs_%j.log
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
  [graphcast_p3frozen]="$STORE/ablation/phase3/graphcast_operational/*/gcsigma_1.0_gcnodes42_frozen/forecast.zarr"
  [graphcast_p3bfrozen]="$STORE/ablation/phase3b/graphcast_operational/*/gcsigma_0.159_gcnodes162_frozen/forecast.zarr" )
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
echo "GC FROZEN ES/VS/SIGK DONE"; ls -la "$OUT"/*p3frozen*.csv "$OUT"/*p3bfrozen*.csv 2>/dev/null
