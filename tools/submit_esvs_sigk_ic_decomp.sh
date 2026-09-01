#!/usr/bin/env bash
# ES/VS/SIGK for the IC-versus-weight decomposition arms (Fuhrer review item 1).
#
# ic_weight_decomposition.py reports spread and fair CRPS for the weight-only /
# IC-only / weight+IC arms, but the multivariate and temporal-path scores have
# never been run on the *_ic_only arms - sigk_production.csv covers only the
# seven production baselines. Without them the decomposition can say how much
# variance each source contributes but not whether either source produces
# joint-calibrated or temporally realistic members.
#
# SIGK is the interesting one: the paper argues a frozen weight perturbation
# yields temporally over-coherent trajectories, and the IC-only arm is the
# natural control, since its members decorrelate from the start instead.
#
# Scored on the 112-init production grid with the fixed 1990-2019 climatological
# scale, matching the propriety fix adopted for review item 2. Leads 24 and 240
# bracket the regime change: IC dominates the spread early, weight noise late.
#
# Usage: sbatch tools/submit_esvs_sigk_ic_decomp.sh
#
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=esvs_ic_decomp
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/esvs_ic_decomp_%j.log
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
B=$STORE/baselines
WB2A=${AIENS_WB2_22:-/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr}
WB2B=${AIENS_WB2_24:-/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr}
OUT=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}/ai-models-ensembles/scratch/table_metrics_ic_decomp
mkdir -p "$OUT"
VARS="2m_temperature mean_sea_level_pressure geopotential temperature u_component_of_wind v_component_of_wind specific_humidity"

# label -> production baseline dir. Three arms per backbone, named so the arm is
# readable straight off the CSV: <backbone>_<wt|ic|both>.
declare -A RUNS=(
  [aurora_wt]="$B/aurora_encoder"
  [aurora_ic]="$B/aurora_ic_only"
  [aurora_both]="$B/aurora_encoder_ic"
  [graphcast_wt]="$B/graphcast_all"
  [graphcast_ic]="$B/graphcast_ic_only"
  [graphcast_both]="$B/graphcast_all_ic"
  [sfno_wt]="$B/sfno_modes10"
  [sfno_ic]="$B/sfno_ic_only"
  [sfno_both]="$B/sfno_modes10_ic"
  [aifs_wt]="$B/aifs_perturbed"
  [aifs_ic]="$B/aifs_ic_only"
  [aifs_both]="$B/aifs_perturbed_ic"
)

esvs () { local lbl=$1 L=$2; shift 2
  $PY -u tools/energy_variogram_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --model-label "$lbl" --scale-json "$SCALE" \
    --out-csv "$OUT/esvs_${lbl}_L${L}.csv" > "$OUT/esvs_${lbl}_L${L}.log" 2>&1; }
sigk () { local lbl=$1 L=$2; shift 2
  $PY -u tools/signature_kernel_score.py --forecast-zarrs $* --truth-zarrs "$WB2A" "$WB2B" \
    --variables $VARS --levels 500 850 --lead "$L" --lead-stride 2 --n-pixels 128 --sigma 1.0 \
    --dyadic 1 --seed 42 --model-label "$lbl" --scale-json "$SCALE" \
    --out-csv "$OUT/sigk_${lbl}_L${L}.csv" > "$OUT/sigk_${lbl}_L${L}.log" 2>&1; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done; }

for L in 24 240; do
  for lbl in "${!RUNS[@]}"; do
    Z=$(ls -d "${RUNS[$lbl]}"/*/forecast.zarr 2>/dev/null)
    n=$(printf '%s\n' $Z | grep -c . )
    if [[ -z "$Z" ]]; then echo "MISSING runs for $lbl: ${RUNS[$lbl]}"; continue; fi
    # An arm short of 112 is scored anyway but flagged here and carried in the
    # CSV's n_inits column, so a short arm can never be compared unknowingly.
    [[ "$n" -lt 112 ]] && echo "WARNING $lbl: only $n/112 inits at lead $L"
    # shellcheck disable=SC2086
    throttle; esvs "$lbl" "$L" $Z &
    # shellcheck disable=SC2086
    throttle; sigk "$lbl" "$L" $Z &
  done
done
wait
echo "IC-DECOMP ES/VS/SIGK DONE"; ls -la "$OUT" | tail -30
