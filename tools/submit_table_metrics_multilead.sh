#!/usr/bin/env bash
# Multi-lead ES / VS / SIGK for the paper table restructure (2026-06-19 review).
#   - Production (7 baselines): ES/VS/SIGK at leads 24, 120, 240  -> Tab. esvs (4) + Tab. C1
#   - Ablation winners (4):     ES/VS/SIGK at leads 120, 240      -> Tab. calibration (3)
# CPU-only (no GPU). Heavy zarr I/O. Per-(label,lead) CSVs land in scratch; a
# follow-up assembles the LaTeX tables.
#
# Usage: sbatch tools/submit_table_metrics_multilead.sh
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=tbl_metrics
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/tbl_metrics_%j.log

set -uo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
WB2A=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr
WB2B=/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr
OUT=/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/table_metrics
mkdir -p "$OUT"

VARS="2m_temperature mean_sea_level_pressure geopotential temperature \
u_component_of_wind v_component_of_wind specific_humidity"

esvs () {  # label  lead  zarr-glob
    local lbl=$1 L=$2; shift 2
    # shellcheck disable=SC2086
    $PY -u tools/energy_variogram_score.py --forecast-zarrs $* \
        --truth-zarrs "$WB2A" "$WB2B" --variables $VARS --levels 500 850 \
        --lead "$L" --model-label "$lbl" \
        --out-csv "$OUT/esvs_${lbl}_L${L}.csv" > "$OUT/esvs_${lbl}_L${L}.log" 2>&1
}
sigk () {  # label  lead  zarr-glob
    local lbl=$1 L=$2; shift 2
    # shellcheck disable=SC2086
    $PY -u tools/signature_kernel_score.py --forecast-zarrs $* \
        --truth-zarrs "$WB2A" "$WB2B" --variables $VARS --levels 500 850 \
        --lead "$L" --lead-stride 2 --n-pixels 128 --sigma 1.0 --dyadic 1 --seed 42 \
        --model-label "$lbl" \
        --out-csv "$OUT/sigk_${lbl}_L${L}.csv" > "$OUT/sigk_${lbl}_L${L}.log" 2>&1
}

# cap background concurrency at ~9 (bash wait -n); pass zarr lists as direct
# word-split args (NOT via echo/read, which split the multi-line ls output and
# silently scored only the first init - the 2026-06-19 n_inits=1 bug).
throttle () { while [ "$(jobs -rp | wc -l)" -ge 9 ]; do wait -n; done; }

PROD=(aifsens atlas aifs_perturbed fcn3 graphcast_all aurora_encoder sfno_modes10)

# ---- production: ES/VS/SIGK at 120, 240 ----
for L in 120 240; do
    for b in "${PROD[@]}"; do
        Z=$(ls -d "$STORE"/baselines/"$b"/*/forecast.zarr)
        # shellcheck disable=SC2086
        throttle; esvs "$b" "$L" $Z &
        # shellcheck disable=SC2086
        throttle; sigk "$b" "$L" $Z &
    done
    wait
    echo "=== production lead $L done ==="
done

# ---- ablation winners: ES/VS/SIGK at 120, 240 ----
declare -A ABL=(
  [aurora_enc]="$STORE/ablation/phase2b/aurora/*/mag_0.025_layer_encoder/forecast.zarr"
  [graphcast_all]="$STORE/ablation/phase1/graphcast_operational/*/mag_0.01_layer_all/forecast.zarr"
  [sfno_modes10]="$STORE/ablation/phase3/sfno/*/mag_0.25_modes10/forecast.zarr"
  [aifs_decoder]="$STORE/ablation/phase2/aifs/*/mag_0.027500_layer_decoder/forecast.zarr"
  # --- Phase 1-3b NON-WINNERS (one per model x phase = 13 rows of Tab. calibration) ---
  [aurora_p1]="$STORE/ablation/phase1/aurora/*/mag_0.03_layer_all/forecast.zarr"
  [aurora_p2]="$STORE/ablation/phase2/aurora/*/mag_0.044176_layer_encoder/forecast.zarr"
  [aurora_p3]="$STORE/ablation/phase3/aurora/*/mag_0.40_layer_unet_bottom/forecast.zarr"
  [aurora_p3b]="$STORE/ablation/phase3b/aurora/*/mag_0.015_layer_enc_012/forecast.zarr"
  [graphcast_p2]="$STORE/ablation/phase2/graphcast_operational/*/mag_0.029665_layer_m2g/forecast.zarr"
  [graphcast_p2b]="$STORE/ablation/phase2b/graphcast_operational/*/mag_0.014_layer_g2m/forecast.zarr"
  [graphcast_p3]="$STORE/ablation/phase3/graphcast_operational/*/gcsigma_1.80_gcnodes42/forecast.zarr"
  [graphcast_p3b]="$STORE/ablation/phase3b/graphcast_operational/*/gcsigma_0.159_gcnodes162/forecast.zarr"
  [sfno_p1]="$STORE/ablation/phase1/sfno/*/mag_0.03_layer_all/forecast.zarr"
  [sfno_p2]="$STORE/ablation/phase2/sfno/*/mag_0.053852_layer_encoder/forecast.zarr"
  [sfno_p2b]="$STORE/ablation/phase2b/sfno/*/mag_0.035_layer_encoder/forecast.zarr"
  [sfno_p3b]="$STORE/ablation/phase3b/sfno/*/mag_0.035_modes20/forecast.zarr"
  [aifs_p1]="$STORE/ablation/phase1/aifs/*/mag_0.01_layer_all/forecast.zarr"
)
for L in 120 240; do
    for lbl in "${!ABL[@]}"; do
        Z=$(ls -d ${ABL[$lbl]})
        # shellcheck disable=SC2086
        throttle; esvs "abl_$lbl" "$L" $Z &
        # shellcheck disable=SC2086
        throttle; sigk "abl_$lbl" "$L" $Z &
    done
    wait
    echo "=== ablation lead $L done ==="
done

echo "ALL DONE. CSVs in $OUT"
ls -la "$OUT"/*.csv
