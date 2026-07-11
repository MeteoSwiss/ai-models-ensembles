#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 6 refresh EXPLORATORY PROBES on the 4-init ablation grid.
#
# Reproduces the fresh-per-step ("refresh") experiments described in the paper
# methods (Refreshing the perturbation across the rollout):
#   1. SFNO cadence sweep  : refresh_every N in {1, 10, 20, 40} at the frozen
#                            production sigma (0.25, coarse modes l<10). Locates
#                            the no-cost cadence and motivates the variance-budget
#                            rescaling sigma_N = 0.25*sqrt(T/N).
#   2. Aurora refresh-20   : two-point sigma probe {0.025, 0.035} at refresh_every=20
#                            (production sigma and 1.4x it).
#   3. AIFS refresh-20     : two-point sigma probe {0.0275, 0.0385} at refresh_every=20.
#
# These probes produced the sec. "Refreshing the perturbation" numbers. The
# PRODUCTION-grid frozen-vs-refresh pairs of Fig. fig:phase6 are a separate
# artifact (tools/submit_sfno_p6c_production.sh for SFNO; the Aurora/AIFS p6c
# production grids were launched with the same env vars at refresh_every=20 and
# the production sigma, output under $STORE/baselines/{aurora,aifs}_p6c/).
#
# Each backbone is gated by its own env vars, read in
# ai_models_ensembles.e2s_inference._model_for_member; --weight-magnitude 0 so
# the upstream checkpoint perturbation is skipped and only the fresh hooks fire.
# One sbatch per (model, init, cell) for process isolation (the SFNO multi-GPU
# path SIGSEGVs otherwise; /dev/shm semaphores are swept around each run).
#
# Output: $STORE/ablation/phase6_probes/<model_id>/<init_tag>/<run_tag>/forecast.zarr
# Usage:  bash tools/submit_refresh_probes.sh [sfno|aurora|aifs|all]   (default all)
# ---------------------------------------------------------------------------
set -euo pipefail

WHICH="${1:-all}"

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
SRC_DIR=/users/sadamov/pyprojects/ai-models-ensembles
LOG_DIR=$STORE/baseline_logs
WORKDIR=/workspace/ai-models-ensembles
E2S_CACHE_DIR=/iopsstor/scratch/cscs/sadamov/e2s_cache
mkdir -p "$LOG_DIR" "$E2S_CACHE_DIR"

BASE_MOUNTS="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"

PARTITION="${PARTITION:-normal}"
LEAD_HOURS="${LEAD_HOURS:-240}"        # ablation-grid standard (T=40 at 6h step)
NUM_MEMBERS=10
SEED=42
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"

# 4 mid-season ablation inits (matches scripts/submit_ablation.sh).
INIT_TIMES=(
    "2024-02-15T00:00"   # DJF
    "2023-05-15T00:00"   # MAM
    "2023-08-15T00:00"   # JJA
    "2024-11-15T00:00"   # SON
)

declare -A MODEL_IDS=( [aurora]=aurora [graphcast]=graphcast_operational [sfno]=sfno [aifs]=aifs )

SWEEP_DELETE="find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true"

count=0
job_idx=0

submit_probe() {
    # $1 model  $2 init_time  $3 run_tag  $4 fresh env-var string  $5 extra CLI args
    local model=$1 init_time=$2 run_tag=$3 fresh_env=$4 extra_cli=$5
    local model_id="${MODEL_IDS[$model]}"
    local container="$STORE/${model}.sqsh"
    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        return 0
    fi
    local init_tag="${init_time%%T*}"; init_tag="${init_tag//-/}"
    local out_dir="$STORE/ablation/phase6_probes/${model_id}/${init_tag}/${run_tag}"
    local out_zarr="$out_dir/forecast.zarr"
    if [[ -d "$out_zarr" ]]; then
        echo "SKIP $model $init_tag $run_tag (exists)"
        return 0
    fi
    [[ -d "$out_dir/_e2s_work" ]] && rm -rf "$out_dir/_e2s_work"
    mkdir -p "$out_dir"

    local delay=$((job_idx % 30 * 2)); job_idx=$((job_idx + 1))
    local job_tag="p6probe_${model}_${init_tag}_${run_tag}"
    local jobid
    jobid=$(sbatch --parsable \
        --begin="now+${delay}minutes" \
        --job-name="$job_tag" \
        --partition="$PARTITION" --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=444G --gres=gpu:4 \
        --time=02:00:00 \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --container-image="$container" \
        --container-mounts="$BASE_MOUNTS" \
        --container-workdir="$WORKDIR" \
        --wrap="${SWEEP_DELETE}; \
            ${fresh_env} \
            python -m ai_models_ensembles.cli infer \
            --model $model \
            --init '${init_time}' \
            --lead-hours $LEAD_HOURS \
            --members $NUM_MEMBERS \
            --weight-magnitude 0 \
            ${extra_cli} \
            --data-source arco \
            --output-levels '$OUTPUT_LEVELS' \
            --output-vars '$OUTPUT_VARS' \
            --seed $SEED \
            --output '${out_zarr}'; \
            STATUS=\$?; ${SWEEP_DELETE}; exit \$STATUS")
    echo "  $job_tag -> $jobid (+${delay}min)"
    count=$((count + 1))
}

if [[ "$WHICH" == "sfno" || "$WHICH" == "all" ]]; then
    for n in 1 10 20 40; do
        for init_time in "${INIT_TIMES[@]}"; do
            submit_probe sfno "$init_time" "refresh${n}_0.25" \
                "SFNO_FRESH=1 SFNO_FRESH_SIGMA=0.25 SFNO_FRESH_MODE_CUT=10 SFNO_FRESH_REFRESH_EVERY=${n}" \
                "--coarse-mode-cut 10"
        done
    done
fi

if [[ "$WHICH" == "aurora" || "$WHICH" == "all" ]]; then
    for s in 0.025 0.035; do
        for init_time in "${INIT_TIMES[@]}"; do
            submit_probe aurora "$init_time" "refresh20_${s}" \
                "AURORA_FRESH=1 AURORA_FRESH_SIGMA=${s} AURORA_FRESH_REFRESH_EVERY=20" \
                ""
        done
    done
fi

if [[ "$WHICH" == "aifs" || "$WHICH" == "all" ]]; then
    for s in 0.0275 0.0385; do
        for init_time in "${INIT_TIMES[@]}"; do
            submit_probe aifs "$init_time" "refresh20_${s}" \
                "AIFS_FRESH=1 AIFS_FRESH_SIGMA=${s} AIFS_FRESH_REFRESH_EVERY=20" \
                ""
        done
    done
fi

echo ""
echo "Submitted $count refresh-probe inference jobs (which=$WHICH, ${LEAD_HOURS}h, 4 ablation inits)."
echo "Monitor: squeue -u \$USER | grep p6probe_"
echo "Output:  $STORE/ablation/phase6_probes/<model_id>/<init_tag>/<run_tag>/forecast.zarr"
