#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SFNO refresh-every-20 (Phase 6c) on the 112-init PRODUCTION grid.
#
# Purpose: put the SFNO frozen-vs-refresh pair of Fig. fig:phase6 on the same
# 112-init production grid as the Aurora/AIFS pairs (the dedicated SFNO refresh
# sweep only ever ran on the 4-init ablation grid). Frozen counterpart is the
# existing production baseline `sfno_modes10` (sigma=0.25, coarse modes l<10).
#
# Config (matches the established 4-init `refresh20_0.35` cell):
#   SFNO_FRESH=1, sigma_N = 0.35 = 0.25*sqrt(40/20) (variance budget, T=40),
#   mode_cut=10, refresh_every=20.  Lead 240h (T=40); the step-41 refresh never
#   fires within 40 steps, so SSR@<=240h is identical to a 360h rollout.
#
# One sbatch per init (process isolation) to avoid the SFNO multi-GPU
# multiprocessing SIGSEGV that bites the week-helper; /dev/shm semaphores are
# swept before and after each run.
#
# Output: $STORE/baselines/sfno_p6c/<YYYYMMDD_HHMM>/forecast.zarr
# After all inits land, compute only the spatial-mean SSR (the single metric
# Fig. fig:phase6 needs) with tools/spatial_mean_ssr.py.
# ---------------------------------------------------------------------------
set -euo pipefail

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
SRC_DIR=/users/sadamov/pyprojects/ai-models-ensembles
LOG_DIR=$STORE/baseline_logs
WORKDIR=/workspace/ai-models-ensembles
E2S_CACHE_DIR=/iopsstor/scratch/cscs/sadamov/e2s_cache
mkdir -p "$LOG_DIR" "$E2S_CACHE_DIR"

MOUNTS="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"

PARTITION="${PARTITION:-normal}"
SIGMA=0.35
MODE_CUT=10
REFRESH=20
LEAD_HOURS=240
NUM_MEMBERS=10
SEED=42
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"

WEEK_STARTS=(
    "2023-01-02" "2023-04-02" "2023-07-02" "2023-10-02"
    "2024-01-02" "2024-04-02" "2024-07-02" "2024-10-02"
)

count=0
init_idx=0
for week_start in "${WEEK_STARTS[@]}"; do
    for day_offset in 0 1 2 3 4 5 6; do
        init_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${week_start}') + timedelta(days=${day_offset})).strftime('%Y-%m-%d'))")
        for hour in "00:00" "12:00"; do
            init_time="${init_date}T${hour}"
            init_tag="${init_date//-/}_${hour//:}"
            out_dir="$STORE/baselines/sfno_p6c/${init_tag}"
            out_zarr="$out_dir/forecast.zarr"

            if [[ -d "$out_zarr" ]]; then
                echo "SKIP $init_tag (exists)"
                continue
            fi
            [[ -d "$out_dir/_e2s_work" ]] && rm -rf "$out_dir/_e2s_work"
            mkdir -p "$out_dir"

            delay=$((init_idx % 30 * 2))   # stagger starts, wrap so it never queues too far out
            init_idx=$((init_idx + 1))
            job_tag="bl_sfno_p6c_${init_tag}"

            jobid=$(sbatch --parsable \
                --begin="now+${delay}minutes" \
                --job-name="$job_tag" \
                --partition="$PARTITION" --account=a122 \
                --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=444G --gres=gpu:4 \
                --time=02:00:00 \
                --output="$LOG_DIR/${job_tag}_%j.out" \
                --error="$LOG_DIR/${job_tag}_%j.err" \
                --container-image="$STORE/sfno.sqsh" \
                --container-mounts="$MOUNTS" \
                --container-workdir="$WORKDIR" \
                --wrap="find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true; \
                    SFNO_FRESH=1 SFNO_FRESH_SIGMA=$SIGMA SFNO_FRESH_MODE_CUT=$MODE_CUT SFNO_FRESH_REFRESH_EVERY=$REFRESH \
                    python -m ai_models_ensembles.cli infer \
                    --model sfno \
                    --init '${init_time}' \
                    --lead-hours $LEAD_HOURS \
                    --members $NUM_MEMBERS \
                    --weight-magnitude 0 \
                    --coarse-mode-cut $MODE_CUT \
                    --data-source arco \
                    --output-levels '$OUTPUT_LEVELS' \
                    --output-vars '$OUTPUT_VARS' \
                    --seed $SEED \
                    --output '${out_zarr}'; \
                    STATUS=\$?; \
                    find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true; \
                    exit \$STATUS")
            echo "  $job_tag -> $jobid (+${delay}min)"
            count=$((count + 1))
        done
    done
done

echo ""
echo "Submitted $count sfno_p6c inference jobs (112 inits, 240h, refresh-every-20 sigma=0.35)."
echo "Monitor: squeue -u \$USER | grep bl_sfno_p6c"
echo "Output:  $STORE/baselines/sfno_p6c/<init_tag>/forecast.zarr"
