#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit probabilistic baseline inference matching the IFS ENS eval period.
#
# 3 probabilistic models (fcn3, atlas, aifsens) x 112 init times:
#   8 weeks: Jan 2-8, Apr 2-8, Jul 2-8, Oct 2-8 in 2023 and 2024
#   x 2 inits/day (00Z, 12Z) = 112 init times per model.
#
# Matches ifs_ens_wb2.zarr exactly (same init_times, lead time, levels).
#
# Submission modes:
#   * WEEK_HELPER (default for all models): one sbatch per (model, week),
#     each runs 14 inits sequentially in a shell helper. /dev/shm semaphores
#     leaked by Python multiprocessing are cleaned between inits to avoid IPC
#     exhaustion -> SIGSEGV in round 2/3 of a later init.
#   * PER_INIT (opt-in via PER_INIT=1 env var): one sbatch per (model, init_time).
#     Full process isolation, ~14x more jobs but immune to the multi-GPU
#     multiprocessing state-accumulation SIGSEGV that affects SFNO when running
#     many inits in one process tree. Use when a model exhibits SIGSEGV with
#     the week-helper despite the IPC cleanup.
#
# Usage:
#   bash scripts/submit_all_inference.sh              # all probabilistic models + sfno_modes10
#   bash scripts/submit_all_inference.sh fcn3 atlas    # specific models
#   PER_INIT=1 bash scripts/submit_all_inference.sh ...  # force per-init for any model
#
# Options (env vars):
#   CHAIN=1                Chain jobs sequentially per model (--dependency=afterany).
#                          Useful for CDS-based models to avoid API throttling.
#   AFTER_JOB=N            Wait for slurm job N before starting first job per model.
#   PER_INIT=1             Force per-init submission for all models.
#   PER_INIT_TIME_LIMIT=T  Walltime per per-init job (default 01:00:00).
#   WEEKS=YYYY-MM-DD,...   Comma-separated week_starts to limit scope.
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/baseline_logs"
WORKDIR=/workspace/ai-models-ensembles

LEAD_HOURS=360
NUM_MEMBERS=10
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"
SEED=42
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="12:00:00"
CHAIN="${CHAIN:-0}"
AFTER_JOB="${AFTER_JOB:-}"  # wait for this job before starting first job per model

# Exact week starts matching ifs_ens_wb2.zarr init_times
if [[ -n "${WEEKS:-}" ]]; then
    IFS=',' read -ra WEEK_STARTS <<< "$WEEKS"
else
    WEEK_STARTS=(
        "2023-01-02"   # DJF 2023
        "2023-04-02"   # MAM 2023
        "2023-07-02"   # JJA 2023
        "2023-10-02"   # SON 2023
        "2024-01-02"   # DJF 2024
        "2024-04-02"   # MAM 2024
        "2024-07-02"   # JJA 2024
        "2024-10-02"   # SON 2024
    )
fi

# Probabilistic baselines + post-hoc SFNO modes10 perturbation as a baseline
MODELS="fcn3 atlas aifsens sfno_modes10"
declare -A MODEL_IDS=( [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens [sfno_modes10]=sfno )
declare -A DATA_SRC=( [fcn3]=arco [atlas]=arco [aifsens]=cds [sfno_modes10]=arco )
declare -A CONTAINER_BASE=( [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens [sfno_modes10]=sfno )
# Per-model extra inference flags (e.g. post-hoc perturbation).
declare -A EXTRA_FLAGS=( [sfno_modes10]="--weight-magnitude 0.25 --coarse-mode-cut 10" )
# PER_INIT mode (env var, default 0 for all models): submit one sbatch per init
# instead of a 14-init week-helper. Use when a model exhibits multiprocessing
# state-accumulation SIGSEGVs across sequential inits (see MEMORY.md, e.g. SFNO).
# Opt in with `PER_INIT=1 bash scripts/submit_all_inference.sh ...`.
REQUESTED="${@:-$MODELS}"
# Single-job per-init walltime (used by PER_INIT mode). Each init is ~12 min
# for SFNO so 1h is comfortable.
PER_INIT_TIME_LIMIT="${PER_INIT_TIME_LIMIT:-01:00:00}"

mkdir -p "$LOG_DIR"

# Per-model last job ID for chaining (seed with AFTER_JOB if set)
declare -A LAST_JOB=()
if [[ -n "$AFTER_JOB" ]]; then
    for m in $REQUESTED; do LAST_JOB[$m]="$AFTER_JOB"; done
fi

count=0
for model in $REQUESTED; do
    model_id="${MODEL_IDS[$model]:-}"
    if [[ -z "$model_id" ]]; then
        echo "SKIP $model: not a probabilistic baseline"
        continue
    fi

    container="$STORE/${CONTAINER_BASE[$model]}.sqsh"
    dsrc="${DATA_SRC[$model]}"
    extra_flags="${EXTRA_FLAGS[$model]:-}"
    # Per-init mode (default 0 / week-helper). Opt in via PER_INIT=1.
    per_init="${PER_INIT:-0}"

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        continue
    fi

    # Container mounts -- overlay the installed package with bind-mounted source
    mounts="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    for week_start in "${WEEK_STARTS[@]}"; do
        week_tag="${week_start//-/}"

        if [[ "$per_init" == "1" ]]; then
            # PER_INIT mode: one sbatch per init -- full process isolation.
            # Stagger starts by 2 min to spread NGC checkpoint fetches.
            init_idx=0
            for day_offset in 0 1 2 3 4 5 6; do
                init_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${week_start}') + timedelta(days=${day_offset})).strftime('%Y-%m-%d'))")
                for hour in "00:00" "12:00"; do
                    init_time="${init_date}T${hour}"
                    init_tag="${init_date//-/}_${hour//:}"
                    out_dir="$STORE/baselines/${model}/${init_tag}"
                    out_zarr="$out_dir/forecast.zarr"

                    if [[ -d "$out_zarr" ]]; then
                        continue
                    fi
                    if [[ -d "$out_dir/_e2s_work" ]]; then
                        rm -rf "$out_dir/_e2s_work"
                    fi

                    delay=$((init_idx * 2))
                    init_idx=$((init_idx + 1))
                    job_tag="bl_${model}_${init_tag}"

                    dep_flag=()
                    if [[ -n "${LAST_JOB[$model]:-}" ]]; then
                        dep_flag=(--dependency="afterany:${LAST_JOB[$model]}")
                    fi

                    jobid=$(sbatch --parsable \
                        "${dep_flag[@]}" \
                        --begin="now+${delay}minutes" \
                        --job-name="$job_tag" \
                        --partition="$PARTITION" \
                        --account=a122 \
                        --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=800G --gres=gpu:4 \
                        --time="$PER_INIT_TIME_LIMIT" \
                        --output="$LOG_DIR/${job_tag}_%j.out" \
                        --error="$LOG_DIR/${job_tag}_%j.err" \
                        --container-image="$container" \
                        --container-mounts="$mounts" \
                        --container-workdir="$WORKDIR" \
                        --wrap="python -m ai_models_ensembles.cli infer --model $model_id --init '${init_time}' --lead-hours $LEAD_HOURS --members $NUM_MEMBERS --data-source $dsrc --output-levels '$OUTPUT_LEVELS' --output-vars '$OUTPUT_VARS' --seed $SEED ${extra_flags} --output '${out_zarr}'")
                    echo "  $job_tag -> $jobid (+${delay}min)"
                    [[ "$CHAIN" == "1" ]] && LAST_JOB[$model]="$jobid"
                    count=$((count + 1))
                done
            done
            continue
        fi

        # WEEK_HELPER mode (default for non-SFNO models): one job per (model, week),
        # 14 inits run sequentially. Between inits, clean Python multiprocessing
        # semaphores leaked into /dev/shm by the multi-GPU pool. Accumulated leaks
        # can exhaust IPC and trigger SIGSEGV in round 2/3 of a later init.
        helper="$STORE/baseline_logs/bl_${model}_${week_tag}.sh"
        cat > "$helper" <<SCRIPT
#!/bin/sh
set -e
SCRIPT

        any_missing=false
        for day_offset in 0 1 2 3 4 5 6; do
            init_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${week_start}') + timedelta(days=${day_offset})).strftime('%Y-%m-%d'))")
            for hour in "00:00" "12:00"; do
                init_time="${init_date}T${hour}"
                init_tag="${init_date//-/}_${hour//:}"

                out_dir="$STORE/baselines/${model}/${init_tag}"
                out_zarr="$out_dir/forecast.zarr"

                if [[ -d "$out_zarr" ]]; then
                    continue
                fi
                if [[ -d "$out_dir/_e2s_work" ]]; then
                    rm -rf "$out_dir/_e2s_work"
                fi

                any_missing=true
                cat >> "$helper" <<SCRIPT
echo "=== ${init_time} ==="
python -m ai_models_ensembles.cli infer \\
    --model $model_id \\
    --init '${init_time}' \\
    --lead-hours $LEAD_HOURS \\
    --members $NUM_MEMBERS \\
    --data-source $dsrc \\
    --output-levels '$OUTPUT_LEVELS' \\
    --output-vars '$OUTPUT_VARS' \\
    --seed $SEED \\
    ${extra_flags} \\
    --output '${out_zarr}'
find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true
sleep 15
SCRIPT
            done
        done

        if ! $any_missing; then
            echo "  SKIP $model week $week_start: all outputs exist"
            rm -f "$helper"
            continue
        fi

        chmod +x "$helper"

        job_tag="bl_${model}_${week_tag}"
        echo "  $job_tag (week helper)"

        dep_flag=()
        if [[ -n "${LAST_JOB[$model]:-}" ]]; then
            dep_flag=(--dependency="afterany:${LAST_JOB[$model]}")
        fi

        jobid=$(sbatch --parsable \
            "${dep_flag[@]}" \
            --job-name="$job_tag" \
            --partition="$PARTITION" \
            --account=a122 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=32 \
            --mem=800G \
            --gres=gpu:4 \
            --time="$TIME_LIMIT" \
            --output="$LOG_DIR/${job_tag}_%j.out" \
            --error="$LOG_DIR/${job_tag}_%j.err" \
            --container-image="$container" \
            --container-mounts="$mounts" \
            --container-workdir="$WORKDIR" \
            --wrap="sh ${helper}")

        [[ "$CHAIN" == "1" ]] && LAST_JOB[$model]="$jobid"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted $count baseline jobs."
echo "Monitor with: squeue -u \$USER | grep bl_"
echo "Logs: $LOG_DIR/"
echo ""
echo "Output layout:"
echo "  $STORE/baselines/<model>/<YYYYMMDD_HHMM>/forecast.zarr"
