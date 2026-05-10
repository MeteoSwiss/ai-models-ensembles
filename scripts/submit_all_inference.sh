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
# Each sbatch job processes one full week (14 inits) for one model
# sequentially, running 10 members per init with 4 GPUs in parallel.
# Total: 3 models x 8 weeks = 24 jobs.
#
# Usage:
#   bash scripts/submit_all_inference.sh              # all 3 prob models
#   bash scripts/submit_all_inference.sh fcn3 atlas    # specific models
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

# Exact week starts matching ifs_ens_wb2.zarr init_times
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

# Probabilistic baselines only
MODELS="fcn3 atlas aifsens"
declare -A MODEL_IDS=( [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens )
declare -A DATA_SRC=( [fcn3]=arco [atlas]=arco [aifsens]=cds )
REQUESTED="${@:-$MODELS}"

mkdir -p "$LOG_DIR"

count=0
for model in $REQUESTED; do
    model_id="${MODEL_IDS[$model]:-}"
    if [[ -z "$model_id" ]]; then
        echo "SKIP $model: not a probabilistic baseline"
        continue
    fi

    container="$STORE/${model}.sqsh"
    dsrc="${DATA_SRC[$model]}"

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        continue
    fi

    for week_start in "${WEEK_STARTS[@]}"; do
        week_tag="${week_start//-/}"

        # Write a helper script for this (model, week) to shared storage
        # so it's accessible from compute nodes. Each helper runs 14 inits
        # sequentially, skipping any that already have output.
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

                out_dir="$STORE/baselines/${model_id}/${init_tag}"
                out_zarr="$out_dir/forecast.zarr"

                if [[ -d "$out_zarr" ]]; then
                    continue
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
    --output '${out_zarr}'
SCRIPT
            done
        done

        if ! $any_missing; then
            echo "  SKIP $model week $week_start: all outputs exist"
            rm -f "$helper"
            continue
        fi

        chmod +x "$helper"

        # Container mounts
        mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
        for rc in ~/.cdsapirc ~/.ecmwfapirc; do
            [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
        done

        job_tag="bl_${model}_${week_tag}"
        echo "  $job_tag (14 inits)"

        sbatch --parsable \
            --job-name="$job_tag" \
            --partition="$PARTITION" \
            --account=a122 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=32 \
            --mem=444G \
            --gres=gpu:4 \
            --time="$TIME_LIMIT" \
            --output="$LOG_DIR/${job_tag}_%j.out" \
            --error="$LOG_DIR/${job_tag}_%j.err" \
            --container-image="$container" \
            --container-mounts="$mounts" \
            --container-workdir="$WORKDIR" \
            --wrap="sh ${helper}"

        ((count++)) || true
    done
done

echo ""
echo "Submitted $count baseline jobs (each processes up to 14 init times)."
echo "Monitor with: squeue -u \$USER | grep bl_"
echo "Logs: $LOG_DIR/"
echo ""
echo "Output layout:"
echo "  $STORE/baselines/<model>/<YYYYMMDD_HHMM>/forecast.zarr"
