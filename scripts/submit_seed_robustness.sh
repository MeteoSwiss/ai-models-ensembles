#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Second-seed robustness probe (reviewer M3): rerun the four production-pick
# ablation cells at SEED=43 on the four ablation inits, so we can quantify
# seed-to-seed CRPSS@240h variability and back the "cells within +-0.02 are
# tied" selection statement.
#
# The seed (42) is NOT part of the ablation output path, and the existing
# seed=42 forecast.zarr must never be overwritten, so these runs write to a
# SEPARATE tree: $STORE/ablation_seed43/<model_id>/<init_tag>/<run_tag>.
# Weight-only (no --ic-zarr), matching the ablation cells. Per-init isolation
# (one sbatch per init) to dodge the SFNO multi-GPU SIGSEGV.
#
# Usage:
#   DRY_RUN=1 bash scripts/submit_seed_robustness.sh        # preview
#   bash scripts/submit_seed_robustness.sh                  # all 4 picks
#   bash scripts/submit_seed_robustness.sh sfno_modes10     # subset
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR=/workspace/ai-models-ensembles
LOG_DIR="$STORE/ablation_logs"
HOST_PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
E2S_CACHE_DIR="/iopsstor/scratch/cscs/sadamov/e2s_cache"

OUT_BASE="$STORE/ablation_seed43"
LEAD_HOURS=240
NUM_MEMBERS=10
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"
SEED="${SEED:-43}"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
DRY_RUN="${DRY_RUN:-0}"

# The four ablation inits (mid-season). Override with SEED_INITS="a b c".
if [[ -n "${SEED_INITS:-}" ]]; then
    read -r -a INIT_DATES <<< "$SEED_INITS"
else
    INIT_DATES=("2023-05-15" "2023-08-15" "2024-02-15" "2024-11-15")
fi
HOUR="00:00"
if [[ "${CANARY:-0}" == "1" ]]; then INIT_DATES=("2023-08-15"); fi

# run -> "model_id data_source container_base run_tag weight_flags"
declare -A RUN_SPEC=(
    [aurora_encoder]="aurora arco aurora mag_0.025_layer_encoder --weight-magnitude 0.025 --layer encoder"
    [graphcast_all]="graphcast_operational arco graphcast mag_0.01_layer_all --weight-magnitude 0.01 --layer all"
    [sfno_modes10]="sfno arco sfno mag_0.25_modes10 --weight-magnitude 0.25 --coarse-mode-cut 10"
    [aifs_perturbed]="aifs cds aifs mag_0.027500_layer_decoder --weight-magnitude 0.0275 --layer decoder"
)

REQUESTED="${*:-aurora_encoder graphcast_all sfno_modes10 aifs_perturbed}"
mkdir -p "$LOG_DIR" "$E2S_CACHE_DIR"

count=0; idx=0
for run in $REQUESTED; do
    spec="${RUN_SPEC[$run]:-}"
    [[ -z "$spec" ]] && { echo "SKIP $run: unknown run"; continue; }
    read -r model_id dsrc cbase run_tag _rest <<< "$spec"
    wflags="${spec#"$model_id $dsrc $cbase $run_tag "}"

    container="$STORE/${cbase}.sqsh"
    [[ -f "$container" ]] || { echo "SKIP $run: container $container missing"; continue; }

    mounts="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    for init_date in "${INIT_DATES[@]}"; do
        init_time="${init_date}T${HOUR}"
        init_tag="${init_date//-/}"
        out_dir="$OUT_BASE/${model_id}/${init_tag}/${run_tag}"
        out_zarr="$out_dir/forecast.zarr"
        if [[ -d "$out_zarr" ]]; then
            echo "  SKIP $run $init_tag: exists ($out_zarr)"
            continue
        fi
        [[ -d "$out_dir/_e2s_work" && "$DRY_RUN" != "1" ]] && rm -rf "$out_dir/_e2s_work"

        delay=$((idx * 1)); idx=$((idx + 1))
        job_tag="seed${SEED}_${run}_${init_tag}"
        wrap="find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true; python -m ai_models_ensembles.cli infer --model $model_id --init '${init_time}' --lead-hours $LEAD_HOURS --members $NUM_MEMBERS --data-source $dsrc --output-levels '$OUTPUT_LEVELS' --output-vars '$OUTPUT_VARS' --seed $SEED ${wflags} --output '${out_zarr}'"

        if [[ "$DRY_RUN" == "1" ]]; then
            [[ $count -lt 4 ]] && echo "  [dry] $job_tag -> $out_zarr"
            count=$((count + 1)); continue
        fi

        sbatch --parsable \
            --begin="now+${delay}minutes" \
            --job-name="$job_tag" \
            --partition="$PARTITION" --account=a122 \
            --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=800G --gres=gpu:4 \
            --time="$TIME_LIMIT" \
            --output="$LOG_DIR/${job_tag}_%j.out" \
            --error="$LOG_DIR/${job_tag}_%j.err" \
            --container-image="$container" \
            --container-mounts="$mounts" \
            --container-workdir="$WORKDIR" \
            --wrap="$wrap" >/dev/null
        echo "  $job_tag (+${delay}min)"
        count=$((count + 1))
    done
done

echo ""
echo "$([[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] would submit" || echo "Submitted") $count seed=${SEED} robustness jobs -> $OUT_BASE"
