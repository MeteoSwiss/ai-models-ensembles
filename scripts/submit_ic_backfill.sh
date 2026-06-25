#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Backfill the 28 missing IC-augmented (_ic) inits for all four backbones.
#
# The four _ic runs (aurora_encoder_ic, graphcast_all_ic, sfno_modes10_ic,
# aifs_perturbed_ic) are 84/112 complete: the first two 2023 weeks
# (Jan 2-8, Apr 2-8, 00/12Z = 28 inits) are NaN at leads >0. ROOT CAUSE
# (verified 2026-06-24): the IFS-ENS perturbed-IC zarr had 100% NaN at the
# 150 + 600 hPa pressure levels on exactly those 28 inits (the post-download
# log-p interpolation pass never ran on the Jan/Apr-2023 init_time slots), so
# every backbone's first AR step ingested NaN at 2 of 13 input levels and the
# whole rollout collapsed to NaN past lead 0 (regardless of model). Fixed by
# tools/fill_ic_perturbed_levels.py (log-p fill, run via tools/submit_ic_fill.sh)
# BEFORE this backfill. Paper claims the _ic runs span the full 112-init grid
# -- repairing the IC zarr then backfilling makes that true.
#
# IC augmentation: each member m is seeded from IFS-ENS EDA member m's
# perturbed analysis in the local zarr below (verified to cover all 28 inits +
# their t-6h). --data-source stays arco/cds as the fallback for the
# surface/boundary fields the IC zarr does not carry. Weight perturbation is
# kept at the production recipe per model (hybrid weight + IC spread).
#
# PER-INIT isolation for ALL models (one sbatch per init, full process
# isolation) -- avoids both the SFNO multi-GPU SIGSEGV and the early-2023
# week-helper NaN-merge that caused this gap.
#
# Usage:
#   DRY_RUN=1 bash scripts/submit_ic_backfill.sh           # preview, no submit
#   bash scripts/submit_ic_backfill.sh                      # all 4 models
#   bash scripts/submit_ic_backfill.sh aurora_encoder_ic    # subset
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR=/workspace/ai-models-ensembles
LOG_DIR="$STORE/baseline_logs"
HOST_PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python

IC_ZARR=/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr
IC_DIR=$(dirname "$IC_ZARR")
E2S_CACHE_DIR="/iopsstor/scratch/cscs/sadamov/e2s_cache"

LEAD_HOURS=240
NUM_MEMBERS=10
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"
SEED=42
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
DRY_RUN="${DRY_RUN:-0}"
STAMP=$(date +%Y-%m-%d)

# 28 missing inits: Jan 2-8 and Apr 2-8 2023, 00Z and 12Z
INIT_DATES=()
for d in 02 03 04 05 06 07 08; do
    INIT_DATES+=("2023-01-${d}" "2023-04-${d}")
done
HOURS=("00:00" "12:00")
# CANARY=1: one init (2023-01-02 00Z) per model, to validate the recipe first.
if [[ "${CANARY:-0}" == "1" ]]; then
    INIT_DATES=("2023-01-02")
    HOURS=("00:00")
fi

# run -> "model_id data_source container_base weight_flags"
declare -A RUN_SPEC=(
    [aurora_encoder_ic]="aurora arco aurora --weight-magnitude 0.025 --layer encoder"
    [graphcast_all_ic]="graphcast_operational arco graphcast --weight-magnitude 0.01 --layer all"
    [sfno_modes10_ic]="sfno arco sfno --weight-magnitude 0.25 --coarse-mode-cut 10"
    [aifs_perturbed_ic]="aifs cds aifs --weight-magnitude 0.0275 --layer decoder"
)

REQUESTED="${*:-aurora_encoder_ic graphcast_all_ic sfno_modes10_ic aifs_perturbed_ic}"
mkdir -p "$LOG_DIR" "$E2S_CACHE_DIR"

# Return 0 if forecast.zarr is good (finite at lead 6h), 1 otherwise.
is_good() {
    "$HOST_PY" - "$1" <<'PY' 2>/dev/null
import sys, numpy as np, xarray as xr
try:
    f = xr.open_zarr(sys.argv[1], consolidated=True, chunks={})
    da = f["mean_sea_level_pressure"].isel(ensemble=0)
    if "init_time" in da.dims: da = da.isel(init_time=0)
    v = float(da.isel(lead_time=1, latitude=360, longitude=720).values)
    sys.exit(0 if np.isfinite(v) else 1)
except Exception:
    sys.exit(1)
PY
}

count=0; idx=0
for run in $REQUESTED; do
    spec="${RUN_SPEC[$run]:-}"
    [[ -z "$spec" ]] && { echo "SKIP $run: unknown run"; continue; }
    read -r model_id dsrc cbase wflags <<< "$spec"
    wflags="${spec#"$model_id $dsrc $cbase "}"

    container="$STORE/${cbase}.sqsh"
    [[ -f "$container" ]] || { echo "SKIP $run: container $container missing"; continue; }

    mounts="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE_DIR}:/workspace/.cache/earth2studio,${IC_DIR}:${IC_DIR}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    for init_date in "${INIT_DATES[@]}"; do
        for hour in "${HOURS[@]}"; do
            init_time="${init_date}T${hour}"
            init_tag="${init_date//-/}_${hour//:}"
            # Skip inits listed in SKIP_INIT_TAGS (space-separated), e.g. one
            # currently being rolled out by the canary -- avoids a duplicate
            # job racing the same output dir.
            if [[ " ${SKIP_INIT_TAGS:-} " == *" ${init_tag} "* ]]; then
                echo "  SKIP $run $init_tag: in SKIP_INIT_TAGS"
                continue
            fi
            out_dir="$STORE/baselines/${run}/${init_tag}"
            out_zarr="$out_dir/forecast.zarr"

            if [[ -d "$out_zarr" ]]; then
                if is_good "$out_zarr"; then
                    echo "  SKIP $run $init_tag: already good"
                    continue
                fi
                # NaN shell -> archive (never delete forecast.zarr), then regen
                archive="$out_dir/forecast.zarr.nan_pre_backfill_${STAMP}"
                echo "  ARCHIVE NaN $run $init_tag -> $(basename "$archive")"
                [[ "$DRY_RUN" == "1" ]] || { rm -rf "$archive"; mv "$out_zarr" "$archive"; }
            fi
            [[ -d "$out_dir/_e2s_work" && "$DRY_RUN" != "1" ]] && rm -rf "$out_dir/_e2s_work"

            delay=$((idx * 1)); idx=$((idx + 1))
            job_tag="ic_${run}_${init_tag}"
            wrap="python -m ai_models_ensembles.cli infer --model $model_id --init '${init_time}' --lead-hours $LEAD_HOURS --members $NUM_MEMBERS --data-source $dsrc --output-levels '$OUTPUT_LEVELS' --output-vars '$OUTPUT_VARS' --seed $SEED ${wflags} --ic-zarr '${IC_ZARR}' --output '${out_zarr}'"

            if [[ "$DRY_RUN" == "1" ]]; then
                [[ $count -lt 2 ]] && echo "  [dry] $job_tag: $wrap"
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
done

echo ""
echo "$([[ "$DRY_RUN" == "1" ]] && echo "[DRY_RUN] would submit" || echo "Submitted") $count per-init backfill jobs."
