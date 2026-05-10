#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate ablation study outputs using SwissClim Evaluations (research branch).
#
# For each ablation run, generates a per-run SwissClim config and submits
# evaluation. Then runs intercomparison across all runs of the same model.
#
# Usage:
#   bash scripts/evaluate_ablation.sh phase1              # evaluate Phase 1
#   bash scripts/evaluate_ablation.sh phase2              # evaluate Phase 2
#   bash scripts/evaluate_ablation.sh phase1 aurora       # single model
#   bash scripts/evaluate_ablation.sh intercompare phase1  # cross-run comparison
#   bash scripts/evaluate_ablation.sh intercompare phase1 aurora  # per-model
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/ablation_logs"
WORKDIR=/workspace/ai-models-ensembles

PARTITION="${PARTITION:-normal}"
TIME_LIMIT="02:00:00"

# WeatherBench2 ERA5 reference data (covers 2022-2025)
WB2_PATHS='[
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr"
  ]'

# Init times (must match submit_ablation.sh)
INIT_TIMES=(
    "2024-02-15T00:00"
    "2023-05-15T00:00"
    "2023-08-15T00:00"
    "2024-11-15T00:00"
)

MODELS="aurora graphcast sfno"
declare -A MODEL_IDS=( [aurora]=aurora [graphcast]=graphcast_operational [sfno]=sfno )

# Evaluation modules: the ablation-relevant subset
# - energy_spectra: LSD for blurring detection
# - probabilistic: CRPS, spread-skill ratio for calibration
# - deterministic: RMSE, MAE per lead time
# - multivariate: bivariate histograms for physical consistency
# - histograms: distribution shifts
EVAL_MODULES="energy_spectra,probabilistic,deterministic,multivariate,histograms"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Generate a SwissClim Evaluations YAML config for a single ablation run.
# ---------------------------------------------------------------------------
generate_config() {
    local zarr_path=$1
    local output_root=$2
    local init_time=$3

    # Validation time range: init_time to init_time + 240h (10 days)
    local init_date="${init_time%%T*}"
    local end_date
    end_date=$(date -d "${init_date} + 10 days" +%Y-%m-%dT23 2>/dev/null || \
               python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${init_date}') + timedelta(days=10)).strftime('%Y-%m-%dT23'))")

    cat <<YAML
paths:
  ml: "${zarr_path}"
  nwp: ${WB2_PATHS}
  output_root: "${output_root}"

selection:
  levels: [500, 850]
  temporal_resolution_hours: null
  datetimes: ["${init_time}:${end_date}"]
  latitudes: [90.0, -89.75]
  longitudes: [0.0, 359.75]

  variables_2d:
    - "2m_temperature"
    - "mean_sea_level_pressure"

  variables_3d:
    - "geopotential"
    - "temperature"
    - "u_component_of_wind"
    - "v_component_of_wind"
    - "specific_humidity"

  ensemble_members: null

  ensemble:
    energy_spectra: mean
    probabilistic: prob
    deterministic: mean
    multivariate: mean
    histograms: pooled

  check_missing: false

derived_variables:
  wind_speed:
    kind: wind_speed
    u: u_component_of_wind
    v: v_component_of_wind

plotting:
  random_seed: 42
  plot_datetime: null
  dpi: 48
  output_mode: both
  histograms_per_lead: false
  maps_per_lead_grid: false
  energy_spectra_spectrogram: true
  probabilistic_line_plots: true

modules:
  maps: false
  histograms: true
  wd_kde: false
  energy_spectra: true
  vertical_profiles: false
  deterministic: true
  ets: false
  probabilistic: true
  ssim: false
  multivariate: true

lead_time:
  mode: stride
  stride:
    hours: 24
  max_hour: 240
  panel:
    strategy: evenly_spaced
    count: 5

metrics:
  deterministic:
    include: ["MAE", "RMSE", "Relative MAE"]
    standardized_include: ["MAE", "RMSE"]
    report_per_level: true
    error_maps: false
    fss:
      quantile: 0.90
      window_size: 9

  multivariate:
    bivariate_pairs:
      - ["temperature", "specific_humidity"]
      - ["u_component_of_wind", "v_component_of_wind"]
    coriolis_parameter: 1.0e-4
    bins: 100

performance:
  dask_scheduler: threaded
YAML
}

# ---------------------------------------------------------------------------
# Submit evaluation for a single ablation run
# ---------------------------------------------------------------------------
submit_eval() {
    local zarr_path=$1
    local init_time=$2
    local phase=$3
    local job_tag=$4

    local eval_root
    eval_root="$(dirname "$zarr_path")/swissclim_eval"

    # Skip if already evaluated
    if [[ -d "$eval_root" ]]; then
        echo "  SKIP $job_tag: eval output exists"
        return 0
    fi

    local config_path
    config_path="$(dirname "$zarr_path")/swissclim_config.yaml"
    generate_config "$zarr_path" "$eval_root" "$init_time" > "$config_path"

    # Use the first available model container (all have swissclim-evaluations)
    local container="$STORE/aurora.sqsh"
    if [[ ! -f "$container" ]]; then
        echo "  SKIP: no container found"
        return 1
    fi

    local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"

    echo "  eval: $job_tag"

    sbatch --parsable \
        --job-name="eval_${job_tag}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --gres=gpu:0 \
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/eval_${job_tag}_%j.out" \
        --error="$LOG_DIR/eval_${job_tag}_%j.err" \
        --container-image="$container" \
        --container-mounts="$mounts" \
        --container-workdir="$WORKDIR" \
        --wrap="cd ${WORKDIR}/SwissClim_Evaluations && \
            pip install -e . --quiet && \
            python -m swissclim_evaluations.cli --config '${config_path}'"
}

# ---------------------------------------------------------------------------
# Evaluate all runs in a phase
# ---------------------------------------------------------------------------
run_eval_phase() {
    local phase=$1
    local filter_model="${2:-}"
    local count=0

    echo "=== Evaluating ${phase} ==="

    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local model_id="${MODEL_IDS[$model]}"
        local base="$STORE/ablation/${phase}/${model_id}"

        if [[ ! -d "$base" ]]; then
            echo "SKIP $model: no ${phase} outputs at $base"
            continue
        fi

        for init_dir in "$base"/*/; do
            [[ ! -d "$init_dir" ]] && continue
            local init_tag
            init_tag=$(basename "$init_dir")

            # Map init_tag back to init_time
            local init_time=""
            for it in "${INIT_TIMES[@]}"; do
                local tag="${it%%T*}"
                tag="${tag//-/}"
                if [[ "$tag" == "$init_tag" ]]; then
                    init_time="$it"
                    break
                fi
            done
            [[ -z "$init_time" ]] && continue

            for run_dir in "$init_dir"/*/; do
                [[ ! -d "$run_dir" ]] && continue
                local zarr_path="${run_dir}forecast.zarr"
                [[ ! -d "$zarr_path" ]] && continue

                local run_tag
                run_tag=$(basename "$run_dir")
                local job_tag="${phase}_${model}_${init_tag}_${run_tag}"

                submit_eval "$zarr_path" "$init_time" "$phase" "$job_tag" && ((count++)) || true
            done
        done
    done

    echo "${phase}: submitted $count evaluation jobs"
}

# ---------------------------------------------------------------------------
# Intercomparison: compare runs within meaningful groups.
#
# Phase 1 grouping:
#   - Per init_time: 5 magnitudes side-by-side (most readable)
#   - Per magnitude: 4 init_times side-by-side (seasonal stability)
#
# Phase 2 grouping:
#   - Per init_time: layer groups side-by-side
# ---------------------------------------------------------------------------
submit_intercompare_job() {
    local model=$1
    local out_dir=$2
    local job_tag=$3
    shift 3
    local path_args=("$@")

    if [[ ${#path_args[@]} -lt 2 ]]; then
        echo "  SKIP $job_tag: need >= 2 eval dirs, got ${#path_args[@]}"
        return 0
    fi
    if [[ -d "$out_dir" ]]; then
        echo "  SKIP $job_tag: output exists"
        return 0
    fi

    local container="$STORE/${model}.sqsh"
    local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"

    local cli_paths=""
    for d in "${path_args[@]}"; do
        cli_paths+=" '${d}'"
    done

    echo "  intercompare: $job_tag (${#path_args[@]} runs)"

    sbatch --parsable \
        --job-name="icmp_${job_tag}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --gres=gpu:0 \
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/icmp_${job_tag}_%j.out" \
        --error="$LOG_DIR/icmp_${job_tag}_%j.err" \
        --container-image="$container" \
        --container-mounts="$mounts" \
        --container-workdir="$WORKDIR" \
        --wrap="cd ${WORKDIR}/SwissClim_Evaluations && \
            pip install -e . --quiet && \
            python -m ai_models_ensembles.cli intercompare \
                ${cli_paths} \
                --out-dir '${out_dir}' \
                --module spectra --module hist --module metrics --module prob --module multivariate"
}

run_intercompare() {
    local phase=$1
    local filter_model="${2:-}"
    local count=0

    echo "=== Intercomparison for ${phase} ==="

    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local model_id="${MODEL_IDS[$model]}"
        local base="$STORE/ablation/${phase}/${model_id}"

        if [[ ! -d "$base" ]]; then
            echo "SKIP $model: no ${phase} outputs"
            continue
        fi

        # --- Group 1: Per init_time (compare magnitudes/layers) ---
        for init_dir in "$base"/*/; do
            [[ ! -d "$init_dir" ]] && continue
            local init_tag
            init_tag=$(basename "$init_dir")

            local dirs=()
            for run_dir in "$init_dir"/*/; do
                local eval_dir="${run_dir}swissclim_eval"
                [[ -d "$eval_dir" ]] && dirs+=("$eval_dir")
            done

            local out="$base/intercompare_by_init/${init_tag}"
            submit_intercompare_job "$model" "$out" \
                "${phase}_${model}_init_${init_tag}" "${dirs[@]}" && ((count++)) || true
        done

        # --- Group 2: Per magnitude/layer group (compare across init_times) ---
        # Collect unique run tags (e.g. mag_0.01_layer_all)
        local run_tags=()
        for run_dir in "$base"/*/*/; do
            [[ ! -d "$run_dir" ]] && continue
            local tag
            tag=$(basename "$run_dir")
            # Deduplicate
            local found=0
            for existing in "${run_tags[@]:-}"; do
                [[ "$existing" == "$tag" ]] && found=1 && break
            done
            [[ $found -eq 0 ]] && run_tags+=("$tag")
        done

        for run_tag in "${run_tags[@]}"; do
            local dirs=()
            for init_dir in "$base"/*/; do
                local eval_dir="${init_dir}${run_tag}/swissclim_eval"
                [[ -d "$eval_dir" ]] && dirs+=("$eval_dir")
            done

            local out="$base/intercompare_by_config/${run_tag}"
            submit_intercompare_job "$model" "$out" \
                "${phase}_${model}_cfg_${run_tag}" "${dirs[@]}" && ((count++)) || true
        done
    done

    echo "${phase}: submitted $count intercomparison jobs"
}

# ---------------------------------------------------------------------------

ACTION="${1:-}"
PHASE_OR_MODEL="${2:-}"
MODEL_FILTER="${3:-}"

case "$ACTION" in
    phase1|phase2|phase2b)
        run_eval_phase "$ACTION" "$PHASE_OR_MODEL"
        ;;
    intercompare)
        run_intercompare "$PHASE_OR_MODEL" "$MODEL_FILTER"
        ;;
    *)
        echo "Usage:"
        echo "  $0 {phase1|phase2|phase2b} [model]         # evaluate runs"
        echo "  $0 intercompare {phase1|phase2|phase2b} [model]  # compare runs"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER | grep eval_"
echo "Logs: $LOG_DIR/"
