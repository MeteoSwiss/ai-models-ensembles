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
TIME_LIMIT="06:00:00"

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
# Generate a SwissClim Evaluations YAML config for one (model, config) combo
# across all init_times. paths.ml is a list of forecast.zarr paths.
# ---------------------------------------------------------------------------
generate_config() {
    local output_root=$1
    shift
    local zarr_paths=("$@")

    # Build YAML list for paths.ml
    local ml_yaml="["
    local first=1
    for zp in "${zarr_paths[@]}"; do
        [[ $first -eq 0 ]] && ml_yaml+=", "
        ml_yaml+="\"${zp}\""
        first=0
    done
    ml_yaml+="]"

    # Build datetime ranges for each init_time
    local dt_yaml="["
    first=1
    for it in "${INIT_TIMES[@]}"; do
        local init_date="${it%%T*}"
        local end_date
        end_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${init_date}') + timedelta(days=10)).strftime('%Y-%m-%dT23'))")
        [[ $first -eq 0 ]] && dt_yaml+=", "
        dt_yaml+="\"${it}:${end_date}\""
        first=0
    done
    dt_yaml+="]"

    cat <<YAML
paths:
  ml: ${ml_yaml}
  nwp: ${WB2_PATHS}
  output_root: "${output_root}"

selection:
  levels: [500, 850]
  temporal_resolution_hours: null
  datetimes: ${dt_yaml}
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
    ets: members
    fss: members

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
  ets: true
  fss: true
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
  ets:
    thresholds: [75, 95]
    report_per_level: true
    aggregate_members_mean: true

  fss:
    thresholds: [75, 95]
    window_size: 9
    report_per_level: true
    aggregate_members_mean: true

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
# Evaluate one (model, run_tag) across all init_times combined.
# ---------------------------------------------------------------------------
submit_eval() {
    local model=$1
    local model_id=$2
    local run_tag=$3
    local phase=$4
    local base=$5

    local eval_root="$base/eval/${run_tag}"

    # Skip if already evaluated
    if [[ -d "$eval_root" ]]; then
        echo "  SKIP ${model} ${run_tag}: eval output exists"
        return 0
    fi

    # Collect forecast.zarr paths across all init_times
    local zarr_paths=()
    for it in "${INIT_TIMES[@]}"; do
        local init_tag="${it%%T*}"
        init_tag="${init_tag//-/}"
        local zp="$base/${init_tag}/${run_tag}/forecast.zarr"
        if [[ -d "$zp" ]]; then
            zarr_paths+=("$zp")
        fi
    done

    if [[ ${#zarr_paths[@]} -eq 0 ]]; then
        echo "  SKIP ${model} ${run_tag}: no forecast.zarr found"
        return 0
    fi

    local config_path="$base/eval/${run_tag}_config.yaml"
    mkdir -p "$(dirname "$config_path")"
    generate_config "$eval_root" "${zarr_paths[@]}" > "$config_path"

    local job_tag="${phase}_${model}_${run_tag}"

    echo "  eval: $job_tag (${#zarr_paths[@]} init_times)"

    sbatch --parsable \
        --job-name="eval_${job_tag}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/eval_${job_tag}_%j.out" \
        --error="$LOG_DIR/eval_${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m swissclim_evaluations.cli --config '${config_path}'"
}

# ---------------------------------------------------------------------------
# Evaluate all (model, config) combos in a phase.
# Each eval job combines all 4 init_times for one (model, magnitude/layer).
# Phase 1: 5 magnitudes x 3 models = 15 eval jobs
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

        # Discover unique run_tags (e.g. mag_0.01_layer_all) across init_times
        local run_tags=()
        for run_dir in "$base"/*/*/; do
            [[ ! -d "$run_dir" ]] && continue
            local tag
            tag=$(basename "$run_dir")
            local found=0
            for existing in "${run_tags[@]:-}"; do
                [[ "$existing" == "$tag" ]] && found=1 && break
            done
            [[ $found -eq 0 ]] && run_tags+=("$tag")
        done

        for run_tag in "${run_tags[@]}"; do
            submit_eval "$model" "$model_id" "$run_tag" "$phase" "$base" && ((count++)) || true
        done
    done

    echo "${phase}: submitted $count evaluation jobs"
}

# ---------------------------------------------------------------------------
# Intercomparison: 1 job per model, comparing all eval outputs (e.g. 5
# magnitudes for Phase 1). Labels are derived from run_tag.
# ---------------------------------------------------------------------------
run_intercompare() {
    local phase=$1
    local filter_model="${2:-}"
    local count=0

    echo "=== Intercomparison for ${phase} ==="

    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local model_id="${MODEL_IDS[$model]}"
        local base="$STORE/ablation/${phase}/${model_id}"
        local eval_base="$base/eval"

        if [[ ! -d "$eval_base" ]]; then
            echo "SKIP $model: no eval outputs at $eval_base"
            continue
        fi

        local eval_dirs=()
        for d in "$eval_base"/*/; do
            [[ -d "$d" ]] && eval_dirs+=("$d")
        done

        if [[ ${#eval_dirs[@]} -lt 2 ]]; then
            echo "SKIP $model: need >= 2 eval dirs, got ${#eval_dirs[@]}"
            continue
        fi

        local out_dir="$base/intercomparison"
        if [[ -d "$out_dir" ]]; then
            echo "  SKIP $model: intercomparison output exists"
            continue
        fi

        local cli_paths=""
        for d in "${eval_dirs[@]}"; do
            cli_paths+=" '${d}'"
        done

        local job_tag="icmp_${phase}_${model}"

        echo "  intercompare $model: ${#eval_dirs[@]} configs -> $out_dir"

        sbatch --parsable \
            --job-name="$job_tag" \
            --partition="$PARTITION" \
            --account=a122 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=16 \
            --mem=128G \
            --gres=gpu:4 \
            --time="$TIME_LIMIT" \
            --output="$LOG_DIR/${job_tag}_%j.out" \
            --error="$LOG_DIR/${job_tag}_%j.err" \
            --wrap="source ${SRC_DIR}/.venv/bin/activate && \
                python -m ai_models_ensembles.cli intercompare \
                    ${cli_paths} \
                    --out-dir '${out_dir}' \
                    --module spectra --module hist --module metrics --module prob --module multivariate --module ets --module fss"

        ((count++))
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
