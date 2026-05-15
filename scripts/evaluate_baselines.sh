#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate probabilistic baselines using SwissClim Evaluations.
#
# Models: atlas, fcn3, aifsens (AI ensembles) + ifs_ens (classical reference).
#
# Two configs per model:
#   * main: stride 6h (full daily cycle), all modules EXCEPT ets/fss
#   * etsfss: stride 24h, only ets/fss in members mode (per-member spread)
#
# Differences vs evaluate_ablation.sh:
#   * 112 init_times (8 weeks x 14 inits) instead of 4 mid-month dates
#   * lead max_hour=360 (vs 240) -- baselines run 15 days
#   * stride 6h for main eval (4x larger than ablation 24h, captures diurnal cycle)
#   * variables_2d includes 10m winds (saved by baselines but not ablation)
#   * IFS ENS treated as a 4th "model" alongside the AI baselines
#   * No run_tag subdirectory layer; eval_root = baselines/<m>/eval
#   * dask_profile=balanced (fewer, fatter workers for big lazy graphs)
#
# Usage:
#   bash scripts/evaluate_baselines.sh eval                    # main eval (4 jobs)
#   bash scripts/evaluate_baselines.sh etsfss                  # ets/fss eval (4 jobs)
#   bash scripts/evaluate_baselines.sh all                     # both (8 jobs)
#   bash scripts/evaluate_baselines.sh eval atlas              # single model
#   bash scripts/evaluate_baselines.sh intercompare            # 4-way intercomparison
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/baseline_eval_logs"

PARTITION="${PARTITION:-normal}"
TIME_LIMIT_MAIN="12:00:00"
TIME_LIMIT_ETSFSS="06:00:00"

WB2_PATHS='[
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr"
  ]'

WEEK_STARTS=(
    "2023-01-02" "2023-04-02" "2023-07-02" "2023-10-02"
    "2024-01-02" "2024-04-02" "2024-07-02" "2024-10-02"
)

INIT_TIMES=()
for ws in "${WEEK_STARTS[@]}"; do
    for day_offset in 0 1 2 3 4 5 6; do
        d=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${ws}') + timedelta(days=${day_offset})).strftime('%Y-%m-%d'))")
        INIT_TIMES+=("${d}T00:00")
        INIT_TIMES+=("${d}T12:00")
    done
done

MODELS="atlas fcn3 aifsens ifs_ens"
declare -A MODEL_KIND=( [atlas]=ai [fcn3]=ai [aifsens]=ai [ifs_ens]=ref )

IFS_ENS_ZARR="/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Build YAML lists for paths.ml and selection.datetimes.
# ---------------------------------------------------------------------------
_ml_yaml() {
    local model=$1
    local kind="${MODEL_KIND[$model]}"
    if [[ "$kind" == "ref" ]]; then
        echo "[\"${IFS_ENS_ZARR}\"]"
        return
    fi
    local out="["
    local first=1
    for it in "${INIT_TIMES[@]}"; do
        local tag="${it%%T*}"
        tag="${tag//-/}_${it##*T}"
        tag="${tag//:/}"
        local zp="$STORE/baselines/${model}/${tag}/forecast.zarr"
        if [[ -d "$zp" ]]; then
            [[ $first -eq 0 ]] && out+=", "
            out+="\"${zp}\""
            first=0
        fi
    done
    out+="]"
    echo "$out"
}

_dt_yaml() {
    local out="["
    local first=1
    for it in "${INIT_TIMES[@]}"; do
        local init_date="${it%%T*}"
        local hh="${it##*T}"
        local end_date
        end_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${init_date}T${hh}') + timedelta(hours=360)).strftime('%Y-%m-%dT%H'))")
        [[ $first -eq 0 ]] && out+=", "
        out+="\"${it}:${end_date}\""
        first=0
    done
    out+="]"
    echo "$out"
}

# ---------------------------------------------------------------------------
# Main config (stride 6h, all modules except ets/fss).
# ---------------------------------------------------------------------------
generate_main_config() {
    local model=$1
    local output_root=$2
    local ml_yaml dt_yaml
    ml_yaml=$(_ml_yaml "$model")
    dt_yaml=$(_dt_yaml)

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
    - "10m_u_component_of_wind"
    - "10m_v_component_of_wind"
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
    energy_spectra: pooled
    probabilistic: prob
    deterministic: mean
    multivariate: mean
    wd_kde: pooled

  check_missing: false

derived_variables:
  wind_speed:
    kind: wind_speed
    u: u_component_of_wind
    v: v_component_of_wind
  geopotential_height:
    kind: geopotential_height
    u: geopotential
  geopotential_height_gradient:
    kind: geopotential_height_gradient
    u: geopotential_height

plotting:
  random_seed: 42
  plot_datetime: null
  plot_ensemble_members: [0]
  dpi: 48
  output_mode: both
  histograms_per_lead: false
  maps_per_lead_grid: false
  energy_spectra_spectrogram: true
  probabilistic_line_plots: true

modules:
  maps: true
  histograms: false
  wd_kde: true
  energy_spectra: true
  vertical_profiles: false
  deterministic: true
  ets: false
  fss: false
  probabilistic: true
  ssim: true
  multivariate: true

lead_time:
  mode: stride
  stride:
    hours: 6
  max_hour: 360
  panel:
    strategy: evenly_spaced
    count: 5

metrics:
  deterministic:
    include: ["MAE", "RMSE", "Relative MAE"]
    standardized_include: ["MAE", "RMSE"]
    report_per_level: true
    error_maps: false

  multivariate:
    bivariate_pairs:
      - ["temperature", "specific_humidity"]
      - ["u_component_of_wind", "v_component_of_wind"]
      - ["geopotential_height_gradient", "wind_speed"]
    coriolis_parameter: 1.0e-4
    bins: 100

performance:
  dask_scheduler: distributed
  dask_profile: balanced
YAML
}

# ---------------------------------------------------------------------------
# ETS/FSS config (stride 24h, members mode).
# ---------------------------------------------------------------------------
generate_etsfss_config() {
    local model=$1
    local output_root=$2
    local ml_yaml dt_yaml
    ml_yaml=$(_ml_yaml "$model")
    dt_yaml=$(_dt_yaml)

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
    - "10m_u_component_of_wind"
    - "10m_v_component_of_wind"
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
    ets: members
    fss: members

  check_missing: false

plotting:
  random_seed: 42
  dpi: 48
  output_mode: both

modules:
  maps: false
  histograms: false
  wd_kde: false
  energy_spectra: false
  vertical_profiles: false
  deterministic: false
  ets: true
  fss: true
  probabilistic: false
  ssim: false
  multivariate: false

lead_time:
  mode: stride
  stride:
    hours: 24
  max_hour: 360

metrics:
  ets:
    thresholds: [75, 95]
    report_per_level: true
    aggregate_members_mean: true
  fss:
    thresholds: [75, 95]
    window_size: 9
    report_per_level: true
    aggregate_members_mean: true

performance:
  dask_scheduler: distributed
  dask_profile: balanced
YAML
}

# ---------------------------------------------------------------------------

submit_main_eval() {
    local model=$1
    local eval_root="$STORE/baselines/${model}/eval"
    local config_path="$STORE/baselines/${model}/eval_main_config.yaml"
    mkdir -p "$(dirname "$config_path")"
    generate_main_config "$model" "$eval_root" > "$config_path"

    local job_tag="eval_baseline_${model}_main"
    echo "  $job_tag"
    sbatch --parsable \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=144 \
        --mem=444G \
        --time="$TIME_LIMIT_MAIN" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m swissclim_evaluations.cli --config '${config_path}'"
}

submit_etsfss_eval() {
    local model=$1
    local eval_root="$STORE/baselines/${model}/eval"
    local config_path="$STORE/baselines/${model}/eval_etsfss_config.yaml"
    mkdir -p "$(dirname "$config_path")"
    generate_etsfss_config "$model" "$eval_root" > "$config_path"

    local job_tag="eval_baseline_${model}_etsfss"
    echo "  $job_tag"
    sbatch --parsable \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=144 \
        --mem=444G \
        --time="$TIME_LIMIT_ETSFSS" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m swissclim_evaluations.cli --config '${config_path}'"
}

run_eval() {
    local kind=$1
    local filter_model="${2:-}"
    echo "=== ${kind} eval (baselines) ==="
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        case "$kind" in
            main)   submit_main_eval "$model" ;;
            etsfss) submit_etsfss_eval "$model" ;;
            all)
                submit_main_eval "$model"
                submit_etsfss_eval "$model"
                ;;
        esac
    done
}

run_intercompare() {
    local after_job="${AFTER_JOB:-}"
    local out_dir="$STORE/baselines/intercomparison"
    if [[ -d "$out_dir" ]]; then
        echo "  SKIP: intercomparison output exists ($out_dir)"
        return 0
    fi
    local cli_paths=""
    for model in $MODELS; do
        local eval_root="$STORE/baselines/${model}/eval"
        if [[ ! -d "$eval_root" ]]; then
            echo "  SKIP ${model}: no eval output at $eval_root"
            continue
        fi
        cli_paths+=" '${eval_root}'"
    done
    if [[ -z "$cli_paths" ]]; then
        echo "No model eval outputs found. Aborting."
        return 1
    fi
    local job_tag="icmp_baselines"
    local dep_flag=()
    [[ -n "$after_job" ]] && dep_flag=(--dependency="afterany:${after_job}")
    echo "  $job_tag -> $out_dir"
    sbatch --parsable \
        "${dep_flag[@]}" \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --gres=gpu:4 \
        --time="06:00:00" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m ai_models_ensembles.cli intercompare \
                ${cli_paths} \
                --out-dir '${out_dir}' \
                --module spectra --module kde --module metrics --module prob --module multivariate --module ets --module fss --module ssim --module maps"
}

# ---------------------------------------------------------------------------

ACTION="${1:-eval}"
shift || true

case "$ACTION" in
    eval|main)     run_eval main "${1:-}" ;;
    etsfss)        run_eval etsfss "${1:-}" ;;
    all)           run_eval all "${1:-}" ;;
    atlas|fcn3|aifsens|ifs_ens) run_eval all "$ACTION" ;;
    intercompare)  run_intercompare ;;
    *)
        echo "Usage:"
        echo "  $0 eval [model]         # stride 6h main eval (default)"
        echo "  $0 etsfss [model]       # stride 24h ETS/FSS members eval"
        echo "  $0 all [model]          # both main + etsfss"
        echo "  $0 intercompare         # 4-way intercomparison"
        exit 1
        ;;
esac

echo ""
echo "Logs: $LOG_DIR/"
