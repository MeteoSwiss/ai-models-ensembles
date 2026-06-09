#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate probabilistic baselines using SwissClim Evaluations.
#
# Models: atlas, fcn3, aifsens (AI ensembles), ifs_ens (classical reference),
# and the post-hoc perturbation baselines aurora_encoder, graphcast_all,
# sfno_modes10. All match the same 112 init_times (8 weeks x 14 inits/week).
#
# Submission strategy (since 2026-05-27): always per-module. The bundled
# "main" eval at 112 inits exceeds the 12 h walltime for every model, so
# `bash evaluate_baselines.sh all <model>` submits one sbatch per (model,
# module) -- see modules in `submit_per_module_eval`. This is faster, more
# robust, and lets each module use --mem=800G --time=12:00:00 (cluster max).
#
# Differences vs evaluate_ablation.sh:
#   * 112 init_times (vs 4 mid-month dates)
#   * lead max_hour=360 (vs 240) -- baselines run 15 days
#   * stride 6h for main config (vs 24h ablation)
#   * variables_2d includes 10m winds (saved by baselines but not ablation)
#   * IFS ENS treated as a 5th "model"
#   * No run_tag subdirectory; eval_root = baselines/<m>/eval
#   * dask_profile=balanced (fewer, fatter workers for big lazy graphs)
#
# Usage:
#   bash scripts/evaluate_baselines.sh all [model]                # per-module submission (default)
#   bash scripts/evaluate_baselines.sh gen-configs <model>        # write YAML templates only
#   bash scripts/evaluate_baselines.sh intercompare               # cross-model intercomp
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/baseline_eval_logs"

PARTITION="${PARTITION:-normal}"
TIME_LIMIT_MAIN="12:00:00"
TIME_LIMIT_ETSFSS="12:00:00"

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

MODELS="atlas fcn3 aifsens ifs_ens sfno_modes10 aurora_encoder graphcast_all aifs_perturbed aifs_perturbed_ic aurora_encoder_ic graphcast_all_ic sfno_modes10_ic atmllm"
declare -A MODEL_KIND=(
    [atlas]=ai [fcn3]=ai [aifsens]=ai [ifs_ens]=ref
    [sfno_modes10]=ai [aurora_encoder]=ai [graphcast_all]=ai
    [aifs_perturbed]=ai [aifs_perturbed_ic]=ai
    [aurora_encoder_ic]=ai [graphcast_all_ic]=ai [sfno_modes10_ic]=ai
    [atmllm]=combined
)

IFS_ENS_ZARR="/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr"
ATMLLM_ZARR="/capstor/store/cscs/swissai/a122/lhuang/outputs/AtmLLM_inference_results/atmllm-evals-112initsteps-combined.zarr"

# Stratified 10-member subsample of IFS ENS (50 members total). Members are
# generated with systematic perturbation spread across indices, so step-5
# sampling preserves the physical diversity while matching the AI ensemble size.
IFS_ENS_MEMBERS="[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]"

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
    if [[ "$kind" == "combined" ]]; then
        # External combined-zarr baselines (one zarr with init_time dim,
        # not a per-init forecast.zarr layout). Currently only AtmLLM.
        case "$model" in
            atmllm) echo "[\"${ATMLLM_ZARR}\"]" ;;
            *) echo "ERROR: kind=combined for unknown model '$model'" >&2; exit 1 ;;
        esac
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
    local ml_yaml dt_yaml members_yaml
    ml_yaml=$(_ml_yaml "$model")
    dt_yaml=$(_dt_yaml)
    members_yaml="null"
    [[ "${MODEL_KIND[$model]}" == "ref" ]] && members_yaml="${IFS_ENS_MEMBERS}"

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

  ensemble_members: ${members_yaml}

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
    local ml_yaml dt_yaml members_yaml
    ml_yaml=$(_ml_yaml "$model")
    dt_yaml=$(_dt_yaml)
    members_yaml="null"
    [[ "${MODEL_KIND[$model]}" == "ref" ]] && members_yaml="${IFS_ENS_MEMBERS}"

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

  ensemble_members: ${members_yaml}

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
  ets: false   # ETS dropped from baseline eval -- not load-bearing for paper, can't fit 112-init in walltime
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
    local dep_flag=()
    [[ -n "${AFTER_JOB:-}" ]] && dep_flag=(--dependency="afterany:${AFTER_JOB}")
    echo "  $job_tag" >&2
    sbatch --parsable \
        "${dep_flag[@]}" \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=144 \
        --mem=800G \
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
    local dep_flag=()
    [[ -n "${AFTER_JOB:-}" ]] && dep_flag=(--dependency="afterany:${AFTER_JOB}")
    echo "  $job_tag" >&2
    sbatch --parsable \
        "${dep_flag[@]}" \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=144 \
        --mem=800G \
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

# ---------------------------------------------------------------------------
# Per-module submission (the default for `all`). One sbatch per (model, module);
# always with --mem=800G --time=12:00:00 cluster max. Always submitted modules:
#   maps, wd_kde, energy_spectra, deterministic, ssim, multivariate, probabilistic, fss
# ETS dropped 2026-05-27 -- not load-bearing for the paper, walltime issues.
# det + ssim used to be bundled (light bundle, parallel streams) but the
# 2-stream version OOM'd at 800G on aurora_encoder + graphcast_all
# (2026-05-28); now each module gets its own per-model sbatch.
# ---------------------------------------------------------------------------
PY="$SRC_DIR/.venv/bin/python3"

# Generate per-module config YAML from the main template; returns path.
_gen_module_subset_config() {
    local model=$1 module=$2 template=$3   # template: "main" or "etsfss"
    local src_name="eval_main_config.yaml"
    [[ "$template" == "etsfss" ]] && src_name="eval_etsfss_config.yaml"
    local src="$STORE/baselines/${model}/${src_name}"
    local dst="$STORE/baselines/${model}/eval_${module}_only_config.yaml"
    $PY -c "
import yaml
with open('$src') as f: c = yaml.safe_load(f)
all_off = {k: False for k in c['modules']}
all_off['${module}'] = True
c['modules'] = all_off
with open('$dst', 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
print('$dst')
"
}

_submit_module_sbatch() {
    local job_tag=$1 cfg=$2
    local dep_flag=()
    [[ -n "${AFTER_JOB:-}" ]] && dep_flag=(--dependency="afterany:${AFTER_JOB}")
    sbatch --parsable \
        "${dep_flag[@]}" \
        --job-name="$job_tag" \
        --partition="$PARTITION" --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=144 --mem=800G --time="12:00:00" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m swissclim_evaluations.cli --config '${cfg}'"
}

submit_per_module_eval() {
    local filter_model="${1:-}"
    echo "=== per-module eval (baselines) ==="
    # Ensure template YAMLs exist for every model we're about to evaluate
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        [[ -f "$STORE/baselines/${model}/eval_main_config.yaml" ]] && continue
        echo "  gen-configs $model (templates missing)"
        gen_configs "$model"
    done

    # Single-module jobs (main template, stride 6h). deterministic + ssim used
    # to share a "light bundle" with parallel streams per model, but that
    # pattern OOM'd at 800G when >=2 models ran concurrently (e.g. aurora +
    # graphcast 2026-05-28). Each module now runs as its own per-model sbatch.
    for module in maps wd_kde energy_spectra multivariate probabilistic deterministic ssim; do
        echo "  -- ${module} (per model) --"
        for model in $MODELS; do
            [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
            local cfg
            cfg=$(_gen_module_subset_config "$model" "$module" "main")
            _submit_module_sbatch "eval_baseline_${model}_${module}" "$cfg"
        done
    done

    # FSS (members mode, stride 24h, etsfss template)
    echo "  -- fss (members mode, per model) --"
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local cfg
        cfg=$(_gen_module_subset_config "$model" "fss" "etsfss")
        _submit_module_sbatch "eval_baseline_${model}_fss" "$cfg"
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
        --mem=800G \
        --gres=gpu:4 \
        --time="12:00:00" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m ai_models_ensembles.cli intercompare \
                ${cli_paths} \
                --out-dir '${out_dir}' \
                --label-from parent \
                --module spectra --module kde --module metrics --module prob --module multivariate --module fss --module ssim --module maps"
}

# ---------------------------------------------------------------------------

gen_configs() {
    # Generate eval_main_config.yaml + eval_etsfss_config.yaml for a model
    # without submitting any sbatch.
    local model=$1
    local eval_root="$STORE/baselines/${model}/eval"
    mkdir -p "$STORE/baselines/${model}"
    generate_main_config   "$model" "$eval_root" > "$STORE/baselines/${model}/eval_main_config.yaml"
    generate_etsfss_config "$model" "$eval_root" > "$STORE/baselines/${model}/eval_etsfss_config.yaml"
    echo "  wrote eval_main_config.yaml + eval_etsfss_config.yaml for $model"
}

ACTION="${1:-all}"
shift || true

case "$ACTION" in
    all)           submit_per_module_eval "${1:-}" ;;
    atlas|fcn3|aifsens|ifs_ens|sfno_modes10|aurora_encoder|graphcast_all|aifs_perturbed|aifs_perturbed_ic|aurora_encoder_ic|graphcast_all_ic|sfno_modes10_ic|atmllm) submit_per_module_eval "$ACTION" ;;
    intercompare)  run_intercompare ;;
    gen-configs)
        target_model="${1:-}"
        if [[ -z "$target_model" ]]; then
            echo "Usage: $0 gen-configs <model>"; exit 1
        fi
        gen_configs "$target_model"
        ;;
    bundled-main)
        # Legacy single-bundled main eval (will TIMEOUT at 112 inits; kept as
        # an escape hatch for debugging or non-standard grids).
        run_eval main "${1:-}"
        ;;
    bundled-etsfss)
        # Legacy single-bundled etsfss eval (ETS dropped, FSS now per-module).
        run_eval etsfss "${1:-}"
        ;;
    *)
        echo "Usage:"
        echo "  $0 all [model]          # per-module submission (default; 7 sbatch jobs)"
        echo "  $0 intercompare         # cross-model intercomparison"
        echo "  $0 gen-configs <model>  # write eval YAML templates only"
        echo "  $0 bundled-main [model] # legacy single bundled main eval (will TIMEOUT)"
        echo "  $0 bundled-etsfss [model] # legacy single bundled etsfss eval"
        exit 1
        ;;
esac

echo ""
echo "Logs: $LOG_DIR/"
