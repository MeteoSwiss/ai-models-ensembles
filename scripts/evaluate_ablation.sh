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

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/ablation_logs"
WORKDIR=/workspace/ai-models-ensembles

PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"

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

MODELS="aurora graphcast sfno aifs"
declare -A MODEL_IDS=( [aurora]=aurora [graphcast]=graphcast_operational [sfno]=sfno [aifs]=aifs )

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
    energy_spectra: pooled
    probabilistic: prob
    deterministic: mean
    multivariate: mean
    wd_kde: pooled
    ets: members
    fss: members

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
  ets: true
  fss: true
  probabilistic: true
  ssim: true
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
      - ["geopotential_height_gradient", "wind_speed"]
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

    # Partial init sets produce numbers incomparable with the 4-init design grid.
    if [[ ${#zarr_paths[@]} -lt ${#INIT_TIMES[@]} ]]; then
        if [[ "${ALLOW_PARTIAL_INITS:-0}" != "1" ]]; then
            echo "  SKIP ${model} ${run_tag}: only ${#zarr_paths[@]}/${#INIT_TIMES[@]} init_times have forecast.zarr (ALLOW_PARTIAL_INITS=1 to override)"
            return 0
        fi
        echo "  WARN ${model} ${run_tag}: evaluating partial ${#zarr_paths[@]}/${#INIT_TIMES[@]} init_times"
    fi

    local config_path="$base/eval/${run_tag}_config.yaml"
    mkdir -p "$(dirname "$config_path")"
    generate_config "$eval_root" "${zarr_paths[@]}" > "$config_path"

    local job_tag="${phase}_${model}_${run_tag}"

    echo "  eval: $job_tag (${#zarr_paths[@]} init_times)"

    local after_job="${AFTER_JOB:-}"
    local dep_flag=()
    [[ -n "$after_job" ]] && dep_flag=(--dependency="afterany:${after_job}")

    sbatch --parsable \
        "${dep_flag[@]}" \
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
# Intercomparison: 2 sbatch jobs per model, both writing to the same
# intercomparison/ output dir.
#
#   Job A (non-prob, WITH mag_0):
#       compares all run_tags including the unperturbed reference
#       (mag_0_layer_all, N=1). Modules: spectra, kde, metrics, multivariate,
#       ets, fss, ssim, maps.
#
#   Job B (prob-only, WITHOUT mag_0):
#       compares only the N>=2 runs (perturbed mags). Module: prob.
#       Required because SwissClim's `common_files()` intersects across
#       every supplied path; mag_0 has no `probabilistic/` subdir, so
#       including it would zero out the intersection and skip CRPS maps,
#       PIT histograms, spaghetti plots, etc.
# ---------------------------------------------------------------------------
NONPROB_MODULES="spectra kde metrics multivariate ets fss ssim maps"
PROB_MODULES="prob"

# Best Phase 2 / 2c layer-group run_tag per model. Used by run_intercompare for
# phase3 and phase3b to pull the Phase 2 winner as a context panel.
#
# IMPORTANT: PHASE2_BEST_LOC tells the script which sub-phase to source the
# tag from (phase2 sqrt-N anchor vs phase2b sigma-sweep refinement). Aurora's
# actual best is Phase 2b sigma=0.025; SFNO's actual best is the Phase 2
# sqrt-N anchor; GraphCast's overall best is Phase 1 mag_0.01_layer_all
# (which is already pulled as a reference panel for every phase2+ intercomp,
# so PHASE2_BEST_TAG[graphcast] still points at the Phase 2 m2g sqrt-N anchor
# for cross-phase context).
declare -A PHASE2_BEST_TAG=(
    [aurora]=mag_0.025_layer_encoder
    [graphcast]=mag_0.029665_layer_m2g
    [sfno]=mag_0.053852_layer_encoder
)
declare -A PHASE2_BEST_LOC=(
    [aurora]=phase2b
    [graphcast]=phase2
    [sfno]=phase2
)

# Phase 3 sqrt(N) baseline run_tag per model -- the threshold-1 (~3000-5000 km
# wavelength) anchor at variance-budget sigma. Used by run_intercompare for
# phase3b to include it as the leftmost Phase 3 reference panel.
declare -A PHASE3_SQRTN_TAG=(
    [aurora]=mag_0.025900_layer_unet_bottom
    [graphcast]=gcsigma_0.312_gcnodes42
    [sfno]=mag_0.049261_modes10
)

# Per-(model, phase) argmin-CRPS winner at lead 240h (own-phase runs only).
# Used by `run_allphases_intercompare` to build a single cross-phase view.
# Updated 2026-05-27 after the full ablation; see [[calibration-winners]].
declare -A ALLPHASES_PHASE1=(
    [aurora]=mag_0.03_layer_all
    [graphcast]=mag_0.01_layer_all
    [sfno]=mag_0.03_layer_all
    [aifs]=mag_0.01_layer_all
)
declare -A ALLPHASES_PHASE2=(
    [aurora]=mag_0.044176_layer_encoder
    [graphcast]=mag_0.029665_layer_m2g
    [sfno]=mag_0.053852_layer_encoder
    [aifs]=mag_0.027500_layer_decoder
)
declare -A ALLPHASES_PHASE2B=(
    [aurora]=mag_0.025_layer_encoder
    [graphcast]=mag_0.014_layer_g2m
    [sfno]=mag_0.053852_layer_encoder
    [aifs]=""
)
declare -A ALLPHASES_PHASE3=(
    [aurora]=mag_0.40_layer_unet_bottom
    [graphcast]=gcsigma_1.00_gcnodes42
    [sfno]=mag_0.25_modes10
    [aifs]=""
)
declare -A ALLPHASES_PHASE3B=(
    [aurora]=mag_0.015_layer_enc_012
    [graphcast]=gcsigma_0.159_gcnodes162
    [sfno]=mag_0.035_modes20
    [aifs]=""
)

_submit_intercompare_sbatch() {
    local job_tag=$1 out_dir=$2 cli_paths=$3 modules=$4 after_job=$5
    local module_flags=""
    for m in $modules; do
        module_flags+=" --module $m"
    done
    local dep_flag=()
    [[ -n "$after_job" ]] && dep_flag=(--dependency="afterany:${after_job}")

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
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m ai_models_ensembles.cli intercompare \
                ${cli_paths} \
                --out-dir '${out_dir}' \
                ${module_flags}"
}

run_intercompare() {
    local phase=$1
    local filter_model="${2:-}"
    local count=0
    local after_job="${AFTER_JOB:-}"

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

        # Build two path lists: with mag_0 (for non-prob) and without (for prob).
        # Reference panels for phase2+:
        #   * mag_0_layer_all   (unperturbed, N=1)   -- pulled from phase1/eval/
        #   * mag_0.01_layer_all (Phase 1 best, N=10) -- pulled from phase1/eval/
        #   * Phase 2 winner per PHASE2_BEST_TAG       -- pulled from phase2/eval/ when phase=phase3
        local mag0_dir_self="$eval_base/mag_0_layer_all/"
        local p1_eval="$STORE/ablation/phase1/${model_id}/eval"
        local p2_loc="${PHASE2_BEST_LOC[$model]:-phase2}"
        local p2_eval="$STORE/ablation/${p2_loc}/${model_id}/eval"
        local mag0_dir_p1="$p1_eval/mag_0_layer_all/"
        local mag001_dir_p1="$p1_eval/mag_0.01_layer_all/"
        local mag0_dir=""
        if [[ -d "$mag0_dir_self" ]]; then
            mag0_dir="$mag0_dir_self"
        elif [[ -d "$mag0_dir_p1" ]]; then
            mag0_dir="$mag0_dir_p1"
        fi
        local with_mag0=()
        local without_mag0=()
        [[ -n "$mag0_dir" ]] && with_mag0+=("$mag0_dir")
        if [[ "$phase" != "phase1" && -d "$mag001_dir_p1" ]]; then
            with_mag0+=("$mag001_dir_p1")
            without_mag0+=("$mag001_dir_p1")
        fi
        if [[ "$phase" == "phase3" || "$phase" == "phase3b" ]]; then
            local p2_best="${PHASE2_BEST_TAG[$model]:-}"
            if [[ -n "$p2_best" && -d "$p2_eval/${p2_best}/" ]]; then
                with_mag0+=("$p2_eval/${p2_best}/")
                without_mag0+=("$p2_eval/${p2_best}/")
            elif [[ -n "$p2_best" ]]; then
                echo "  WARN $model: PHASE2_BEST_TAG '${p2_best}' not found at $p2_eval"
            fi
        fi
        # phase2b: sigma sweep around Phase 2 winners. Pull ALL Phase 2 eval
        # dirs (the sqrt(N) baselines) so the intercomp shows the existing
        # anchor + the new sigmas side-by-side. Always pulls from the literal
        # phase2/ directory regardless of PHASE2_BEST_LOC (which may point at
        # phase2b for a model whose Phase 2b winner is the "best Phase 2").
        local p2_eval_literal="$STORE/ablation/phase2/${model_id}/eval"
        if [[ "$phase" == "phase2b" && -d "$p2_eval_literal" ]]; then
            for d in "$p2_eval_literal"/*/; do
                [[ -d "$d" ]] || continue
                with_mag0+=("$d")
                without_mag0+=("$d")
            done
        fi
        # phase3b also reaches into phase3/ for the sqrt(N) baseline anchor
        # (threshold-1 entry at variance-budget sigma) so the threshold sweep
        # is intercompared against its own first threshold row.
        if [[ "$phase" == "phase3b" ]]; then
            local p3_sqrtn="${PHASE3_SQRTN_TAG[$model]:-}"
            local p3_eval="$STORE/ablation/phase3/${model_id}/eval"
            if [[ -n "$p3_sqrtn" && -d "$p3_eval/${p3_sqrtn}/" ]]; then
                with_mag0+=("$p3_eval/${p3_sqrtn}/")
                without_mag0+=("$p3_eval/${p3_sqrtn}/")
            elif [[ -n "$p3_sqrtn" ]]; then
                echo "  WARN $model: PHASE3_SQRTN_TAG '${p3_sqrtn}' not found at $p3_eval"
            fi
        fi
        for d in "$eval_base"/*/; do
            [[ -d "$d" ]] || continue
            [[ "$d" == "$mag0_dir_self" ]] && continue
            with_mag0+=("$d")
            without_mag0+=("$d")
        done

        if [[ ${#with_mag0[@]} -lt 2 ]]; then
            echo "SKIP $model: need >= 2 eval dirs, got ${#with_mag0[@]}"
            continue
        fi

        local out_dir="$base/intercomparison"
        if [[ -d "$out_dir" ]]; then
            echo "  SKIP $model: intercomparison output exists"
            continue
        fi

        local paths_with=""
        for d in "${with_mag0[@]}"; do
            paths_with+=" '${d}'"
        done
        local paths_without=""
        for d in "${without_mag0[@]}"; do
            paths_without+=" '${d}'"
        done

        echo "  intercompare $model: A=non-prob (${#with_mag0[@]} dirs incl. mag_0), B=prob (${#without_mag0[@]} dirs no mag_0) -> $out_dir"

        # Job A: non-probabilistic modules, WITH mag_0
        _submit_intercompare_sbatch \
            "icmp_${phase}_${model}_nonprob" \
            "$out_dir" "$paths_with" "$NONPROB_MODULES" "$after_job"
        count=$((count + 1))

        # Job B: probabilistic only, WITHOUT mag_0 (skipped if not enough perturbed runs)
        if [[ ${#without_mag0[@]} -ge 2 ]]; then
            _submit_intercompare_sbatch \
                "icmp_${phase}_${model}_prob" \
                "$out_dir" "$paths_without" "$PROB_MODULES" "$after_job"
            count=$((count + 1))
        else
            echo "  SKIP $model prob intercomp: need >= 2 perturbed eval dirs (mag_0 excluded)"
        fi
    done

    echo "${phase}: submitted $count intercomparison jobs"
}

# ---------------------------------------------------------------------------
# run_allphases_intercompare
#   Final cross-phase summary. One intercomp per model showing the per-phase
#   argmin-CRPS winner (see ALLPHASES_PHASE{1,2,2c,3,3b} above). Writes to
#   $STORE/ablation/allphases/<model_id>/intercomparison/. Two sbatch per
#   model (non-prob + prob), same split pattern as run_intercompare.
# ---------------------------------------------------------------------------
run_allphases_intercompare() {
    local filter_model="${1:-}"
    local after_job="${AFTER_JOB:-}"
    local count=0

    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local model_id="${MODEL_IDS[$model]}"

        # Source eval dirs for the per-phase winners
        declare -A srcs=(
            [phase1]="${ALLPHASES_PHASE1[$model]}"
            [phase2]="${ALLPHASES_PHASE2[$model]}"
            [phase2b]="${ALLPHASES_PHASE2B[$model]}"
            [phase3]="${ALLPHASES_PHASE3[$model]}"
            [phase3b]="${ALLPHASES_PHASE3B[$model]}"
        )

        local with_mag0=()
        local without_mag0=()
        # mag_0 reference (unperturbed) from phase1
        local mag0_p1="$STORE/ablation/phase1/${model_id}/eval/mag_0_layer_all/"
        [[ -d "$mag0_p1" ]] && with_mag0+=("$mag0_p1")

        # Add per-phase winners
        for phase in phase1 phase2 phase2b phase3 phase3b; do
            local tag="${srcs[$phase]}"
            [[ -z "$tag" ]] && continue
            local d="$STORE/ablation/${phase}/${model_id}/eval/${tag}/"
            if [[ ! -d "$d" ]]; then
                echo "  WARN $model: ${phase} winner '${tag}' not found at $d"
                continue
            fi
            with_mag0+=("$d")
            without_mag0+=("$d")
        done

        if [[ ${#without_mag0[@]} -lt 2 ]]; then
            echo "SKIP $model: need >= 2 perturbed eval dirs, got ${#without_mag0[@]}"
            continue
        fi

        local out_dir="$STORE/ablation/allphases/${model_id}/intercomparison"
        if [[ -d "$out_dir" ]]; then
            echo "  SKIP $model: $out_dir exists -- delete to rerun"
            continue
        fi

        local paths_with=""
        for d in "${with_mag0[@]}"; do paths_with+=" '${d}'"; done
        local paths_without=""
        for d in "${without_mag0[@]}"; do paths_without+=" '${d}'"; done

        echo "  allphases intercompare $model: ${#with_mag0[@]} dirs (incl. mag_0), ${#without_mag0[@]} perturbed -> $out_dir"

        _submit_intercompare_sbatch \
            "icmp_allphases_${model}_nonprob" \
            "$out_dir" "$paths_with" "$NONPROB_MODULES" "$after_job"
        count=$((count + 1))

        _submit_intercompare_sbatch \
            "icmp_allphases_${model}_prob" \
            "$out_dir" "$paths_without" "$PROB_MODULES" "$after_job"
        count=$((count + 1))
    done

    echo "allphases: submitted $count intercomparison jobs"
}

# ---------------------------------------------------------------------------

ACTION="${1:-}"
PHASE_OR_MODEL="${2:-}"
MODEL_FILTER="${3:-}"

case "$ACTION" in
    phase1|phase2|phase2b|phase3|phase3b)
        run_eval_phase "$ACTION" "$PHASE_OR_MODEL"
        ;;
    intercompare)
        run_intercompare "$PHASE_OR_MODEL" "$MODEL_FILTER"
        ;;
    allphases_intercompare)
        run_allphases_intercompare "$PHASE_OR_MODEL"
        ;;
    *)
        echo "Usage:"
        echo "  $0 {phase1|phase2|phase2b|phase3|phase3b} [model]                # evaluate runs"
        echo "  $0 intercompare {phase1|phase2|phase2b|phase3|phase3b} [model]   # per-phase intercomp"
        echo "  $0 allphases_intercompare [model]                                        # final cross-phase summary"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER | grep eval_"
echo "Logs: $LOG_DIR/"
