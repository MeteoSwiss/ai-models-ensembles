#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Recompute the geostrophic-balance bivariate histogram (|grad Z| vs wind
# speed, level 500) over the HURRICANE MILTON DOMAIN ONLY for appendix Fig C2
# (figures/bivariate_geostrophic_500hPa.pdf).
#
# Why a fresh eval: the cached GLOBAL multivariate npz under
#   $STORE/baselines/<model>/eval/multivariate/
# cannot be subset to the Milton box (no lat/lon dim is stored in the npz, only
# the 2D histogram). The swissclim multivariate module honours a bounding box
# via selection.latitudes / selection.longitudes (core/data_selection.py applies
# ds.sel BEFORE histogramming), so we re-run it over the box.
#
# Milton box (0-360 lon convention, matching the zarr; see
# tools/milton/era5_control.py:27):  LON 240-300, LAT 5-40.
#
# Scope-minimised: only the geostrophic bivariate pair
# (geopotential_height_gradient vs wind_speed) is computed, only variables_3d
# needed for it (geopotential, u, v) are loaded, and only the multivariate
# module is enabled. Aggregates over ALL lead times (mode=stride, max 360h),
# exactly like the current figure.
#
# Distinct output_root keeps the global Tq npz untouched:
#   $STORE/baselines/<model>/eval_milton/multivariate/
#     bivariate_geopotential_height_gradient_wind_speed_level500_enspooled.npz
#
# Pooled members (2026-07-07): ensemble mode is "pooled" with the 1e6 paired
# subsample (bivariate_max_samples: auto), matching Fig C2's caption, and the
# geostrophic reference line uses f at the box-centre latitude 22.5N
# (f = 2*Omega*sin(22.5) = 5.58e-5 s^-1), NOT the earlier 1.0e-4 (=f at 43N,
# wrong for a 5-40N subtropical box).
#
# All 8 baselines used by the figure run inside ONE sbatch (the box is tiny:
# 240x140 grid pts, so each model is cheap).
#
# Usage:  bash tools/submit_geostrophic_milton.sh
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$SRC_DIR/.venv/bin/python3"
LOG_DIR="$STORE/baseline_eval_logs"
SCRATCH_TMP="/iopsstor/scratch/cscs/sadamov/tmp"
mkdir -p "$LOG_DIR" "$SCRATCH_TMP"

PARTITION="${PARTITION:-normal}"

# Milton bounding box (0-360 lon convention, matching the zarr).
MILTON_LON0=240
MILTON_LON1=300
MILTON_LAT0=5
MILTON_LAT1=40

# The 8 baselines used by tools/plot_bivariate.py PANELS.
MODELS="aurora_encoder graphcast_all sfno_modes10 aifs_perturbed aifsens fcn3 atlas ifs_ens"
declare -A MODEL_KIND=(
    [aurora_encoder]=ai [graphcast_all]=ai [sfno_modes10]=ai [aifs_perturbed]=ai
    [aifsens]=ai [fcn3]=ai [atlas]=ai [ifs_ens]=ref
)

IFS_ENS_ZARR="/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr"
IFS_ENS_MEMBERS="[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]"

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

# Build paths.ml list (per-model forecast.zarr, or the ifs_ens source zarr).
_ml_yaml() {
    local model=$1
    local kind="${MODEL_KIND[$model]}"
    if [[ "$kind" == "ref" ]]; then
        echo "[\"${IFS_ENS_ZARR}\"]"
        return
    fi
    local out="[" first=1
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

# Build selection.datetimes list (init:end, 360h windows).
_dt_yaml() {
    local out="[" first=1
    for it in "${INIT_TIMES[@]}"; do
        local init_date="${it%%T*}" hh="${it##*T}" end_date
        end_date=$(python3 -c "from datetime import datetime,timedelta; print((datetime.fromisoformat('${init_date}T${hh}') + timedelta(hours=360)).strftime('%Y-%m-%dT%H'))")
        [[ $first -eq 0 ]] && out+=", "
        out+="\"${it}:${end_date}\""
        first=0
    done
    out+="]"
    echo "$out"
}

# Milton-box multivariate-only config, geostrophic pair only.
generate_milton_config() {
    local model=$1 output_root=$2
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
  levels: [500]
  temporal_resolution_hours: null
  datetimes: ${dt_yaml}
  latitudes: [${MILTON_LAT0}, ${MILTON_LAT1}]
  longitudes: [${MILTON_LON0}, ${MILTON_LON1}]

  variables_2d: []

  variables_3d:
    - "geopotential"
    - "u_component_of_wind"
    - "v_component_of_wind"

  ensemble_members: ${members_yaml}

  ensemble:
    multivariate: pooled

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

modules:
  maps: false
  histograms: false
  wd_kde: false
  energy_spectra: false
  vertical_profiles: false
  deterministic: false
  ets: false
  fss: false
  probabilistic: false
  ssim: false
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
  multivariate:
    bivariate_pairs:
      - ["geopotential_height_gradient", "wind_speed"]
    coriolis_parameter: 5.58e-5
    bivariate_max_samples: auto
    bins: 100

performance:
  dask_scheduler: distributed
  dask_profile: balanced
YAML
}

# Write per-model configs.
CONFIGS=()
for model in $MODELS; do
    eval_root="$STORE/baselines/${model}/eval_milton"
    cfg="$STORE/baselines/${model}/eval_milton_geostrophic_config.yaml"
    mkdir -p "$(dirname "$cfg")"
    generate_milton_config "$model" "$eval_root" > "$cfg"
    CONFIGS+=("$cfg")
    echo "wrote $cfg  ->  $eval_root/multivariate/" >&2
done

# Body that runs all 8 evals sequentially inside one job.
RUN_BODY="set -euo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR='${SCRATCH_TMP}'
export DASK_TEMPORARY_DIRECTORY='${SCRATCH_TMP}'
source '${SRC_DIR}/.venv/bin/activate'
export SSL_CERT_FILE=\$(python -c 'import certifi;print(certifi.where())')
"
for cfg in "${CONFIGS[@]}"; do
    RUN_BODY+="echo '=== ${cfg} ==='
python -m swissclim_evaluations.cli --config '${cfg}'
"
done

JOB_TAG="eval_geostrophic_milton"
echo "" >&2
echo "Submitting ${JOB_TAG} (8 baselines, Milton box, geostrophic-only) ..." >&2
sbatch --parsable \
    --job-name="$JOB_TAG" \
    --partition="$PARTITION" \
    --account=a122 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=128 \
    --mem=320G \
    --time="06:00:00" \
    --output="$LOG_DIR/${JOB_TAG}_%j.out" \
    --error="$LOG_DIR/${JOB_TAG}_%j.err" \
    --wrap="$RUN_BODY"
