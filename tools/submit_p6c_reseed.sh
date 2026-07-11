#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase-6 refresh RESEED (corrected fresh-hook seeding, 2026-07-10) for the
# weight-space backbones. Regenerates the fig:phase6 refresh (p6c) grids with
# the fixed per-(member, tensor, epoch) seeding so the reported spatial-mean
# SSR + sec 4.3 CRPS costs can be updated.
#
# Mirrors the ORIGINAL per-init production launch (one sbatch per init = process
# isolation; this is REQUIRED for SFNO to avoid the multi-GPU multiprocessing
# SIGSEGV, and harmless for the others). Per-model settings are taken verbatim
# from the original bl_<model>_p6c logs:
#   sfno   : SFNO_FRESH  sigma=0.35   modes10 refresh-20  mem=444G arco ~8 min/init
#   aurora : AURORA_FRESH sigma=0.025 encoder refresh-20  mem=800G arco ~2h10/init
#   aifs   : AIFS_FRESH  sigma=0.0275 decoder refresh-20  mem=800G CDS  ~2h/init
#
# Writes to $STORE/baselines/<model>_p6c_reseed/ (originals untouched). Global
# throttle keeps ALL p6reseed jobs (any model) at <= MAX_CONCURRENT nodes.
#
# Usage: bash tools/submit_p6c_reseed.sh <sfno|aurora|aifs> [spot|full]
#   spot (default) = 8 season-spanning anchor inits; full = all 112.
#   MAX_CONCURRENT env (default 12) caps concurrent nodes across all models.
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL="${1:?usage: submit_p6c_reseed.sh <sfno|aurora|aifs> [spot|full]}"
MODE="${2:-spot}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"

STORE=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles
SRC_DIR=/users/sadamov/pyprojects/ai-models-ensembles
LOG_DIR=$STORE/baseline_logs
WORKDIR=/workspace/ai-models-ensembles
PARTITION="${PARTITION:-normal}"

# Per-model config (from the original bl_<model>_p6c logs + scripts/submit_ablation.sh).
# mem=800G for all (full node). Cache: AIFS uses the PERSISTENT $STORE CDS cache
# backup (the scratch e2s_cache purges and does not hold the CDS inits, so a live
# CDS fetch is needed otherwise); sfno/aurora use the scratch arco cache.
case "$MODEL" in
  sfno)   MEM=800G; TIME=01:00:00; DSRC=arco; E2S_CACHE=/iopsstor/scratch/cscs/sadamov/e2s_cache; FRESH_ENV="SFNO_FRESH=1 SFNO_FRESH_SIGMA=0.35 SFNO_FRESH_MODE_CUT=10 SFNO_FRESH_REFRESH_EVERY=20"; EXTRA_CLI="--coarse-mode-cut 10"; NEED_SSL=0 ;;
  aurora) MEM=800G; TIME=03:00:00; DSRC=arco; E2S_CACHE=/iopsstor/scratch/cscs/sadamov/e2s_cache; FRESH_ENV="AURORA_FRESH=1 AURORA_FRESH_SIGMA=0.025 AURORA_FRESH_REFRESH_EVERY=20"; EXTRA_CLI=""; NEED_SSL=0 ;;
  aifs)   MEM=800G; TIME=03:00:00; DSRC=cds;  E2S_CACHE=$STORE/e2s_cache_backup;                  FRESH_ENV="AIFS_FRESH=1 AIFS_FRESH_SIGMA=0.0275 AIFS_FRESH_REFRESH_EVERY=20"; EXTRA_CLI=""; NEED_SSL=1 ;;
  *) echo "unknown model '$MODEL' (want sfno|aurora|aifs)"; exit 1 ;;
esac
CONTAINER="$STORE/${MODEL}.sqsh"
[[ -f "$CONTAINER" ]] || { echo "container $CONTAINER not found"; exit 1; }
mkdir -p "$LOG_DIR" "$E2S_CACHE"

MOUNTS="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE}:/workspace/.cache/earth2studio"
# CDS/ECMWF creds (matches scripts/submit_ablation.sh): mount to both ~ and /root.
for rc in ~/.cdsapirc ~/.ecmwfapirc; do
    [[ -f "$rc" ]] && MOUNTS+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
done

LEAD_HOURS=240; NUM_MEMBERS=10; SEED=42; OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"

ANCHORS=("2023-01-02T00:00" "2023-04-02T00:00" "2023-07-02T00:00" "2023-10-02T00:00" "2024-01-02T00:00" "2024-04-02T00:00" "2024-07-02T00:00" "2024-10-02T00:00")
if [[ "$MODE" == "full" ]]; then
    INITS=()
    for ws in 2023-01-02 2023-04-02 2023-07-02 2023-10-02 2024-01-02 2024-04-02 2024-07-02 2024-10-02; do
        for off in 0 1 2 3 4 5 6; do
            # GNU date (coreutil, always on PATH); python3 is not resolvable in a
            # detached background shell, which produced empty dates + malformed inits.
            d=$(date -d "${ws} + ${off} days" +%Y-%m-%d)
            [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "FATAL: date arithmetic failed for ${ws}+${off} days (got '${d}')"; exit 1; }
            for hh in "00:00" "12:00"; do INITS+=("${d}T${hh}"); done
        done
    done
else
    INITS=("${ANCHORS[@]}")
fi

SSL_EXPORT=""
[[ "$NEED_SSL" == "1" ]] && SSL_EXPORT='export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())" 2>/dev/null || true); '
SWEEP="find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true"

count=0
for init_time in "${INITS[@]}"; do
    [[ "$init_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}$ ]] || { echo "SKIP malformed init '$init_time'"; continue; }
    init_tag="${init_time%%T*}"; init_tag="${init_tag//-/}_${init_time##*T}"; init_tag="${init_tag//:/}"
    out_dir="$STORE/baselines/${MODEL}_p6c_reseed/${init_tag}"; out_zarr="$out_dir/forecast.zarr"
    if [[ -d "$out_zarr" ]]; then echo "SKIP $MODEL $init_tag (exists)"; continue; fi
    if squeue --me -h -o "%j" 2>/dev/null | grep -qx "p6reseed_${MODEL}_${init_tag}"; then echo "SKIP $MODEL $init_tag (already queued)"; continue; fi
    [[ -d "$out_dir/_e2s_work" ]] && rm -rf "$out_dir/_e2s_work"
    mkdir -p "$out_dir"
    # Global throttle: cap concurrent+pending p6reseed jobs (ANY model) at MAX_CONCURRENT nodes.
    while [[ $(squeue --me -h -o "%j" 2>/dev/null | grep -c '^p6reseed_') -ge $MAX_CONCURRENT ]]; do sleep 60; done
    job_tag="p6reseed_${MODEL}_${init_tag}"
    jobid=$(sbatch --parsable \
        --job-name="$job_tag" --partition="$PARTITION" --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=$MEM --gres=gpu:4 --time=$TIME \
        --output="$LOG_DIR/${job_tag}_%j.out" --error="$LOG_DIR/${job_tag}_%j.err" \
        --container-image="$CONTAINER" --container-mounts="$MOUNTS" --container-workdir="$WORKDIR" \
        --wrap="${SWEEP}; ${SSL_EXPORT}${FRESH_ENV} \
            python -m ai_models_ensembles.cli infer --model $MODEL --init '${init_time}' \
            --lead-hours $LEAD_HOURS --members $NUM_MEMBERS --weight-magnitude 0 ${EXTRA_CLI} \
            --data-source $DSRC --output-levels '$OUTPUT_LEVELS' --output-vars '$OUTPUT_VARS' \
            --seed $SEED --output '${out_zarr}'; STATUS=\$?; ${SWEEP}; exit \$STATUS")
    echo "  $job_tag -> $jobid"
    count=$((count + 1))
done
echo ""
echo "Submitted $count ${MODEL}_p6c_reseed jobs (MODE=$MODE, one-per-init, <=${MAX_CONCURRENT} nodes, FIXED seeding)."
echo "Output: $STORE/baselines/${MODEL}_p6c_reseed/<init_tag>/forecast.zarr"
