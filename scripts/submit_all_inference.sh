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
# Submission modes:
#   * WEEK_HELPER (default for all models): one sbatch per (model, week),
#     each runs 14 inits sequentially in a shell helper. /dev/shm semaphores
#     leaked by Python multiprocessing are cleaned between inits to avoid IPC
#     exhaustion -> SIGSEGV in round 2/3 of a later init.
#   * PER_INIT (opt-in via PER_INIT=1 env var): one sbatch per (model, init_time).
#     Full process isolation, ~14x more jobs but immune to the multi-GPU
#     multiprocessing state-accumulation SIGSEGV that affects SFNO when running
#     many inits in one process tree. Use when a model exhibits SIGSEGV with
#     the week-helper despite the IPC cleanup. Aurora is a likely candidate
#     for needing PER_INIT (also multi-GPU multiprocessing); test default
#     week-helper first, fall back to PER_INIT=1 on failure. GraphCast is
#     sequential JAX -- no SIGSEGV risk, week-helper always fine.
#
# Usage:
#   bash scripts/submit_all_inference.sh              # all probabilistic models + sfno_modes10
#   bash scripts/submit_all_inference.sh fcn3 atlas    # specific models
#   PER_INIT=1 bash scripts/submit_all_inference.sh ...  # force per-init for any model
#
# Options (env vars):
#   CHAIN=1                Chain jobs sequentially per model (--dependency=afterany).
#                          Useful for CDS-based models to avoid API throttling.
#   AFTER_JOB=N            Wait for slurm job N before starting first job per model.
#   PER_INIT=1             Force per-init submission for all models.
#   PER_INIT_TIME_LIMIT=T  Walltime per per-init job (default 01:00:00).
#   WEEKS=YYYY-MM-DD,...   Comma-separated week_starts to limit scope.
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$STORE/baseline_logs}"
WORKDIR=/workspace/ai-models-ensembles

LEAD_HOURS=360
NUM_MEMBERS=10
OUTPUT_LEVELS="500,850"
OUTPUT_VARS="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,geopotential,mean_sea_level_pressure,specific_humidity,temperature,u_component_of_wind,v_component_of_wind"
SEED=42
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="12:00:00"
CHAIN="${CHAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"
AFTER_JOB="${AFTER_JOB:-}"  # wait for this job before starting first job per model

# Exact week starts matching ifs_ens_wb2.zarr init_times
if [[ -n "${WEEKS:-}" ]]; then
    IFS=',' read -ra WEEK_STARTS <<< "$WEEKS"
else
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
fi

# Probabilistic baselines + post-hoc perturbation baselines (one per det model).
# Perturbation recipes are the user's picks after inspecting all phase
# (1, 2, 2c, 3, 3b) ablation intercomps. See [[calibration-winners]] in memory.
#   aurora_encoder: layer-group encoder,    sigma=0.025  (Phase 2b)
#   graphcast_all:  layer all (no targeting), sigma=0.01 (Phase 1)
#   sfno_modes10:   coarse-modes l<10,      sigma=0.25   (Phase 3)
#   aifs_perturbed: layer-group decoder,    sigma=0.0275 (Phase 2)
# Review-response runs (2026-08-25, Fuhrer review):
#   *_ic_only  - IC perturbation ONLY (IFS-ENS EDA analyses), weight noise off.
#                Isolates the IC contribution against the weight-only and the
#                weight+IC (_ic) arms -> review item 1.
#   rival cells - the runner-up ablation configs within the +-0.02 CRPSS band
#                of each production pick, rerun on held-out production inits
#                -> review item 3.
MODELS="fcn3 atlas aifsens sfno_modes10 aurora_encoder graphcast_all aifs_perturbed"
IC_ZARR="/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr"
IC_ONLY_MODELS="aurora_ic_only graphcast_ic_only sfno_ic_only aifs_ic_only"
RIVAL_MODELS="aurora_enc_s044 graphcast_m2g graphcast_g2m sfno_enc_s054 sfno_enc_s035 aifs_all_s010"
declare -A MODEL_IDS=(
    [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens
    [sfno_modes10]=sfno [aurora_encoder]=aurora [graphcast_all]=graphcast_operational
    [aifs_perturbed]=aifs
    [aurora_ic_only]=aurora [graphcast_ic_only]=graphcast_operational
    [sfno_ic_only]=sfno [aifs_ic_only]=aifs
    [aurora_enc_s044]=aurora [graphcast_m2g]=graphcast_operational
    [graphcast_g2m]=graphcast_operational [sfno_enc_s054]=sfno
    [sfno_enc_s035]=sfno [aifs_all_s010]=aifs
)
declare -A DATA_SRC=(
    [fcn3]=arco [atlas]=arco [aifsens]=cds
    [sfno_modes10]=arco [aurora_encoder]=arco [graphcast_all]=arco
    [aifs_perturbed]=cds
    [aurora_ic_only]=arco [graphcast_ic_only]=arco
    [sfno_ic_only]=arco [aifs_ic_only]=cds
    [aurora_enc_s044]=arco [graphcast_m2g]=arco [graphcast_g2m]=arco
    [sfno_enc_s054]=arco [sfno_enc_s035]=arco [aifs_all_s010]=cds
)
declare -A CONTAINER_BASE=(
    [fcn3]=fcn3 [atlas]=atlas [aifsens]=aifsens
    [sfno_modes10]=sfno [aurora_encoder]=aurora [graphcast_all]=graphcast
    [aifs_perturbed]=aifs
    [aurora_ic_only]=aurora [graphcast_ic_only]=graphcast
    [sfno_ic_only]=sfno [aifs_ic_only]=aifs
    [aurora_enc_s044]=aurora [graphcast_m2g]=graphcast
    [graphcast_g2m]=graphcast [sfno_enc_s054]=sfno
    [sfno_enc_s035]=sfno [aifs_all_s010]=aifs
)
# Per-model extra inference flags (post-hoc perturbation recipe per variant).
declare -A EXTRA_FLAGS=(
    [sfno_modes10]="--weight-magnitude 0.25 --coarse-mode-cut 10"
    [aurora_encoder]="--weight-magnitude 0.025 --layer encoder"
    [graphcast_all]="--weight-magnitude 0.01 --layer all"
    [aifs_perturbed]="--weight-magnitude 0.0275 --layer decoder"
    # IC-only: no weight noise at all (--weight-magnitude defaults to 0).
    [aurora_ic_only]="--ic-zarr $IC_ZARR"
    [graphcast_ic_only]="--ic-zarr $IC_ZARR"
    [sfno_ic_only]="--ic-zarr $IC_ZARR"
    [aifs_ic_only]="--ic-zarr $IC_ZARR"
    # Runner-up ablation cells (Tab. calibration, CRPSS@240 within ~0.02 of
    # each winner): aurora Phase 2, graphcast Phase 2 + 2b, sfno Phase 2 + 2b,
    # aifs Phase 1. Magnitudes are the EXACT ablation values (the table rounds
    # them), so these runs reproduce the same configuration out-of-sample.
    [aurora_enc_s044]="--weight-magnitude 0.044176 --layer encoder"
    [graphcast_m2g]="--weight-magnitude 0.029665 --layer m2g"
    [graphcast_g2m]="--weight-magnitude 0.014 --layer g2m"
    [sfno_enc_s054]="--weight-magnitude 0.053852 --layer encoder"
    [sfno_enc_s035]="--weight-magnitude 0.035 --layer encoder"
    [aifs_all_s010]="--weight-magnitude 0.01 --layer all"
)
# PER_INIT mode (env var, default 0 for all models): submit one sbatch per init
# instead of a 14-init week-helper. Use when a model exhibits multiprocessing
# state-accumulation SIGSEGVs across sequential inits (see MEMORY.md, e.g. SFNO).
# Opt in with `PER_INIT=1 bash scripts/submit_all_inference.sh ...`.
REQUESTED="${@:-$MODELS}"
# Single-job per-init walltime (used by PER_INIT mode). Each init is ~12 min
# for SFNO so 1h is comfortable.
PER_INIT_TIME_LIMIT="${PER_INIT_TIME_LIMIT:-01:00:00}"

mkdir -p "$LOG_DIR"

# Per-model last job ID for chaining (seed with AFTER_JOB if set)
declare -A LAST_JOB=()
if [[ -n "$AFTER_JOB" ]]; then
    for m in $REQUESTED; do LAST_JOB[$m]="$AFTER_JOB"; done
fi

count=0
for model in $REQUESTED; do
    model_id="${MODEL_IDS[$model]:-}"
    if [[ -z "$model_id" ]]; then
        echo "SKIP $model: not a probabilistic baseline"
        continue
    fi

    container="$STORE/${CONTAINER_BASE[$model]}.sqsh"
    dsrc="${DATA_SRC[$model]}"
    extra_flags="${EXTRA_FLAGS[$model]:-}"
    # Per-init mode (default 0 / week-helper). Opt in via PER_INIT=1.
    per_init="${PER_INIT:-0}"

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        continue
    fi

    # Container mounts -- overlay the installed package with bind-mounted source
    # Also bind-mount a persistent host dir to /workspace/.cache/earth2studio so
    # the earth2studio CDS GRIB cache survives across jobs/models. Without this,
    # each container instance rebuilds the cache from scratch -- and CDS-bound
    # models (aifsens, aifs_perturbed) re-pull tens of GBs per init from a slow
    # queue. The host dir is created up front in this script.
    # CDS-bound models (aifs*, aifsens) MUST use the persistent cache on
    # capstor: it holds the 5376 pre-downloaded CDS GRIBs (README_DO_NOT_DELETE)
    # that make an AIFS init ~8 min instead of the ~2 h it takes when every
    # (var, time) has to queue against the CDS API. The scratch copy gets
    # purged (down to 898 entries by 2026-08-26), so it is only safe for the
    # ARCO-bound models.
    if [[ "$dsrc" == "cds" ]]; then
        E2S_CACHE_DIR="$STORE/e2s_cache_backup"
    else
        E2S_CACHE_DIR="/iopsstor/scratch/cscs/sadamov/e2s_cache"
    fi
    mkdir -p "$E2S_CACHE_DIR"
    mounts="${SRC_DIR}:${WORKDIR},${SRC_DIR}/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles,${STORE}:${STORE},${E2S_CACHE_DIR}:/workspace/.cache/earth2studio"
    # Perturbed-analysis store for the --ic-zarr runs.
    IC_DIR=$(dirname "$IC_ZARR")
    [[ -d "$IC_DIR" ]] && mounts+=",${IC_DIR}:${IC_DIR}"
    # The week-helper script is read from inside the container, so LOG_DIR has
    # to be visible there. Only $STORE is mounted by default, so an overridden
    # LOG_DIR (e.g. on iopsstor scratch) needs its own bind mount.
    case "$LOG_DIR" in
        "$STORE"/*) ;;
        *) mounts+=",${LOG_DIR}:${LOG_DIR}" ;;
    esac
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    for week_start in "${WEEK_STARTS[@]}"; do
        week_tag="${week_start//-/}"

        if [[ "$per_init" == "1" ]]; then
            # PER_INIT mode: one sbatch per init -- full process isolation.
            # Stagger starts by 2 min to spread NGC checkpoint fetches.
            init_idx=0
            for day_offset in 0 1 2 3 4 5 6; do
                init_date=$(date -d "${week_start} +${day_offset} days" +%Y-%m-%d)
                for hour in "00:00" "12:00"; do
                    init_time="${init_date}T${hour}"
                    init_tag="${init_date//-/}_${hour//:}"
                    out_dir="$STORE/baselines/${model}/${init_tag}"
                    out_zarr="$out_dir/forecast.zarr"

                    if [[ -d "$out_zarr" ]]; then
                        continue
                    fi
                    if [[ -d "$out_dir/_e2s_work" ]]; then
                        rm -rf "$out_dir/_e2s_work"
                    fi

                    delay=$((init_idx * 2))
                    init_idx=$((init_idx + 1))
                    job_tag="bl_${model}_${init_tag}"

                    dep_flag=()
                    if [[ -n "${LAST_JOB[$model]:-}" ]]; then
                        dep_flag=(--dependency="afterany:${LAST_JOB[$model]}")
                    fi

                    if [[ "$DRY_RUN" == "1" ]]; then
                        echo "  DRY $job_tag: $model_id ${extra_flags} -> $out_zarr"
                        count=$((count + 1)); continue
                    fi
                    jobid=$(sbatch --parsable \
                        "${dep_flag[@]}" \
                        --begin="now+${delay}minutes" \
                        --job-name="$job_tag" \
                        --partition="$PARTITION" \
                        --account=ab016 \
                        --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=800G --gres=gpu:4 \
                        --time="$PER_INIT_TIME_LIMIT" \
                        --output="$LOG_DIR/${job_tag}_%j.out" \
                        --error="$LOG_DIR/${job_tag}_%j.err" \
                        --container-image="$container" \
                        --container-mounts="$mounts" \
                        --container-workdir="$WORKDIR" \
                        --wrap="python -m ai_models_ensembles.cli infer --model $model_id --init '${init_time}' --lead-hours $LEAD_HOURS --members $NUM_MEMBERS --data-source $dsrc --output-levels '$OUTPUT_LEVELS' --output-vars '$OUTPUT_VARS' --seed $SEED ${extra_flags} --output '${out_zarr}'")
                    echo "  $job_tag -> $jobid (+${delay}min)"
                    [[ "$CHAIN" == "1" ]] && LAST_JOB[$model]="$jobid"
                    count=$((count + 1))
                done
            done
            continue
        fi

        # WEEK_HELPER mode (default for non-SFNO models): one job per (model, week),
        # 14 inits run sequentially. Between inits, clean Python multiprocessing
        # semaphores leaked into /dev/shm by the multi-GPU pool. Accumulated leaks
        # can exhaust IPC and trigger SIGSEGV in round 2/3 of a later init.
        helper="$LOG_DIR/bl_${model}_${week_tag}.sh"
        cat > "$helper" <<SCRIPT
#!/bin/sh
set -e
SCRIPT

        any_missing=false
        for day_offset in 0 1 2 3 4 5 6; do
            init_date=$(date -d "${week_start} +${day_offset} days" +%Y-%m-%d)
            for hour in "00:00" "12:00"; do
                init_time="${init_date}T${hour}"
                init_tag="${init_date//-/}_${hour//:}"

                out_dir="$STORE/baselines/${model}/${init_tag}"
                out_zarr="$out_dir/forecast.zarr"

                if [[ -d "$out_zarr" ]]; then
                    continue
                fi
                if [[ -d "$out_dir/_e2s_work" ]]; then
                    rm -rf "$out_dir/_e2s_work"
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
    ${extra_flags} \\
    --output '${out_zarr}'
find /dev/shm -maxdepth 1 \( -name 'sem.mp-*' -o -name 'sem.pym-*' -o -name 'sem.tmp.*' \) -delete 2>/dev/null || true
sleep 15
SCRIPT
            done
        done

        if ! $any_missing; then
            echo "  SKIP $model week $week_start: all outputs exist"
            rm -f "$helper"
            continue
        fi

        chmod +x "$helper"

        job_tag="bl_${model}_${week_tag}"
        echo "  $job_tag (week helper)"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  DRY $job_tag ($(grep -c '^python -m' "$helper") inits)"
            sed -n '1,14p' "$helper"
            count=$((count + 1)); continue
        fi

        dep_flag=()
        if [[ -n "${LAST_JOB[$model]:-}" ]]; then
            dep_flag=(--dependency="afterany:${LAST_JOB[$model]}")
        fi

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  DRY $job_tag ($(grep -c '^python' "$helper") inits)"
            sed -n '1,14p' "$helper"
            count=$((count + 1)); continue
        fi
        jobid=$(sbatch --parsable \
            "${dep_flag[@]}" \
            --job-name="$job_tag" \
            --partition="$PARTITION" \
            --account=ab016 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=32 \
            --mem=800G \
            --gres=gpu:4 \
            --time="$TIME_LIMIT" \
            --output="$LOG_DIR/${job_tag}_%j.out" \
            --error="$LOG_DIR/${job_tag}_%j.err" \
            --container-image="$container" \
            --container-mounts="$mounts" \
            --container-workdir="$WORKDIR" \
            --wrap="sh ${helper}")

        [[ "$CHAIN" == "1" ]] && LAST_JOB[$model]="$jobid"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted $count baseline jobs."
echo "Monitor with: squeue -u \$USER | grep bl_"
echo "Logs: $LOG_DIR/"
echo ""
echo "Output layout:"
echo "  $STORE/baselines/<model>/<YYYYMMDD_HHMM>/forecast.zarr"
