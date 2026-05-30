#!/usr/bin/env bash
# Re-launch Phase 1 SFNO + GraphCast after fixing complex-skip + hyperparam-perturb
# bugs in e2s_perturbation.py. Old buggy runs were archived to
# $STORE/ablation/phase1_buggy_pre_audit_2026-05-18/. mag_0 is preserved
# (sigma=0 is bit-identical regardless of the fixes).
#
# Flow:
#   1) Submit fresh inference for SFNO + GraphCast (5 mags x 4 inits each = 40 jobs).
#   2) Chain per-model eval with afterok dependency on inference.
#   3) Chain per-model intercomparison after eval.
#
# Usage: bash scripts/relaunch_phase1_sfno_graphcast.sh
set -euo pipefail

STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/ablation_logs"
PARTITION="${PARTITION:-normal}"
WORKDIR=/workspace/ai-models-ensembles
TIME_LIMIT="12:00:00"
NUM_MEMBERS=10
LEAD_HOURS=240
OUTPUT_LEVELS="500,850"
SEED=42

INIT_TIMES=(
    "2024-02-15T00:00"
    "2023-05-15T00:00"
    "2023-08-15T00:00"
    "2024-11-15T00:00"
)
MAGS="0.001 0.003 0.01 0.03 0.1"

declare -A MODEL_IDS=( [sfno]=sfno [graphcast]=graphcast_operational )
declare -A DATA_SRC=( [sfno]=arco [graphcast]=arco )

mkdir -p "$LOG_DIR"

# Submit one inference job for a (model, init, mag) triple.
# Echoes the job ID on stdout.
submit_one() {
    local model=$1 init_time=$2 wmag=$3
    local model_id="${MODEL_IDS[$model]}"
    local container="$STORE/${model}.sqsh"
    local dsrc="${DATA_SRC[$model]}"
    local init_tag="${init_time%%T*}"
    init_tag="${init_tag//-/}"
    local out_dir="$STORE/ablation/phase1/${model_id}/${init_tag}/mag_${wmag}_layer_all"
    local out_zarr="$out_dir/forecast.zarr"
    if [[ -d "$out_zarr" ]]; then
        echo "SKIP_EXISTS:$model:$init_tag:$wmag" >&2
        echo ""
        return
    fi
    mkdir -p "$out_dir"

    local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    local job_tag="phase1_${model}_${init_tag}_m${wmag}_lall"
    sbatch --parsable \
        --job-name="abl_${job_tag}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=444G --gres=gpu:4 \
        --time="$TIME_LIMIT" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --container-image="$container" \
        --container-mounts="$mounts" \
        --container-workdir="$WORKDIR" \
        --wrap="python -m ai_models_ensembles.cli infer \
            --model $model_id \
            --init '$init_time' \
            --lead-hours $LEAD_HOURS \
            --members $NUM_MEMBERS \
            --weight-magnitude $wmag \
            --data-source $dsrc \
            --output-levels '$OUTPUT_LEVELS' \
            --seed $SEED \
            --output '$out_zarr'"
}

# ---------------------------------------------------------------------------
# Phase 1 inference
# ---------------------------------------------------------------------------
declare -A INF_IDS_BY_MODEL
for model in sfno graphcast; do
    ids=""
    echo "=== Submitting Phase 1 inference for $model ==="
    for init in "${INIT_TIMES[@]}"; do
        for mag in $MAGS; do
            jid=$(submit_one "$model" "$init" "$mag")
            if [[ -n "$jid" ]]; then
                ids+="${jid}:"
                echo "  submitted $model init=$init mag=$mag -> $jid"
            fi
        done
    done
    INF_IDS_BY_MODEL[$model]="${ids%:}"
done

echo
echo "=== Inference submitted. Job ID summaries: ==="
for model in sfno graphcast; do
    echo "  $model: ${INF_IDS_BY_MODEL[$model]}"
done

# ---------------------------------------------------------------------------
# Per-model eval with afterok dependency on that model's inference
# ---------------------------------------------------------------------------
echo
echo "=== Chaining eval jobs ==="
declare -A EVAL_IDS_BY_MODEL
for model in sfno graphcast; do
    model_id="${MODEL_IDS[$model]}"
    inf_ids="${INF_IDS_BY_MODEL[$model]}"
    if [[ -z "$inf_ids" ]]; then
        echo "  $model: no inference jobs; skip eval chaining"
        continue
    fi
    # Use the evaluate_ablation script with AFTER_JOB chaining.
    # AFTER_JOB takes a colon-separated list of job IDs (one or more).
    eval_id=$(AFTER_JOB="$inf_ids" bash "$SRC_DIR/scripts/evaluate_ablation.sh" phase1 "$model" 2>&1 \
              | tee /dev/stderr | grep -oE "Submitted batch job [0-9]+" | tail -1 | awk '{print $NF}')
    EVAL_IDS_BY_MODEL[$model]="$eval_id"
    echo "  $model eval -> $eval_id (afterok:${inf_ids})"
done

# ---------------------------------------------------------------------------
# Per-model intercomparison after eval
# ---------------------------------------------------------------------------
echo
echo "=== Chaining intercomparison jobs ==="
for model in sfno graphcast; do
    eval_id="${EVAL_IDS_BY_MODEL[$model]:-}"
    if [[ -z "$eval_id" ]]; then
        echo "  $model: no eval job; skip intercompare"
        continue
    fi
    AFTER_JOB="$eval_id" bash "$SRC_DIR/scripts/evaluate_ablation.sh" intercompare phase1 "$model" \
        2>&1 | tee /dev/stderr | grep -oE "Submitted batch job [0-9]+" | tail -1
done

echo
echo "Done. Monitor with: squeue -u sadamov"
echo "Logs: $LOG_DIR/"
