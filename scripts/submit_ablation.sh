#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ablation study for weight perturbation across 3 deterministic models.
#
# Fully reproducible: every parameter combination is enumerated here.
# Run phases sequentially -- Phase 2 magnitudes are set based on Phase 1.
#
# Usage:
#   bash scripts/submit_ablation.sh phase1              # magnitude sweep
#   bash scripts/submit_ablation.sh phase2              # layer-group sweep
#   bash scripts/submit_ablation.sh phase2b             # refinement
#   bash scripts/submit_ablation.sh all                 # all phases
#   bash scripts/submit_ablation.sh phase1 aurora       # single model
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/ablation_logs"
WORKDIR=/workspace/ai-models-ensembles

PARTITION="${PARTITION:-normal}"
TIME_LIMIT="12:00:00"
NUM_MEMBERS=10
LEAD_HOURS=240
OUTPUT_LEVELS="500,850"
SEED=42

# 4 init times: one per season, mid-month, avoiding standard verification weeks
# (standard verification: first week of each season 2023-2024)
# Training cutoffs: GraphCast 2021, SFNO 2015, Aurora 2022
# All 4 dates are outside all training ranges.
INIT_TIMES=(
    "2024-02-15T00:00"   # DJF (winter)
    "2023-05-15T00:00"   # MAM (spring)
    "2023-08-15T00:00"   # JJA (summer)
    "2024-11-15T00:00"   # SON (autumn)
)

MODELS="aurora graphcast sfno"
declare -A MODEL_IDS=( [aurora]=aurora [graphcast]=graphcast_operational [sfno]=sfno )
declare -A DATA_SRC=( [aurora]=arco [graphcast]=arco [sfno]=arco )

# ---------------------------------------------------------------------------
# Phase 1: Magnitude sweep (5 magnitudes x 3 models x 4 inits = 60 runs)
# All layers perturbed. Goal: find optimal magnitude per model.
# ---------------------------------------------------------------------------
PHASE1_MAGNITUDES="0.001 0.003 0.01 0.03 0.1"

# ---------------------------------------------------------------------------
# Phase 2: Layer-group sweep (3 groups x 3 models x 4 inits = 36 runs)
# Uses best magnitude from Phase 1 (set here after Phase 1 analysis).
# Layer groups: model-specific architectural components.
# Named groups are resolved by _MODEL_LAYER_GROUPS in e2s_perturbation.py.
#
# Aurora (644 tensors):  backbone (594) | decoder (17) | encoder (33)
# GraphCast (267 tensors): g2m (36) | m2g (30) | m2m (198)
# SFNO (79 tensors):    early (26) | middle (27) | late (26)
# ---------------------------------------------------------------------------
declare -A PHASE2_GROUPS_aurora=( [0]=encoder [1]=backbone [2]=decoder )
declare -A PHASE2_GROUPS_graphcast=( [0]=g2m [1]=m2m [2]=m2g )
declare -A PHASE2_GROUPS_sfno=( [0]=early [1]=middle [2]=late )

# Best magnitudes from Phase 1 -- UPDATE AFTER PHASE 1 ANALYSIS
declare -A PHASE2_BEST_MAG=( [aurora]=0.01 [graphcast]=0.01 [sfno]=0.01 )

# ---------------------------------------------------------------------------
# Phase 2b: Refinement (3 fine magnitudes x 3 models x 4 inits = 36 runs)
# Narrow magnitude range around Phase 1 winner, best layer group from Phase 2.
# ---------------------------------------------------------------------------
PHASE2B_MAGNITUDES="0.005 0.01 0.02"

# Best layer group from Phase 2 -- UPDATE AFTER PHASE 2 ANALYSIS
declare -A PHASE2B_BEST_LAYER=( [aurora]=all [graphcast]=all [sfno]=all )

# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

submit_job() {
    local model=$1
    local model_id="${MODEL_IDS[$model]}"
    local container="$STORE/${model}.sqsh"
    local dsrc="${DATA_SRC[$model]}"
    local init_time=$2
    local wmag=$3
    local layer_spec=$4
    local phase=$5

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        return 1
    fi

    # Build output path
    local init_tag="${init_time%%T*}"
    init_tag="${init_tag//-/}"
    local out_dir="$STORE/ablation/${phase}/${model_id}/${init_tag}/mag_${wmag}_layer_${layer_spec//[:.]/_}"
    local out_zarr="$out_dir/forecast.zarr"

    # Skip if output already exists (allows resuming)
    if [[ -d "$out_zarr" ]]; then
        echo "SKIP $model $init_time mag=$wmag layer=$layer_spec: output exists"
        return 0
    fi
    mkdir -p "$out_dir"

    # Container mounts
    local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    local job_tag="${phase}_${model}_${init_tag}_m${wmag}_l${layer_spec//[:.]/_}"

    # Build layer CLI arg
    local layer_arg=""
    if [[ "$layer_spec" != "all" ]]; then
        layer_arg="--layer '$layer_spec'"
    fi

    echo "  $job_tag"

    sbatch --parsable \
        --job-name="abl_${job_tag}" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --mem=444G \
        --gres=gpu:4 \
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
            $layer_arg \
            --data-source $dsrc \
            --output-levels '$OUTPUT_LEVELS' \
            --seed $SEED \
            --output '$out_zarr'"
}

# ---------------------------------------------------------------------------

run_phase1() {
    local filter_model="${1:-}"
    echo "=== Phase 1: Magnitude sweep ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        for init_time in "${INIT_TIMES[@]}"; do
            for wmag in $PHASE1_MAGNITUDES; do
                submit_job "$model" "$init_time" "$wmag" "all" "phase1" && ((count++)) || true
            done
        done
    done
    echo "Phase 1: submitted $count jobs"
}

run_phase2() {
    local filter_model="${1:-}"
    echo "=== Phase 2: Layer-group sweep ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local wmag="${PHASE2_BEST_MAG[$model]}"

        # Get model-specific layer groups via nameref
        local -n groups="PHASE2_GROUPS_${model}"

        for init_time in "${INIT_TIMES[@]}"; do
            for idx in "${!groups[@]}"; do
                local layer_spec="${groups[$idx]}"
                submit_job "$model" "$init_time" "$wmag" "$layer_spec" "phase2" && ((count++)) || true
            done
        done
    done
    echo "Phase 2: submitted $count jobs"
}

run_phase2b() {
    local filter_model="${1:-}"
    echo "=== Phase 2b: Refinement ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local layer_spec="${PHASE2B_BEST_LAYER[$model]}"
        for init_time in "${INIT_TIMES[@]}"; do
            for wmag in $PHASE2B_MAGNITUDES; do
                submit_job "$model" "$init_time" "$wmag" "$layer_spec" "phase2b" && ((count++)) || true
            done
        done
    done
    echo "Phase 2b: submitted $count jobs"
}

# ---------------------------------------------------------------------------

PHASE="${1:-all}"
MODEL_FILTER="${2:-}"

case "$PHASE" in
    phase1)  run_phase1 "$MODEL_FILTER" ;;
    phase2)  run_phase2 "$MODEL_FILTER" ;;
    phase2b) run_phase2b "$MODEL_FILTER" ;;
    all)
        run_phase1 "$MODEL_FILTER"
        run_phase2 "$MODEL_FILTER"
        run_phase2b "$MODEL_FILTER"
        ;;
    *)
        echo "Usage: $0 {phase1|phase2|phase2b|all} [model]"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER | grep abl_"
echo "Logs: $LOG_DIR/"
echo ""
echo "Output layout:"
echo "  $STORE/ablation/{phase}/{model}/{init_date}/mag_{m}_layer_{l}/forecast.zarr"
