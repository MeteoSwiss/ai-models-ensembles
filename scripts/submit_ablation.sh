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
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
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
# Tensor counts verified from checkpoint dumps 2026-05-18; see
# memory/checkpoint_perturbation_audit.md.
# Aurora (644 tensors):    backbone (594) | decoder (17) | encoder (33)
# GraphCast (264 tensors): g2m (36) | m2g (30) | m2m (198)
# SFNO (87 tensors):       processor (80) | decoder (3) | encoder (3) | residual (1)
# ---------------------------------------------------------------------------
declare -A PHASE2_GROUPS_aurora=( [0]=encoder [1]=backbone [2]=decoder )
declare -A PHASE2_GROUPS_graphcast=( [0]=g2m [1]=m2m [2]=m2g )
declare -A PHASE2_GROUPS_sfno=( [0]=encoder [1]=processor [2]=decoder )

# N_TOTAL = number of perturbable tensors when --layer all is used.
# N_GROUP_<model>[<group>] = number of perturbable tensors in that group.
# Used to apply sqrt(N_TOTAL / N_GROUP) variance scaling so partial-weight
# perturbations match the full-weight output-variance budget at sigma_full.
declare -A N_TOTAL=( [aurora]=644 [graphcast]=264 [sfno]=87 )
declare -A N_GROUP_aurora=( [encoder]=33 [backbone]=594 [decoder]=17 )
declare -A N_GROUP_graphcast=( [g2m]=36 [m2g]=30 [m2m]=198 )
declare -A N_GROUP_sfno=( [encoder]=3 [processor]=80 [decoder]=3 )

# Consolidated sigma_full from Phase 1 analysis. Phase 2 partial-group runs
# scale this by sqrt(N_TOTAL / N_GROUP).
declare -A PHASE2_SIGMA_FULL=( [aurora]=0.01 [graphcast]=0.01 [sfno]=0.01 )

# ---------------------------------------------------------------------------
# Phase 2b: Refinement (3 fine magnitudes x 3 models x 4 inits = 36 runs)
# Narrow magnitude range around Phase 1 winner, best layer group from Phase 2.
# ---------------------------------------------------------------------------
PHASE2B_MAGNITUDES="0.005 0.01 0.02"

# Best layer group from Phase 2 -- UPDATE AFTER PHASE 2 ANALYSIS
declare -A PHASE2B_BEST_LAYER=( [aurora]=all [graphcast]=all [sfno]=all )

# ---------------------------------------------------------------------------
# Phase 3: Physically-motivated coarse-scale-only perturbation
# (lambda >= ~3000 km, upper synoptic and above per SwissClim bands).
# One coarse-scale layer group per model, sqrt(N) variance scaling vs full.
# Aurora: encoder_layers.2 only (96 tensors, 2048 ch, 4 deg / ~5000 km
#         attention receptive field). Decoder_layers.0 dropped post-Phase 2
#         (encoder >> decoder for spread; see memory/phase2_aurora_findings).
# SFNO:   BLOCKED -- needs 240-dim spectral mode layout from makani source.
# GraphCast: BLOCKED -- needs runtime activation hook (edge weights shared).
# ---------------------------------------------------------------------------
declare -A PHASE3_GROUP=( [aurora]=coarse_encoder )
declare -A PHASE3_N_GROUP=( [aurora]=96 )
declare -A PHASE3_SIGMA_FULL=( [aurora]=0.01 )

# SFNO Phase 3 uses a different mechanism (sub-axis slice perturbation, not
# named layer group). Driven by --coarse-mode-cut N applied to the 8 SFNO
# spectral conv weights (*.filter.filter.weight, last axis = 240 lat modes).
# N=10 targets wavelengths >= 4000 km (lmax=10 -> 2*pi*R/10 = 4000 km).
# Sigma is scaled from sigma_full=0.01 by sqrt(N_full_DOF / N_partial_DOF):
#   N_full = 572.5M DOF (87 tensors, complex counts 2x), N_partial = 23.6M
#   DOF (8 * 384*384*10 complex elements * 2 DOF each) -> sigma = 0.04926.
declare -A PHASE3_SFNO_MODE_CUT=( [sfno]=10 )
declare -A PHASE3_SFNO_SIGMA=( [sfno]=0.049261 )

# GraphCast Phase 3 uses runtime activation perturbation on the first 42 mesh
# nodes (levels 0+1 of the multi-mesh, ~3300 km separation). Cannot use
# materialise_perturbed_package (weights are shared across all edges/nodes).
# Phase 3 sweeps four sigma values because the scaling formula breaks down
# for activation perturbation on a tiny spatial subset:
#   0.01  = same fractional magnitude as Phase 1/2 weight perturbation
#   0.03  = ~3x baseline
#   0.10  = ~10x baseline
#   0.312 = strict sqrt(40962/42) sqrt-N scaling -- likely upper bound
PHASE3_GC_SIGMAS="0.01 0.03 0.10 0.312"
PHASE3_GC_NODES=42  # level-0 + level-1 mesh vertices

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
    local members="${6:-$NUM_MEMBERS}"
    local mode_cut="${7:-}"  # Phase 3 SFNO: pass --coarse-mode-cut N
    local gc_sigma="${8:-}"  # Phase 3 GraphCast: activation perturb sigma
    local gc_nodes="${9:-}"  # Phase 3 GraphCast: number of leading mesh nodes

    if [[ ! -f "$container" ]]; then
        echo "SKIP $model: container $container not found"
        return 1
    fi

    # Build output path
    local init_tag="${init_time%%T*}"
    init_tag="${init_tag//-/}"
    local run_tag="mag_${wmag}_layer_${layer_spec//[:.]/_}"
    if [[ -n "$mode_cut" ]]; then
        run_tag="mag_${wmag}_modes${mode_cut}"
    fi
    if [[ -n "$gc_sigma" ]]; then
        # GraphCast Phase 3: wmag is unused (weight perturbation disabled),
        # gc_sigma is the activation perturbation magnitude on first gc_nodes.
        run_tag="gcsigma_${gc_sigma}_gcnodes${gc_nodes}"
    fi
    local out_dir="$STORE/ablation/${phase}/${model_id}/${init_tag}/${run_tag}"
    local out_zarr="$out_dir/forecast.zarr"

    # Skip if output already exists (allows resuming)
    if [[ -d "$out_zarr" ]]; then
        echo "SKIP $model $init_time mag=$wmag layer=$layer_spec mode_cut=${mode_cut:-}: output exists"
        return 0
    fi
    mkdir -p "$out_dir"

    # Container mounts
    local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
    for rc in ~/.cdsapirc ~/.ecmwfapirc; do
        [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
    done

    local job_tag="${phase}_${model}_${init_tag}_${run_tag}"

    # Build layer + mode-cut CLI args
    local layer_arg=""
    if [[ "$layer_spec" != "all" ]]; then
        layer_arg="--layer '$layer_spec'"
    fi
    local mode_cut_arg=""
    if [[ -n "$mode_cut" ]]; then
        mode_cut_arg="--coarse-mode-cut $mode_cut"
    fi
    local gc_args=""
    if [[ -n "$gc_sigma" ]]; then
        gc_args="--graph-coarse-sigma $gc_sigma --graph-coarse-nodes $gc_nodes"
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
            --members $members \
            --weight-magnitude $wmag \
            $layer_arg \
            $mode_cut_arg \
            $gc_args \
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

# Unperturbed baseline: 1 member, mag=0. Provides the deterministic reference
# against which Phase 1 magnitude sweep scores are compared.
# One slurm job per model -- runs all 4 init_times in parallel, one per GPU.
run_phase1_unperturbed() {
    local filter_model="${1:-}"
    echo "=== Phase 1: Unperturbed baseline (mag=0, N=1, 4 inits/node) ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local model_id="${MODEL_IDS[$model]}"
        local container="$STORE/${model}.sqsh"
        local dsrc="${DATA_SRC[$model]}"

        if [[ ! -f "$container" ]]; then
            echo "SKIP $model: container $container not found"
            continue
        fi

        # Helper script: parallel inference of 4 init_times, one per GPU.
        # Includes a sequential checkpoint warmup so HF Hub-cached models
        # (e.g. Aurora) don't hit a shared-cache race when 4 parallel processes
        # try to download the same checkpoint file simultaneously.
        local helper="$LOG_DIR/unperturbed_${model}.sh"
        cat > "$helper" <<'WARMUP_HEADER'
#!/bin/bash
set -e
WARMUP_HEADER
        cat >> "$helper" <<SCRIPT
echo "[warmup] caching ${model_id} checkpoint..."
CUDA_VISIBLE_DEVICES=0 python -c "from ai_models_ensembles.e2s_models import load_model; load_model('${model_id}')" >/dev/null 2>&1 || true
echo "[warmup] done"

SCRIPT

        local any_missing=false
        local gpu_id=0
        for init_time in "${INIT_TIMES[@]}"; do
            local init_tag="${init_time%%T*}"
            init_tag="${init_tag//-/}"
            local out_dir="$STORE/ablation/phase1/${model_id}/${init_tag}/mag_0_layer_all"
            local out_zarr="$out_dir/forecast.zarr"

            if [[ -d "$out_zarr" ]]; then
                echo "  SKIP $model $init_time: output exists" >&2
                gpu_id=$((gpu_id + 1))
                continue
            fi
            mkdir -p "$out_dir"
            any_missing=true

            cat >> "$helper" <<SCRIPT
(
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    echo "[GPU ${gpu_id}] starting ${init_time}"
    python -m ai_models_ensembles.cli infer \\
        --model $model_id \\
        --init '${init_time}' \\
        --lead-hours $LEAD_HOURS \\
        --members 1 \\
        --weight-magnitude 0 \\
        --data-source $dsrc \\
        --output-levels '$OUTPUT_LEVELS' \\
        --seed $SEED \\
        --output '${out_zarr}'
    echo "[GPU ${gpu_id}] done ${init_time}"
) &
SCRIPT
            gpu_id=$((gpu_id + 1))
        done
        echo "wait" >> "$helper"

        if ! $any_missing; then
            echo "  SKIP $model: all 4 unperturbed outputs exist"
            rm -f "$helper"
            continue
        fi
        chmod +x "$helper"

        # Container mounts
        local mounts="${SRC_DIR}:${WORKDIR},${STORE}:${STORE}"
        for rc in ~/.cdsapirc ~/.ecmwfapirc; do
            [[ -f "$rc" ]] && mounts+=",${rc}:${rc},${rc}:/root/$(basename "$rc")"
        done

        local job_tag="phase1_unperturbed_${model}"
        echo "  $job_tag (4 init_times in parallel)"

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
            --wrap="bash ${helper}"

        count=$((count + 1))
    done
    echo "Phase 1 unperturbed: submitted $count jobs"
}

run_phase2() {
    local filter_model="${1:-}"
    echo "=== Phase 2: Layer-group sweep (with sqrt(N) variance scaling) ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue
        local sigma_full="${PHASE2_SIGMA_FULL[$model]}"
        local n_total="${N_TOTAL[$model]}"

        # Model-specific layer groups and per-group tensor counts via nameref
        local -n groups="PHASE2_GROUPS_${model}"
        local -n group_counts="N_GROUP_${model}"

        for init_time in "${INIT_TIMES[@]}"; do
            for idx in "${!groups[@]}"; do
                local layer_spec="${groups[$idx]}"
                local n_partial="${group_counts[$layer_spec]}"
                # sigma_partial = sigma_full * sqrt(n_total / n_partial)
                local sigma_scaled
                sigma_scaled=$(python3 -c \
                    "import math; print(f'{$sigma_full * math.sqrt($n_total / $n_partial):.6f}')")
                echo "  ${model} ${layer_spec}: N=${n_partial}/${n_total} -> sigma=${sigma_scaled}"
                submit_job "$model" "$init_time" "$sigma_scaled" "$layer_spec" "phase2" && ((count++)) || true
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

run_phase3() {
    local filter_model="${1:-}"
    echo "=== Phase 3: Coarse-scale-only perturbation (with sqrt(N) scaling) ==="
    local count=0
    for model in $MODELS; do
        [[ -n "$filter_model" && "$model" != "$filter_model" ]] && continue

        # SFNO uses a different mechanism: sub-axis mode-cut on spectral conv
        # weights, no named layer group. Branch here.
        if [[ "$model" == "sfno" ]]; then
            local mode_cut="${PHASE3_SFNO_MODE_CUT[$model]:-}"
            local sigma_scaled="${PHASE3_SFNO_SIGMA[$model]:-}"
            if [[ -z "$mode_cut" || -z "$sigma_scaled" ]]; then
                echo "  SKIP $model: Phase 3 mode-cut config missing"
                continue
            fi
            echo "  ${model}: --coarse-mode-cut=${mode_cut} sigma=${sigma_scaled}"
            for init_time in "${INIT_TIMES[@]}"; do
                # layer_spec "all": no --layer flag needed; --coarse-mode-cut
                # restricts the perturbed set to spectral conv tensors.
                submit_job "$model" "$init_time" "$sigma_scaled" "all" "phase3" \
                    "$NUM_MEMBERS" "$mode_cut" && ((count++)) || true
            done
            continue
        fi

        # GraphCast uses runtime activation perturbation (not weight
        # perturbation); sweep 4 sigmas to find the right magnitude.
        if [[ "$model" == "graphcast" ]]; then
            echo "  ${model}: --graph-coarse-nodes=${PHASE3_GC_NODES} sigmas=${PHASE3_GC_SIGMAS}"
            for init_time in "${INIT_TIMES[@]}"; do
                for gc_sigma in $PHASE3_GC_SIGMAS; do
                    # weight magnitude is 0 (no weight perturbation);
                    # layer_spec "all" unused; pass gc_sigma + gc_nodes.
                    submit_job "$model" "$init_time" "0" "all" "phase3" \
                        "$NUM_MEMBERS" "" "$gc_sigma" "$PHASE3_GC_NODES" \
                        && ((count++)) || true
                done
            done
            continue
        fi

        if [[ -z "${PHASE3_GROUP[$model]:-}" ]]; then
            echo "  SKIP $model: Phase 3 group not yet defined"
            continue
        fi
        local layer_spec="${PHASE3_GROUP[$model]}"
        local n_partial="${PHASE3_N_GROUP[$model]}"
        local n_total="${N_TOTAL[$model]}"
        local sigma_full="${PHASE3_SIGMA_FULL[$model]}"
        local sigma_scaled
        sigma_scaled=$(python3 -c \
            "import math; print(f'{$sigma_full * math.sqrt($n_total / $n_partial):.6f}')")
        echo "  ${model} ${layer_spec}: N=${n_partial}/${n_total} -> sigma=${sigma_scaled}"
        for init_time in "${INIT_TIMES[@]}"; do
            submit_job "$model" "$init_time" "$sigma_scaled" "$layer_spec" "phase3" && ((count++)) || true
        done
    done
    echo "Phase 3: submitted $count jobs"
}

# ---------------------------------------------------------------------------

PHASE="${1:-all}"
MODEL_FILTER="${2:-}"

case "$PHASE" in
    phase1)  run_phase1 "$MODEL_FILTER" ;;
    phase1_unperturbed) run_phase1_unperturbed "$MODEL_FILTER" ;;
    phase2)  run_phase2 "$MODEL_FILTER" ;;
    phase2b) run_phase2b "$MODEL_FILTER" ;;
    phase3)  run_phase3 "$MODEL_FILTER" ;;
    all)
        run_phase1 "$MODEL_FILTER"
        run_phase2 "$MODEL_FILTER"
        run_phase2b "$MODEL_FILTER"
        run_phase3 "$MODEL_FILTER"
        ;;
    *)
        echo "Usage: $0 {phase1|phase1_unperturbed|phase2|phase2b|phase3|all} [model]"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER | grep abl_"
echo "Logs: $LOG_DIR/"
echo ""
echo "Output layout:"
echo "  $STORE/ablation/{phase}/{model}/{init_date}/mag_{m}_layer_{l}/forecast.zarr"
