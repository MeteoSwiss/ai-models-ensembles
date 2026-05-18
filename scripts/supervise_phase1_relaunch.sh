#!/usr/bin/env bash
#SBATCH --job-name=supervise_phase1_relaunch
#SBATCH --partition=normal
#SBATCH --account=a122
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles/ablation_logs/supervise_phase1_relaunch_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles/ablation_logs/supervise_phase1_relaunch_%j.err
#
# Supervisor: runs *after* all SFNO + GraphCast Phase 1 inference jobs finish.
# Submits eval (per-model) and then intercomparison (per-model).
#
# Usage:
#   sbatch --dependency=afterany:<id1>:<id2>:... scripts/supervise_phase1_relaunch.sh
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[supervise] $(date) -- triggered after inference completion"

# Submit eval per model (no AFTER_JOB needed; inference is already done).
for model in sfno graphcast; do
    echo "[supervise] eval for $model"
    bash "$SRC_DIR/scripts/evaluate_ablation.sh" phase1 "$model" 2>&1 \
        | tee "/tmp/eval_${model}_$$.log"
done

# Collect eval job IDs per model from the log.
declare -A EVAL_IDS
for model in sfno graphcast; do
    ids=$(grep -oE '^[0-9]+$' "/tmp/eval_${model}_$$.log" | paste -sd:)
    EVAL_IDS[$model]="$ids"
    echo "[supervise] eval IDs for $model: ${EVAL_IDS[$model]}"
done

# Submit intercomparison per model with afterok on that model's eval.
for model in sfno graphcast; do
    ids="${EVAL_IDS[$model]:-}"
    echo "[supervise] intercompare for $model (afterany:$ids)"
    AFTER_JOB="$ids" bash "$SRC_DIR/scripts/evaluate_ablation.sh" intercompare phase1 "$model"
done

echo "[supervise] done"
