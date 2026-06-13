#!/usr/bin/env bash
# Phase 5 Milton re-run: track the 4 perturbed-IC baselines across the 14
# Milton-week inits. Mirrors submit_milton_tracker.sh exactly except for the
# BASELINES list. Tracks are written to TRACKS_ROOT/<baseline_ic>/<init_tag>/
# via track_one_init.py's existing logic -- the _ic baselines go to disjoint
# directory names so the original weight-only tracks are preserved.
set -euo pipefail

BASELINES=(aurora_encoder_ic graphcast_all_ic sfno_modes10_ic aifs_perturbed_ic)
DAYS=(02 03 04 05 06 07 08)
HOURS=(0000 1200)

PAIRS=()
for b in "${BASELINES[@]}"; do
    for d in "${DAYS[@]}"; do
        for h in "${HOURS[@]}"; do
            PAIRS+=("${b}|202410${d}_${h}")
        done
    done
done

N=${#PAIRS[@]}
echo "Submitting $N Phase 5 tracker array tasks (4 baselines x 7 days x 2 hours = 56)"

LIST=/iopsstor/scratch/cscs/sadamov/milton_case_study/pair_list_phase5.txt
printf "%s\n" "${PAIRS[@]}" > "$LIST"
echo "wrote $LIST"

mkdir -p /capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs

sbatch --parsable \
    --job-name=milton_track_p5 \
    --partition=normal --account=a122 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
    --time=01:00:00 \
    --array=1-${N}%10 \
    --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs/milton_track_p5_%A_%a.log \
    --wrap="
        source /users/sadamov/pyprojects/ai-models-ensembles/tools/milton/env.sh
        LINE=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $LIST)
        BASELINE=\$(echo \$LINE | cut -d'|' -f1)
        INIT=\$(echo \$LINE | cut -d'|' -f2)
        echo \"task \$SLURM_ARRAY_TASK_ID  baseline=\$BASELINE  init=\$INIT\"
        /iopsstor/scratch/cscs/sadamov/venvs/milton_case/bin/python \
            /users/sadamov/pyprojects/ai-models-ensembles/tools/milton/track_one_init.py \
            \$BASELINE \$INIT
    "
