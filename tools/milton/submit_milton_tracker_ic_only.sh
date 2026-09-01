#!/usr/bin/env bash
# IC-only Milton re-run: track the 4 IC-only baselines across the 14 Milton-week
# inits. Mirrors submit_milton_tracker_phase5.sh except for the BASELINES list.
# Completes the weight-only / IC-only / weight+IC triptych for the case study, so
# the Milton dispersion gain can be attributed to the IC perturbation rather than
# assumed (review item 1). Tracks go to TRACKS_ROOT/<baseline>/<init_tag>/.
set -euo pipefail

BASELINES=(aurora_ic_only graphcast_ic_only sfno_ic_only aifs_ic_only)
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
echo "Submitting $N IC-only tracker array tasks (4 baselines x 7 days x 2 hours = 56)"

LIST=/iopsstor/scratch/cscs/sadamov/milton_case_study/pair_list_iconly.txt
printf "%s\n" "${PAIRS[@]}" > "$LIST"
echo "wrote $LIST"

mkdir -p /capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs

sbatch --parsable \
    --job-name=milton_track_ic \
    --partition=normal --account=ab016 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
    --time=01:00:00 \
    --array=1-${N}%10 \
    --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs/milton_track_ic_%A_%a.log \
    --wrap="
        source /users/sadamov/pyprojects/ai-models-ensembles/tools/milton/env.sh
        LINE=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $LIST)
        BASELINE=\$(echo \$LINE | cut -d'|' -f1)
        INIT=\$(echo \$LINE | cut -d'|' -f2)
        echo \"task \$SLURM_ARRAY_TASK_ID  baseline=\$BASELINE  init=\$INIT\"
        /capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python \
            /users/sadamov/pyprojects/ai-models-ensembles/tools/milton/track_one_init.py \
            \$BASELINE \$INIT
    "
