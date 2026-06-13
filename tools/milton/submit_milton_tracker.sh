#!/usr/bin/env bash
# SLURM array job: track Milton across all 6 baselines x 14 inits.
# (ifs_ens has its own pipeline -- handled separately later)
#
# Total array tasks: 6 baselines x 14 inits = 84.
# Each task: 10 members sequentially, ~4-6 min wall, ~50 GB peak memory.
set -euo pipefail

BASELINES=(aurora_encoder graphcast_all sfno_modes10 aifsens fcn3 atlas)
DAYS=(02 03 04 05 06 07 08)
HOURS=(0000 1200)

# Build the (baseline, init_tag) pair list
PAIRS=()
for b in "${BASELINES[@]}"; do
    for d in "${DAYS[@]}"; do
        for h in "${HOURS[@]}"; do
            PAIRS+=("${b}|20241004_$((10#$d * 100))${h}")
        done
    done
done

# Above produces wrong format -- rebuild
PAIRS=()
for b in "${BASELINES[@]}"; do
    for d in "${DAYS[@]}"; do
        for h in "${HOURS[@]}"; do
            PAIRS+=("${b}|202410${d}_${h}")
        done
    done
done

N=${#PAIRS[@]}
echo "Submitting $N array tasks (6 baselines x 7 days x 2 hours = 84)"

# Write the pair list to a file
LIST=/iopsstor/scratch/cscs/sadamov/milton_case_study/pair_list.txt
printf "%s\n" "${PAIRS[@]}" > "$LIST"
echo "wrote $LIST"

mkdir -p /capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs

sbatch --parsable \
    --job-name=milton_track \
    --partition=normal --account=a122 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
    --time=01:00:00 \
    --array=1-${N}%10 \
    --output=/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation_logs/milton_track_%A_%a.log \
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
