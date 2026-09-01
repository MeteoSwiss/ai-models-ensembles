#!/usr/bin/env bash
# SLURM array job: track Milton across a set of baselines x the 14 Milton-week
# inits (7 days x 2 hours). One array task = one (baseline, init): 10 members
# sequentially, ~4-6 min wall, ~50 GB peak. ifs_ens has its own pipeline.
#
# Usage: submit_milton_tracker.sh [arm]
#   weight    (default)  weight-only + trained-prob baselines (6)
#   ic_only               IC-only arms (4)          -- review item 1
#   phase5                perturbed-IC arms (4)     -- Phase 5
# Each arm writes to disjoint TRACKS_ROOT/<baseline>/<init_tag>/ dirs.
set -euo pipefail

ARM="${1:-weight}"
case "$ARM" in
    weight)  BASELINES=(aurora_encoder graphcast_all sfno_modes10 aifsens fcn3 atlas) ;;
    ic_only) BASELINES=(aurora_ic_only graphcast_ic_only sfno_ic_only aifs_ic_only) ;;
    phase5)  BASELINES=(aurora_encoder_ic graphcast_all_ic sfno_modes10_ic aifs_perturbed_ic) ;;
    *) echo "unknown arm '$ARM' (want: weight | ic_only | phase5)" >&2; exit 2 ;;
esac
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
echo "arm=$ARM  submitting $N array tasks (${#BASELINES[@]} baselines x 7 days x 2 hours)"

SCRATCH=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/sadamov}
STORE=${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}
REPO=${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}
PY=${AIENS_PY:-/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python}

LIST="$SCRATCH/milton_case_study/pair_list_${ARM}.txt"
printf "%s\n" "${PAIRS[@]}" > "$LIST"
echo "wrote $LIST"
mkdir -p "$STORE/ablation_logs"

sbatch --parsable \
    --job-name="milton_track_${ARM}" \
    --partition=normal --account=ab016 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
    --time=01:00:00 \
    --array=1-${N}%10 \
    --output="$STORE/ablation_logs/milton_track_${ARM}_%A_%a.log" \
    --wrap="
        source $REPO/tools/milton/env.sh
        LINE=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $LIST)
        BASELINE=\$(echo \$LINE | cut -d'|' -f1)
        INIT=\$(echo \$LINE | cut -d'|' -f2)
        echo \"task \$SLURM_ARRAY_TASK_ID  baseline=\$BASELINE  init=\$INIT\"
        $PY $REPO/tools/milton/track_one_init.py \$BASELINE \$INIT
    "
