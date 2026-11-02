#!/usr/bin/env bash
# Rebuild the Milton case study end to end on ONE node:
#   1. track all (baseline, init) pairs, PAR at a time (track_one_init.py; existing
#      per-member stitch files are reused, so a rerun only does what is missing)
#   2. ERA5 control track + ERA5 window netCDF (era5_control.py)
#   3. master + verification tables (aggregate_tracks.py), summary CSV (analyze_milton_stats.py)
#   4. paper figures F1-F5 (figures_milton.py) and F9 (plot_aifs_wt_vs_ic_spread.py)
# Everything is written under $AIENS_STORE/analysis/milton_case_study (durable, unlike scratch),
# figures into figures/. Replaces the one-node-per-pair array of submit_milton_tracker.sh:
# nodes are allocated exclusively, so the array paid a whole node for every 2-4 min task.
#
#   sbatch --account=${AIENS_ACCOUNT:-ab016} tools/milton/submit_milton_rebuild.sh
#   PAR=16 SKIP_FIGS=1 sbatch --account=... tools/milton/submit_milton_rebuild.sh
#
#SBATCH --job-name=milton_rebuild
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=750G
#SBATCH --output=%x_%j.log

set -uo pipefail

REPO=${AIENS_REPO:-/users/sadamov/pyprojects/ai-models-ensembles}
STORE=${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}
TMP_ROOT=${AIENS_SCRATCH:-/iopsstor/scratch/cscs/$USER}
PAR=${PAR:-20}

cd "$REPO"
source tools/milton/env.sh
export AIENS_SCRATCH=$STORE/analysis   # tools/_env.py: BASE = $AIENS_SCRATCH/milton_case_study
BASE=$AIENS_SCRATCH/milton_case_study
export TMPDIR=$TMP_ROOT/tmp DASK_TEMPORARY_DIRECTORY=$TMP_ROOT/tmp MPLBACKEND=Agg PYTHONUNBUFFERED=1
export SSL_CERT_FILE=$("$MILTON_PYTHON" -c 'import certifi; print(certifi.where())')
mkdir -p "$TMPDIR" "$BASE/logs"

BASELINES=(
    aurora_encoder graphcast_all sfno_modes10 aifs_perturbed aifsens fcn3 atlas
    aurora_ic_only graphcast_ic_only sfno_ic_only aifs_ic_only
    aurora_encoder_ic graphcast_all_ic sfno_modes10_ic aifs_perturbed_ic
    ifs_ens
)
PAIRS=()
for b in "${BASELINES[@]}"; do
    for d in 02 03 04 05 06 07 08; do
        for h in 0000 1200; do PAIRS+=("$b 202410${d}_${h}"); done
    done
done

echo "$(date -Is) tracking ${#PAIRS[@]} pairs, $PAR at a time -> $BASE/tracks"
export BASE
# one line per pair: xargs passes "<baseline> <init>" as $1 $2 ($0 is the placeholder "_")
printf '%s\n' "${PAIRS[@]}" | xargs -P "$PAR" -L 1 sh -c \
    '"$MILTON_PYTHON" tools/milton/track_one_init.py "$1" "$2" > "$BASE/logs/track_$1_$2.log" 2>&1' _ || true
incomplete=$(for p in "${PAIRS[@]}"; do set -- $p; grep -q "^done $1 $2" "$BASE/logs/track_$1_$2.log" 2>/dev/null || echo "$1_$2"; done)
member_fail=$(grep -l ': FAILED' "$BASE"/logs/track_*.log 2>/dev/null)
echo "$(date -Is) tracking finished; incomplete pairs: $(echo "$incomplete" | grep -c .); logs with member failures: $(echo "$member_fail" | grep -c .)"
if [[ -n "$incomplete" || -n "$member_fail" ]]; then
    echo "incomplete: $incomplete"; echo "member failures in: $member_fail"
    echo "fix and resubmit (finished members are reused)"; exit 1
fi

echo "$(date -Is) ERA5 control"
"$MILTON_PYTHON" tools/milton/era5_control.py || exit 1
echo "$(date -Is) aggregate + stats"
"$MILTON_PYTHON" tools/milton/aggregate_tracks.py || exit 1
"$MILTON_PYTHON" tools/milton/analyze_milton_stats.py || exit 1

if [[ "${SKIP_FIGS:-0}" != 1 ]]; then
    echo "$(date -Is) figures"
    "$MILTON_PYTHON" - <<'EOF' || exit 1
import sys
sys.path[:0] = ["tools/milton", "tools"]
import pandas as pd
import figures_milton as fm
master = pd.read_csv(fm.BASE / "milton_master_tracks.csv", parse_dates=["time"])
verif = pd.read_csv(fm.BASE / "milton_verification.csv", parse_dates=["time"])
fm.f1_track_spaghetti(master)
fm.f2_intensity_vs_lead(master)
fm.f3_cascading_combined()
fm.f4_storm_relative_composite()
fm.f5_track_intensity_err_vs_lead(verif)
EOF
    "$MILTON_PYTHON" tools/milton/plot_aifs_wt_vs_ic_spread.py || exit 1
fi
echo "$(date -Is) done"
