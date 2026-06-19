#!/usr/bin/env bash
# Regenerate three Hurricane Milton figures that read forecast.zarr:
#   - F4 storm-relative warm-core composite (milton_F4_storm_relative_composite.pdf)
#     at init 2024-10-05 00 UTC, lead 120 h (valid 2024-10-10 00 UTC). At this init
#     IFS-ENS MSL is present, so ALL baselines (incl. IFS-ENS) use the same MSL-min
#     centring -- removes the methodological inconsistency the reviewer flagged.
#   - F3 cascading-detection panels for all 9 baselines.
#   - F8 member/mean/spread 10 m wind maps, init 2024-10-06 00 UTC, lead 96 h.
# All with in-panel matplotlib title()/suptitle() removed (info lives in captions).
#
# Reads only forecast.zarr + the existing era5_milton_window.nc; no tracking, no GPU.
# The F4 warm-core peak numbers are printed to the log (look for the F4 NUMBERS block)
# so the caption + body prose can be updated from them once the job finishes.
#SBATCH --job-name=milton_figs_regen
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/users/sadamov/pyprojects/ai-models-ensembles/logs/milton_figs_regen_%j.log

set -euo pipefail

REPO=/users/sadamov/pyprojects/ai-models-ensembles
PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python

export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"
# cartopy may fetch coastline/border shapefiles on first run on a compute node.
export SSL_CERT_FILE=$("$PY" -c 'import certifi; print(certifi.where())')

echo "host=$(hostname) start=$(date -Is)"
echo "python=$PY"

cd "$REPO"

# ------------------------------------------------------------------ F4 (new init)
echo "========== F4 NUMBERS (init 20241005_0000, lead 120h, valid 2024-10-10 00 UTC) =========="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "/users/sadamov/pyprojects/ai-models-ensembles/tools/milton")
import figures_milton as fm
# Default init_tag is now 20241005_0000; prints ERA5 peak + per-baseline
# peak (m) and suppression (% vs ERA5) for every baseline incl. IFS-ENS.
fm.f4_storm_relative_composite()
PYEOF
echo "========== END F4 NUMBERS =========="

# ----------------------------------------------------------------- F3 (all panels)
echo "========== F3 cascading-detection panels (9 baselines) =========="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "/users/sadamov/pyprojects/ai-models-ensembles/tools/milton")
import figures_milton as fm
import pandas as pd
master = pd.read_csv(fm.BASE / "milton_master_tracks.csv", parse_dates=["time"])
for b in [
    "ifs_ens",
    "aifs_perturbed",
    "aifs_perturbed_ic",
    "aifsens",
    "aurora_encoder_ic",
    "graphcast_all_ic",
    "sfno_modes10_ic",
    "fcn3",
    "atlas",
]:
    print(f"--- F3 {b} ---")
    fm.f3_cascading_detection(b)
PYEOF

# ----------------------------------------------------------------- F8 (spread maps)
echo "========== F8 member/mean/spread maps (init 20241006_0000, lead 96h) =========="
"$PY" "$REPO/tools/plot_milton_member_spread_maps.py" \
    --init 20241006_0000 \
    --lead-h 96 \
    --models aifs_perturbed_ic ifs_ens aifsens \
    --out milton_F8_member_spread_maps_96h

echo "done=$(date -Is)"
