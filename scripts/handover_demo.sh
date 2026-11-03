#!/usr/bin/env bash
# Small end-to-end check that a new user can run the pipeline: GraphCast with the
# production perturbation (graphcast_all: all weights, sigma 0.01), one init
# (2023-01-02 00 UTC), 2 members, 24 h, through the production submitter; then the
# SwissClim deterministic + probabilistic modules against ERA5; then a comparison with
# the production run of the same init (same seeds, so the members should agree up to
# GPU nondeterminism and the IC source). See docs/HANDOVER.md, section 3.
#
# All data comes from Clariden: the initial condition from the local ERA5 zarr (production
# used the same ERA5 from ARCO on GCS), the GraphCast weights from $STORE/e2s_cache_backup,
# the truth from the same ERA5 zarr, the reference from $STORE. Only the venv step downloads
# Python packages.
#
#   bash scripts/handover_demo.sh venv      # once: .venv in this checkout, built on a compute node (~2 min)
#   bash scripts/handover_demo.sh infer     # GPU job, ~4 min
#   bash scripts/handover_demo.sh eval      # after infer: SwissClim on the demo forecast (CPU job, ~7 min)
#   bash scripts/handover_demo.sh compare   # after infer: demo vs production, members 0-1 at +24 h
#   bash scripts/handover_demo.sh plot      # after eval: summary figure $DEMO_ROOT/demo_summary.png
#
# Needs Unix groups s83 (store, container images, weights) and weatherbench (ERA5), a Slurm
# account (AIENS_ACCOUNT, default ab016) and `uv` on PATH for the venv step.
# Outputs go to DEMO_ROOT (default $AIENS_SCRATCH/aiens_demo, purgeable).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export AIENS_SCRATCH="${AIENS_SCRATCH:-/iopsstor/scratch/cscs/$USER}"
STORE="${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}"
DEMO_ROOT="${DEMO_ROOT:-$AIENS_SCRATCH/aiens_demo}"
ACCOUNT="${AIENS_ACCOUNT:-ab016}"
PARTITION="${PARTITION:-debug}"
INIT_TAG=20230102_0000
FORECAST="$DEMO_ROOT/baselines/graphcast_all/$INIT_TAG/forecast.zarr"
PROD="$STORE/baselines/graphcast_all/$INIT_TAG/forecast.zarr"
WB2="${AIENS_WB2_22:-/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr}"

mkdir -p "$DEMO_ROOT/logs" "$AIENS_SCRATCH/tmp"

case "${1:-}" in
venv)
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --time=00:45:00 --cpus-per-task=16 --mem=64G \
        --job-name=aiens_demo_venv --output="$DEMO_ROOT/logs/venv_%j.log" \
        --wrap="set -e; cd '$SRC_DIR'
            uv venv --managed-python --python 3.11 .venv
            uv pip install --python .venv/bin/python -r env/host-requirements.txt
            uv pip install --python .venv/bin/python -e . --no-deps
            .venv/bin/python -c 'import swissclim_evaluations, ai_models_ensembles; print(\"venv ok\")'"
    ;;
infer)
    PER_INIT=1 WEEKS=2023-01-02 INITS="$INIT_TAG" LEAD_HOURS=24 NUM_MEMBERS=2 \
        DATA_SOURCE="swissclim:$WB2" E2S_CACHE="$STORE/e2s_cache_backup" \
        PARTITION="$PARTITION" PER_INIT_TIME_LIMIT=00:45:00 \
        OUT_ROOT="$DEMO_ROOT/baselines" LOG_DIR="$DEMO_ROOT/logs" \
        bash "$SRC_DIR/scripts/submit_all_inference.sh" graphcast_all
    echo "forecast -> $FORECAST, logs -> $DEMO_ROOT/logs"
    ;;
eval)
    [[ -d "$FORECAST" ]] || { echo "no forecast at $FORECAST, run 'infer' first" >&2; exit 1; }
    cfg="$DEMO_ROOT/eval_config.yaml"
    cat > "$cfg" <<YAML
paths:
  ml: ["$FORECAST"]
  nwp: ["$WB2"]
  output_root: "$DEMO_ROOT/eval"

selection:
  levels: [500, 850]
  temporal_resolution_hours: null
  datetimes: ["2023-01-02T00:00:2023-01-03T00"]
  latitudes: [90.0, -89.75]
  longitudes: [0.0, 359.75]
  variables_2d: ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind"]
  variables_3d: ["geopotential", "temperature"]
  ensemble_members: null
  ensemble:
    deterministic: mean
    probabilistic: prob
  check_missing: false

plotting:
  random_seed: 42
  plot_datetime: null
  plot_ensemble_members: [0]
  dpi: 48
  output_mode: both
  probabilistic_line_plots: true

modules:
  maps: false
  histograms: false
  wd_kde: false
  energy_spectra: false
  vertical_profiles: false
  deterministic: true
  ets: false
  fss: false
  probabilistic: true
  ssim: false
  multivariate: false

lead_time:
  mode: stride
  stride:
    hours: 6
  max_hour: 24
  panel:
    strategy: evenly_spaced
    count: 5

metrics:
  deterministic:
    include: ["MAE", "RMSE"]
    standardized_include: ["MAE", "RMSE"]
    report_per_level: true
    error_maps: false

performance:
  dask_scheduler: distributed
  dask_profile: balanced
YAML
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --time=00:45:00 --cpus-per-task=64 --mem=256G \
        --job-name=aiens_demo_eval --output="$DEMO_ROOT/logs/eval_%j.log" \
        --wrap="source '$SRC_DIR/.venv/bin/activate'
            export TMPDIR='$AIENS_SCRATCH/tmp' DASK_TEMPORARY_DIRECTORY='$AIENS_SCRATCH/tmp'
            python -m swissclim_evaluations.cli --config '$cfg'"
    echo "config -> $cfg, scores -> $DEMO_ROOT/eval"
    ;;
compare)
    "$SRC_DIR/.venv/bin/python" - "$FORECAST" "$PROD" <<'PY'
import sys

import numpy as np
import xarray as xr

demo, prod = (xr.open_zarr(p) for p in sys.argv[1:3])
lead = demo.lead_time.values[-1]
hours = int(lead / np.timedelta64(1, "h"))
for var in ["2m_temperature", "mean_sea_level_pressure"]:
    a = demo[var].sel(lead_time=lead).squeeze("init_time").load()
    b = prod[var].sel(lead_time=lead, ensemble=a.ensemble.values).squeeze("init_time").load()
    spread = float(np.sqrt(((a.isel(ensemble=0) - a.isel(ensemble=1)) ** 2).mean()))
    print(f"{var} +{hours} h: RMS(member 0 - member 1) = {spread:.4g}")
    for m in a.ensemble.values:
        d = float(np.sqrt(((a.sel(ensemble=m) - b.sel(ensemble=m)) ** 2).mean()))
        print(f"  member {m}: RMS(demo - production) = {d:.4g}")
PY
    ;;
plot)
    "$SRC_DIR/.venv/bin/python" - "$FORECAST" "$DEMO_ROOT/eval" "$DEMO_ROOT/demo_summary.png" <<'PY'
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

forecast, ev, png = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
t2m = xr.open_zarr(forecast)["2m_temperature"].isel(init_time=0).sel(lead_time=pd.Timedelta(hours=24))
fig, axes = plt.subplots(1, 3, figsize=(17, 3.6))
t2m.sel(ensemble=0).plot(ax=axes[0], cmap="RdYlBu_r")
axes[0].set_title("2m temperature, member 0, +24 h")
(t2m.sel(ensemble=0) - t2m.sel(ensemble=1)).plot(ax=axes[1], cmap="RdBu_r", robust=True)
axes[1].set_title("member 0 - member 1 (spread from the weight noise)")
crps = pd.read_csv(ev / "probabilistic" / "crps_line_2m_temperature_by_lead_ensprob.csv")
rmse = pd.read_csv(next((ev / "deterministic").glob("det_line_2m_temperature_RMSE_by_lead_*.csv")))
axes[2].plot(crps.lead_time_hours, crps.CRPS, marker="o", label="CRPS")
axes[2].plot(rmse.lead_time_hours, rmse.RMSE, marker="o", label="RMSE of the ensemble mean")
axes[2].set(xlabel="lead time (h)", ylabel="K", title="2m temperature vs ERA5 (SwissClim)")
axes[2].legend()
fig.tight_layout()
fig.savefig(png, dpi=100)
print(png)
PY
    ;;
*)
    sed -n '2,22p' "$0"
    exit 1
    ;;
esac
