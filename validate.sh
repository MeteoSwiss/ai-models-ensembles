#!/usr/bin/env bash
# Validate environment and configuration before submitting jobs
set -euo pipefail
IFS=$'\n\t'

# Load config and helpers
source "./config.sh"

# Base config validation
validate_config

# Python and package checks
PY_VER_REQ="3.10"
PY_VER_ACT=$(python -c 'import sys; print("%d.%d"% (sys.version_info[0], sys.version_info[1]))')
if [[ "$PY_VER_ACT" != "$PY_VER_REQ"* ]]; then
  echo "Warning: Python $PY_VER_REQ.x is recommended (found $PY_VER_ACT)" >&2
fi

# Required Python packages
REQ_PKGS=(
  xarray zarr numcodecs seaborn scores typer
)
for pkg in "${REQ_PKGS[@]}"; do
  python - <<PY >/dev/null 2>&1 || { echo "Missing Python package: $pkg" >&2; exit 1; }
import importlib, sys
try:
    importlib.import_module("$pkg")
except Exception as e:
    sys.exit(1)
PY

done

# ai-models CLI available
require_cmd ai-models

# Earthkit optional check
python - <<'PY' >/dev/null 2>&1 || echo "Note: earthkit.data not found; downloads may be limited" >&2
import importlib
importlib.import_module("earthkit.data")
PY

# eccodes/GRIB tools
if ! command -v grib_ls >/dev/null 2>&1; then
  echo "Warning: eccodes (grib_ls) not found; GRIB operations may fail" >&2
fi

# ImageMagick
if ! command -v convert >/dev/null 2>&1 && ! command -v magick >/dev/null 2>&1; then
  echo "Warning: ImageMagick not found; GIF creation may fail" >&2
fi

# Optional fd for cleanup acceleration
if ! command -v fd >/dev/null 2>&1; then
  echo "Note: fd not found; GRIB cleanup will be slower (fallback to find)" >&2
fi

# ECMWF credentials
if [[ ! -f "$HOME/.cdsapirc" ]]; then
  echo "Warning: ECMWF CDS credentials not found at ~/.cdsapirc; downloads will fail" >&2
fi

# Earthkit cache dir info
if [[ -n "${EARTHKIT_CACHE_DIR:-}" ]]; then
  echo "Using EARTHKIT_CACHE_DIR=$EARTHKIT_CACHE_DIR"
fi

# Output dir
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
[[ -w "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR is not writable: $OUTPUT_DIR" >&2; exit 1; }

# CLI help smoke test
python - <<'PY' >/dev/null || { echo "ai-ens CLI not importable" >&2; exit 1; }
import importlib
import ai_models_ensembles.cli as cli
PY

echo "Validation OK. Environment looks good."
