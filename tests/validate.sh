#!/usr/bin/env bash
# Validate environment and configuration before submitting jobs
set -euo pipefail
IFS=$'\n\t'

# Change to repository root
cd "$(dirname "$0")/.."

# Load config and helpers
source "./config.sh"

# Validation function (moved from config.sh)
validate_config() {
  mkdir -p "$LOG_DIR"
  [[ -d "$OUTPUT_DIR" ]] || mkdir -p "$OUTPUT_DIR"
  [[ -w "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR is not writable: $OUTPUT_DIR" >&2; exit 1; }
  [[ "$DATE_TIME" =~ ^[0-9]{12}$ ]] || { echo "DATE_TIME must be YYYYMMDDHHMM" >&2; exit 1; }
  case "$MODEL_NAME" in
    graphcast|fourcastnetv2-small|gencast) : ;;
    *) echo "Unknown MODEL_NAME: $MODEL_NAME" >&2; exit 1 ;;
  esac
  # NUM_MEMBERS sanity
  if [[ "${NUM_MEMBERS}" -gt 50 ]]; then
    echo "NUM_MEMBERS=${NUM_MEMBERS} exceeds 50 (IFS ensemble limit). Reduce to <= 50." >&2
    exit 1
  fi
  # External tools
  require_cmd ai-models
  require_cmd bc
  if command -v convert >/dev/null 2>&1 || command -v magick >/dev/null 2>&1; then :; else echo "Warning: ImageMagick not found; GIF creation may fail" >&2; fi
  # ECMWF credentials (MARS)
  if [[ ! -f "$HOME/.ecmwfapirc" ]]; then
    echo "Warning: ECMWF MARS credentials not found at ~/.ecmwfapirc; downloads will fail." >&2
  fi
}

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

# Optional ECMWF MARS connectivity probe (only if credentials exist)
if [[ -f "$HOME/.ecmwfapirc" ]]; then
  # Skip if earthkit.data isn't installed
  if python - <<'PY' >/dev/null 2>&1; then
import importlib
importlib.import_module("earthkit.data")
PY
  python - <<'PY' || true
import os, shutil
try:
  import earthkit.data as ekd
  from earthkit.data import settings
  tmp = os.path.join(os.getenv("TMPDIR", "/tmp"), "earthkit_mars_check")
  os.makedirs(tmp, exist_ok=True)
  try:
    settings.set("user-cache-directory", tmp)
  except Exception:
    pass
  # Small metadata-only request to test auth/connectivity
  req = {
    "class": "od",
    "date": "2023-01-01",
    "expver": "1",
    "levtype": "sfc",
    "param": "2t",
    "step": "0",
    "stream": "oper",
    "time": "00",
    "type": "fc",
    "area": [50, 0, 40, 10],
    "grid": [1, 1],
  }
  ds = ekd.from_source("mars", req, lazily=True)
  try:
    _ = ds.metadata()
  except Exception:
    # Some versions may fetch lazily; if it raises, it's fine to report failure below
    pass
  print("MARS connectivity: OK")
except ImportError:
  print("Note: earthkit.data not installed; skipping MARS check")
except Exception as e:
  print(f"Warning: MARS connectivity check failed: {e}")
finally:
  try:
    shutil.rmtree(tmp, ignore_errors=True)
  except Exception:
    pass
PY
  else
  echo "Note: earthkit.data not installed; skipping MARS check" >&2
  fi
else
  echo "Note: ECMWF MARS credentials not found at ~/.ecmwfapirc; skipping MARS connectivity check" >&2
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
if [[ ! -f "$HOME/.ecmwfapirc" ]]; then
  echo "Warning: ECMWF CDS credentials not found at ~/.ecmwfapirc; downloads will fail" >&2
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

echo "Validation completed; please review any warnings above."
