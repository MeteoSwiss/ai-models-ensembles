#!/usr/bin/env bash
# Validate environment and configuration before submitting jobs.
set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "$0")/.." || exit 1

source "./scripts/config.sh"

# Pull the canonical model registry from Python so this list does not drift.
KNOWN_MODELS=$(python - <<'PY' 2>/dev/null
try:
    from ai_models_ensembles.e2s_models import REGISTRY
    print(" ".join(sorted(REGISTRY)))
except Exception:
    pass
PY
)

validate_config() {
  mkdir -p "$LOG_DIR"
  [[ -d "$OUTPUT_DIR" ]] || mkdir -p "$OUTPUT_DIR"
  [[ -w "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR is not writable: $OUTPUT_DIR" >&2; exit 1; }
  [[ "$DATE_TIME" =~ ^[0-9]{12}$ ]] || { echo "DATE_TIME must be YYYYMMDDHHMM" >&2; exit 1; }
  if [[ -n "$KNOWN_MODELS" ]]; then
    case " $KNOWN_MODELS " in
      *" $MODEL_NAME "*) : ;;
      *) echo "Unknown MODEL_NAME=$MODEL_NAME. Known: $KNOWN_MODELS" >&2; exit 1 ;;
    esac
  fi
  require_cmd envsubst
  if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    [[ -f "$CONTAINER_IMAGE" ]] || {
      echo "CONTAINER_IMAGE=$CONTAINER_IMAGE not found" >&2; exit 1; }
  fi
}
validate_config

PY_VER_REQ="3.11"
PY_VER_ACT=$(python -c 'import sys; print("%d.%d"% (sys.version_info[0], sys.version_info[1]))')
if [[ "$PY_VER_ACT" != "$PY_VER_REQ"* ]]; then
  echo "Warning: Python $PY_VER_REQ.x is recommended (found $PY_VER_ACT)" >&2
fi

REQ_PKGS=(xarray zarr typer swissclim_evaluations earth2studio)
for pkg in "${REQ_PKGS[@]}"; do
  python - <<PY >/dev/null 2>&1 || { echo "Missing Python package: $pkg" >&2; exit 1; }
import importlib, sys
try:
    importlib.import_module("$pkg")
except Exception:
    sys.exit(1)
PY
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
[[ -w "$OUTPUT_DIR" ]] || { echo "OUTPUT_DIR is not writable: $OUTPUT_DIR" >&2; exit 1; }

python - <<'PY' >/dev/null || { echo "ai-ens CLI not importable" >&2; exit 1; }
import ai_models_ensembles.cli  # noqa: F401
PY

echo "Validation completed; please review any warnings above."
