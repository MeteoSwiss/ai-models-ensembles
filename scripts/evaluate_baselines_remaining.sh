#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DEPRECATED 2026-05-27 -- the per-module submission logic has been folded
# into `evaluate_baselines.sh`. The `all` action there now does per-module
# submission directly (one sbatch per (model, module), --mem=800G, 12h).
#
# This file is kept as a thin shim so old invocations keep working.
#
# Old:
#   bash scripts/evaluate_baselines_remaining.sh
#   MODELS=aurora_encoder ENERGY_SPECTRA_REDO=aurora_encoder bash ... remaining.sh
# New:
#   bash scripts/evaluate_baselines.sh all
#   bash scripts/evaluate_baselines.sh all aurora_encoder
# ---------------------------------------------------------------------------
set -euo pipefail
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[evaluate_baselines_remaining.sh] deprecated -- forwarding to 'evaluate_baselines.sh all'."

# Iterate MODELS (if set) so the per-module submission runs scoped per model.
if [[ -n "${MODELS:-}" ]]; then
    for m in $MODELS; do
        bash "$SRC_DIR/scripts/evaluate_baselines.sh" all "$m"
    done
else
    bash "$SRC_DIR/scripts/evaluate_baselines.sh" all
fi
