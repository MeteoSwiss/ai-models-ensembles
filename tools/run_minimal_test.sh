#!/bin/bash
# Minimal sanity test for ai-models-ensembles. No GPU, no network.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "ai-models-ensembles - minimal sanity test"
echo "========================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run: bash tools/setup_uv.sh"
    exit 1
fi
source .venv/bin/activate

if [ ! -f "scripts/config.sh" ]; then
    echo "scripts/config.sh not found"
    exit 1
fi
source scripts/config.sh

echo "Step 1: Activated venv"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATE_TIME:  $DATE_TIME"
echo "  MODEL_NAME: $MODEL_NAME"
echo ""

echo "Step 2: CLI help"
ai-ens --help > /dev/null
echo "  ai-ens --help: OK"
echo ""

echo "Step 3: Registered models"
ai-ens models
echo ""

echo "Step 4: earth2studio + swissclim_evaluations import"
python - <<'PY'
import importlib, sys
ok = True
for mod in ("earth2studio", "swissclim_evaluations", "ai_models_ensembles.e2s_inference"):
    try:
        importlib.import_module(mod)
        print(f"  {mod}: OK")
    except Exception as exc:
        print(f"  {mod}: FAIL ({exc})")
        ok = False
sys.exit(0 if ok else 1)
PY
echo ""

echo "All checks passed. See tools/QUICKSTART_TEST.md for the full workflow."
