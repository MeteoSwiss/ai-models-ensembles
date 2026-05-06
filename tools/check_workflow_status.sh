#!/bin/bash
# Check status of the ai-models-ensembles workflow.
# Shows what data/outputs exist for the current configuration.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d ".venv" ] && [ -z "${CONTAINER_IMAGE:-}" ]; then
    source .venv/bin/activate
fi
source scripts/config.sh

echo "=========================================="
echo "Workflow Status Check"
echo "=========================================="
echo "  DATE_TIME:           $DATE_TIME"
echo "  MODEL_NAME:          $MODEL_NAME"
echo "  NUM_MEMBERS:         $NUM_MEMBERS"
echo "  LEAD_TIME:           ${LEAD_TIME}h"
echo "  PERTURBATION_INIT:   $PERTURBATION_INIT"
echo "  PERTURBATION_LATENT: $PERTURBATION_LATENT"
echo "  LAYER:               ${LAYER:-<all>}"
echo "  CROP_REGION:         $CROP_REGION"
echo "  DATA_SOURCE:         $DATA_SOURCE"
echo "  CONTAINER_IMAGE:     ${CONTAINER_IMAGE:-<host venv>}"
echo ""

check_path() {
    local path="$1"
    local description="$2"
    if [ -e "$path" ]; then
        local size
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "OK  $description"
        echo "    $path  ($size)"
    else
        echo "--  $description (not found)"
        echo "    expected: $path"
    fi
}

echo "=========================================="
echo "Step 1: Inputs"
echo "=========================================="
check_path "$TARGET_PATH"   "ERA5 verification target (TARGET_PATH)"
check_path "$IFS_ENS_PATH"  "IFS ENS baseline zarr (IFS_ENS_PATH)"
echo ""

echo "=========================================="
echo "Step 2: AI inference"
echo "=========================================="
echo "Sweeping PERTURBATION_LATENTS=${PERTURBATION_LATENTS}"
echo ""
have_any_forecast=0
for latent in $PERTURBATION_LATENTS; do
    pdir="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${latent}_layer_${LAYER}"
    if [ -d "$pdir/forecast.zarr" ]; then
        size=$(du -sh "$pdir/forecast.zarr" 2>/dev/null | cut -f1)
        echo "OK  latent=$latent  forecast.zarr  ($size)"
        have_any_forecast=1
    else
        echo "--  latent=$latent  forecast.zarr  (not found at $pdir)"
    fi
done
echo ""

echo "=========================================="
echo "Step 3: AI verification (SwissClim)"
echo "=========================================="
have_any_verify=0
for latent in $PERTURBATION_LATENTS; do
    rdir="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${latent}_layer_${LAYER}/${CROP_REGION}"
    sdir="$rdir/swissclim_${MODEL_NAME}"
    if [ -d "$sdir" ]; then
        echo "OK  latent=$latent  $sdir"
        have_any_verify=1
    else
        echo "--  latent=$latent  $sdir"
    fi
done
echo ""

echo "=========================================="
echo "Step 4: IFS ENS baseline verification"
echo "=========================================="
ifs_verify_dir="${OUTPUT_DIR}/${DATE_TIME}/_ifs_ens/${CROP_REGION}/swissclim_ifs_ens"
check_path "$ifs_verify_dir" "IFS ENS verify output"
have_ifs_verify=0
[ -d "$ifs_verify_dir" ] && have_ifs_verify=1
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
if [ $have_any_forecast -eq 0 ]; then
    echo "Status: no forecasts yet"
    echo "Next: sbatch scripts/submit_ml_inference.sh"
elif [ $have_any_verify -eq 0 ] && [ $have_ifs_verify -eq 0 ]; then
    echo "Status: forecasts present, no verification"
    echo "Next: sbatch scripts/submit_verification.sh"
else
    echo "Status: forecasts + verification in place"
    echo "Next: ai-ens intercompare"
fi
echo ""
