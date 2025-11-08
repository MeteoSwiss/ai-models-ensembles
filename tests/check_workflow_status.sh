#!/bin/bash
#
# Check status of the ai-models-ensembles workflow
# Shows what data/outputs exist for the current configuration
#
set -euo pipefail

# Change to repository root
cd "$(dirname "$0")/.."

# Load configuration
source .venv/bin/activate
source config.sh

echo "=========================================="
echo "Workflow Status Check"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  DATE_TIME:          $DATE_TIME"
echo "  MODEL_NAME:         $MODEL_NAME"
echo "  NUM_MEMBERS:        $NUM_MEMBERS"
echo "  PERTURBATION_INIT:  $PERTURBATION_INIT"
echo "  PERTURBATION_LATENT: $PERTURBATION_LATENT"
echo "  LAYER:              $LAYER"
echo "  CROP_REGION:        $CROP_REGION"
echo ""

# Helper function to check file/dir exists
check_path() {
    local path="$1"
    local description="$2"
    
    if [ -e "$path" ]; then
        if [ -d "$path" ]; then
            local size=$(du -sh "$path" 2>/dev/null | cut -f1)
            echo "✓ $description"
            echo "  Location: $path"
            echo "  Size: $size"
        else
            local size=$(ls -lh "$path" 2>/dev/null | awk '{print $5}')
            echo "✓ $description"
            echo "  Location: $path"
            echo "  Size: $size"
        fi
        return 0
    else
        echo "✗ $description (not found)"
        echo "  Expected: $path"
        return 1
    fi
}

count_files() {
    local dir="$1"
    local pattern="$2"
    if [ -d "$dir" ]; then
        find "$dir" -name "$pattern" 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

echo "=========================================="
echo "Step 1: Initial Conditions (ERA5)"
echo "=========================================="
check_path "$MODEL_DIR/init_field.grib" "Initial field GRIB"
check_path "$MODEL_DIR/ground_truth.zarr" "Ground truth Zarr"
echo ""

echo "=========================================="
echo "Step 2: Reference Data (IFS)"
echo "=========================================="
check_path "$MODEL_DIR/ifs_ens.zarr" "IFS Ensemble Zarr"
check_path "$MODEL_DIR/ifs_control.zarr" "IFS Control Zarr"
echo ""

echo "=========================================="
echo "Step 3: Model Fields Configuration"
echo "=========================================="
check_path "$MODEL_DIR/fields.txt" "Fields file"
if [ -f "$MODEL_DIR/fields.txt" ]; then
    echo "  Total fields: $(wc -l < "$MODEL_DIR/fields.txt")"
fi
echo ""

echo "=========================================="
echo "Step 4: Ensemble Members"
echo "=========================================="
echo "Perturbation directory: $PERTURBATION_DIR"
if [ -d "$PERTURBATION_DIR" ]; then
    echo "✓ Perturbation directory exists"
    
    # Count member directories
    member_count=0
    for i in $(seq 0 $((NUM_MEMBERS - 1))); do
        if [ -d "$PERTURBATION_DIR/$i" ]; then
            member_count=$((member_count + 1))
        fi
    done
    
    echo "  Member directories: $member_count / $NUM_MEMBERS"
    
    # Check a few members for completeness
    echo ""
    echo "  Sample member status:"
    for i in 0 1 2; do
        if [ $i -lt $NUM_MEMBERS ]; then
            member_dir="$PERTURBATION_DIR/$i"
            if [ -d "$member_dir" ]; then
                has_init=$([ -f "$member_dir/init_field.grib" ] && echo "✓" || echo "✗")
                has_params=$([ -d "$member_dir/params" ] && echo "✓" || echo "✗")
                has_output=$([ -d "$PERTURBATION_DIR/forecast.zarr/member/$i" ] && echo "✓" || echo "✗")
                echo "    Member $i: init=$has_init params=$has_params output=$has_output"
            else
                echo "    Member $i: ✗ (not created)"
            fi
        fi
    done
else
    echo "✗ Perturbation directory not found"
fi
echo ""

echo "=========================================="
echo "Step 5: Forecast Output"
echo "=========================================="
check_path "$PERTURBATION_DIR/forecast.zarr" "Ensemble forecast Zarr"
if [ -d "$PERTURBATION_DIR/forecast.zarr" ]; then
    # Check how many members have data
    member_dirs=$(find "$PERTURBATION_DIR/forecast.zarr/member" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Members with forecast data: $member_dirs"
fi
echo ""

echo "=========================================="
echo "Step 6: Verification & Plots"
echo "=========================================="
echo "Region directory: $REGION_DIR"
if [ -d "$REGION_DIR" ]; then
    echo "✓ Region directory exists"
    
    # Check for plots
    png_count=$(count_files "$REGION_DIR/png_$MODEL_NAME" "*.png")
    echo "  PNG plots: $png_count files"
    
    gif_count=$(count_files "$REGION_DIR/0/animations" "*.gif")
    echo "  GIF animations: $gif_count files"
    
    # Check for artifacts
    if [ -d "$REGION_DIR/artifacts_$MODEL_NAME" ]; then
        artifact_count=$(count_files "$REGION_DIR/artifacts_$MODEL_NAME" "*.nc")
        echo "  NetCDF artifacts: $artifact_count files"
    fi
else
    echo "✗ Region directory not found (verification not run)"
fi
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

# Determine workflow progress
steps_complete=0
total_steps=6

[ -f "$MODEL_DIR/init_field.grib" ] && steps_complete=$((steps_complete + 1))
[ -d "$MODEL_DIR/ifs_ens.zarr" ] && steps_complete=$((steps_complete + 1))
[ -f "$MODEL_DIR/fields.txt" ] && steps_complete=$((steps_complete + 1))
[ -d "$PERTURBATION_DIR" ] && steps_complete=$((steps_complete + 1))
[ -d "$PERTURBATION_DIR/forecast.zarr" ] && steps_complete=$((steps_complete + 1))
[ -d "$REGION_DIR" ] && steps_complete=$((steps_complete + 1))

echo "Workflow progress: $steps_complete / $total_steps steps complete"
echo ""

if [ $steps_complete -eq 0 ]; then
    echo "Status: Not started"
    echo ""
    echo "Next step: Download initial conditions"
    echo "  ai-ens download-reanalysis"
elif [ $steps_complete -lt 4 ]; then
    echo "Status: Data preparation in progress"
    echo ""
    echo "Next step: Complete downloads and run inference"
    echo "  ai-ens infer --member 0"
elif [ $steps_complete -lt 6 ]; then
    echo "Status: Inference complete or in progress"
    echo ""
    echo "Next step: Run verification"
    echo "  ai-ens verify"
else
    echo "Status: Complete! ✓"
    echo ""
    echo "View results in:"
    echo "  Plots: $REGION_DIR/png_$MODEL_NAME/"
    echo "  Animations: $REGION_DIR/0/animations/"
fi
echo ""
