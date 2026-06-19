#!/usr/bin/env bash
# Multi-lead model-soups uplift data (tab:uplift restructure, 2026-06-19).
#   For the 3 production picks (aurora_encoder s=0.025, graphcast_all s=0.01,
#   sfno_modes10 s=0.25) on the 4-init ablation grid, compute at leads 24/120/240 h:
#     - dRMSE of a single perturbed member vs unperturbed control (mean over 10 members)
#     - dRMSE of the ensemble mean vs control
#   (CRPSS at the same leads already exists in figures/calibration_basis_table.tex.)
# CPU-only (no GPU). Heavy zarr I/O. Per-(model,lead) CSVs land in scratch; a
# follow-up assembles the LaTeX table.
#
# Usage: sbatch tools/submit_uplift_multilead.sh
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=uplift_ml
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/uplift_multilead_%j.log

set -uo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

OUT=/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/uplift_multilead
mkdir -p "$OUT"

for L in 24 120 240; do
    echo "=== lead ${L} h ==="
    $PY -u tools/compute_member_vs_mean_rmse.py \
        --lead "$L" \
        --out "$OUT/uplift_L${L}.csv" \
        > "$OUT/uplift_L${L}.log" 2>&1
    echo "  -> $OUT/uplift_L${L}.csv (exit $?)"
done

echo "DONE. CSVs in $OUT"
