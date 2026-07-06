#!/usr/bin/env bash
# Rank (Talagrand) histograms at 240 h for the appendix C4 figure.
# CPU sbatch: reading the 112-init production zarrs on a login node gets
# resource-killed (see per_pixel_ssr / spatial-mean SSR history).
#
# Usage:
#   sbatch tools/submit_rank_histograms.sh                 # ifs_ens only (default)
#   sbatch tools/submit_rank_histograms.sh all             # all 8 baselines
#   sbatch tools/submit_rank_histograms.sh ifs_ens aifsens # a subset
#
# Output: /iopsstor/scratch/cscs/sadamov/rank_hist_<baseline>.npz
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=rank_hist
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/rank_hist_%j.log

set -euo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

BASELINES=("${@:-ifs_ens}")
$PY -u tools/compute_rank_histograms.py --baselines "${BASELINES[@]}"
