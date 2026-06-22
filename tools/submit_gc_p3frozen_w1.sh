#!/usr/bin/env bash
# wd_kde (W1) for the FROZEN GraphCast Phase-3 winner (gcsigma_1.0_gcnodes42_frozen).
# Its eval produced the ridgeline npz but not the wasserstein-summary CSV, so the
# calibration-table W1 cell was missing. wd_kde-only config, host venv (CPU), 4 inits.
#SBATCH --job-name=gcp3frz_w1
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=320G
#SBATCH --time=03:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/gcp3frz_w1_%j.log
set -euo pipefail
STORE="/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SRC_DIR="/users/sadamov/pyprojects/ai-models-ensembles"
export PYTHONUNBUFFERED=1
export TMPDIR="/iopsstor/scratch/cscs/sadamov/tmp"
export DASK_TEMPORARY_DIRECTORY="/iopsstor/scratch/cscs/sadamov/tmp"
mkdir -p "$TMPDIR"
source "${SRC_DIR}/.venv/bin/activate"
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
cd "$SRC_DIR"
cfg="$STORE/ablation/phase3/graphcast_operational/eval/gcsigma_1.0_gcnodes42_frozen_wdkde_w1_config.yaml"
echo "===== [$(date -u +%H:%M:%S)] wd_kde (W1) gc P3 frozen -> ${cfg} ====="
python -m swissclim_evaluations.cli --config "$cfg"
echo "===== [$(date -u +%H:%M:%S)] done ====="
ls -la "$STORE/ablation/phase3/graphcast_operational/eval/gcsigma_1.0_gcnodes42_frozen/wd_kde/wd_kde_wasserstein_averaged_enspooled.csv"
