#!/usr/bin/env bash
# Fill the 150 + 600 hPa log-pressure levels on the 28 early-2023 IC inits
# (init_time indices 0:56 = Jan 2-8 + Apr 2-8 2023 clusters, prod inits + their
# T-6h companions) of the IFS-ENS perturbed-IC zarr. Those two PL levels are
# never archived by MARS for type=pf step=0 and were left NaN on these inits
# (the post-download log-p pass never ran on them), which made every _ic rollout
# on these inits produce all-NaN output past lead 0.
#
# CPU sbatch (not login node): the per-init-sharded v3 store does a full-shard
# read-modify-write per (var, init), ~1.8 TB of I/O over the 56 inits -- enough
# to OOM/exit-144 a login node. Idempotent: re-running skips already-finite
# target cells, so a canary fill of index 0 is harmless to repeat here.
#
# Usage: sbatch tools/submit_ic_fill.sh
#
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ic_fill
#SBATCH --output=/iopsstor/scratch/cscs/sadamov/ic_fill_%j.log

set -uo pipefail

PY=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/iopsstor/scratch/cscs/sadamov/tmp
mkdir -p "$TMPDIR"

cd /users/sadamov/pyprojects/ai-models-ensembles

INDICES="${1:-0:56}"
echo "Filling 150/600 hPa on IC zarr init_time indices: $INDICES"
$PY -u tools/fill_ic_perturbed_levels.py --indices "$INDICES"
echo "FILL DONE"
