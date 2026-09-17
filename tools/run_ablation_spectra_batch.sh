#!/usr/bin/env bash
# Debug-partition batch runner for the Tab. 4 energy_spectra re-eval: runs the
# spectra-only configs written by tools/submit_ablation_spectra_reeval.sh for
# every target run that still lacks its LSD CSV, up to $PAR at a time on one
# node, inside the 1.5 h debug limit. Idempotent: resubmit until nothing is left.
#
# Usage:  sbatch --partition=debug --account=ab016 --time=01:30:00 --nodes=1 \
#           --cpus-per-task=144 --mem=800G tools/run_ablation_spectra_batch.sh
set -uo pipefail
STORE="${AIENS_STORE:-/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles}"
# sbatch runs a spooled copy of this script, so the repo path cannot be derived from it
SRC_DIR="${AIENS_SRC:-/users/sadamov/pyprojects/ai-models-ensembles}"
PAR="${PAR:-4}"
source "$SRC_DIR/.venv/bin/activate"

RUNS=(
  phase1/aurora/mag_0.03_layer_all
  phase2/aurora/mag_0.044176_layer_encoder
  phase2b/aurora/mag_0.025_layer_encoder
  phase3/aurora/mag_0.40_layer_unet_bottom
  phase3b/aurora/mag_0.015_layer_enc_012
  phase1/graphcast_operational/mag_0.01_layer_all
  phase2/graphcast_operational/mag_0.029665_layer_m2g
  phase2b/graphcast_operational/mag_0.014_layer_g2m
  phase3/graphcast_operational/gcsigma_1.0_gcnodes42_frozen
  phase3b/graphcast_operational/gcsigma_0.159_gcnodes162_frozen
  phase1/sfno/mag_0.03_layer_all
  phase2/sfno/mag_0.053852_layer_encoder
  phase2b/sfno/mag_0.035_layer_encoder
  phase3/sfno/mag_0.25_modes10
  phase3b/sfno/mag_0.035_modes20
  phase1/aifs/mag_0.01_layer_all
  phase2/aifs/mag_0.027500_layer_decoder
)

todo=()
for r in "${RUNS[@]}"; do
  phase="${r%%/*}"; rest="${r#*/}"; model="${rest%%/*}"; run="${rest#*/}"
  d="$STORE/ablation/$phase/$model/eval/$run"
  if ls "$d"/energy_spectra/energy_ratios_3d_lead_time_*_enspooled.csv >/dev/null 2>&1; then
    echo "have  $r"; continue
  fi
  rm -rf "$d/energy_spectra"   # partial output of a preempted attempt
  todo+=("$(dirname "$d")/${run}_energy_spectra_only_config.yaml")   # eval-level config, not the snapshot SwissClim drops into the run dir
done
echo "todo: ${#todo[@]}"
[[ ${#todo[@]} -eq 0 ]] && exit 0

printf '%s\n' "${todo[@]}" | xargs -P "$PAR" -I{} bash -c \
  'echo "START $(date +%T) {}"; python -m swissclim_evaluations.cli --config "{}" > "{}.log" 2>&1; echo "END   $(date +%T) rc=$? {}"'
echo "batch finished $(date +%T)"
