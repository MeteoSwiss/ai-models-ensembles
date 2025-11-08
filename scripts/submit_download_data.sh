#!/usr/bin/bash -l
set -euo pipefail
IFS=$'\n\t'
#SBATCH --job-name=${DL_JOB_NAME}
#SBATCH --nodes=${DL_NODES}
#SBATCH --ntasks=${DL_NTASKS}
#SBATCH --cpus-per-task=${DL_CPUS_PER_TASK}
#SBATCH --mem-per-cpu=${DL_MEM_PER_CPU}
#SBATCH --partition=${DL_PARTITION}
#SBATCH --account=${DL_ACCOUNT}
#SBATCH --output=${LOG_DIR}/out_dl_%j.log
#SBATCH --error=${LOG_DIR}/err_dl_%j.log
#SBATCH --time=${DL_TIME}
#SBATCH --no-requeue

# Change to repository root
cd "$(dirname "$0")/.." || exit 1

source ./scripts/config.sh
bash ./tools/validate.sh

${DRY_RUN:+echo} srun bash -c '
echo "Downloading (Re)Analysis and IFS for $MODEL_NAME and $DATE_TIME with 51 members"
echo "This will generate ~350 GB of data"

create_dir_if_not_exists "$OUTPUT_DIR/source_files"
cp -r "$SRC_DIR"/* "$OUTPUT_DIR/source_files"

create_dir_if_not_exists "$DATE_DIR"
create_dir_if_not_exists "$MODEL_DIR"

proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields $MODEL_NAME > ${MODEL_DIR}/fields.txt"

proceed_if_not_exists "${MODEL_DIR}/init_field.grib" "python -u -m ai_models_ensembles.cli download-reanalysis --out-dir \"$OUTPUT_DIR\" --start \"$DATE_TIME\" --end \"$END_DATE_TIME\" --interval \"$INTERVAL\" --model \"$MODEL_NAME\""
proceed_if_not_exists "${MODEL_DIR}/ifs_ens.zarr/.zmetadata" "python -u -m ai_models_ensembles.cli download-ifs-ensemble --out-dir \"$OUTPUT_DIR\" --date-time \"$DATE_TIME\" --interval \"$INTERVAL\" --num-days \"$NUM_DAYS\" --model \"$MODEL_NAME\""
proceed_if_not_exists "${MODEL_DIR}/ifs_control.zarr/.zmetadata" "python -u -m ai_models_ensembles.cli download-ifs-control --out-dir \"$OUTPUT_DIR\" --date-time \"$DATE_TIME\" --interval \"$INTERVAL\" --num-days \"$NUM_DAYS\" --model \"$MODEL_NAME\""
echo "*****DONE*****"
'
