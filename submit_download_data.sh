#!/usr/bin/bash -l
#SBATCH --job-name=ai_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=pp-long
#SBATCH --account=s83
#SBATCH --output=logs/out_dl_%j.log
#SBATCH --error=logs/err_dl_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue

source ./config.sh

srun bash -c '
echo "Downloading (Re)Analysis and IFS for $MODEL_NAME and $DATE_TIME with 51 members"
echo "This will generate ~350 GB of data"

create_dir_if_not_exists "$OUTPUT_DIR/source_files"
cp -r "$SRC_DIR"/* "$OUTPUT_DIR/source_files"

create_dir_if_not_exists "$DATE_DIR"
create_dir_if_not_exists "$MODEL_DIR"

proceed_if_not_exists "${MODEL_DIR}/fields.txt" "ai-models --fields $MODEL_NAME > ${MODEL_DIR}/fields.txt"

proceed_if_not_exists "${MODEL_DIR}/init_field.grib" "python -m \
    ai_models_ensembles.download_re_analysis $OUTPUT_DIR $DATE_TIME $END_DATE_TIME $INTERVAL $MODEL_NAME"
proceed_if_not_exists "${MODEL_DIR}/ifs_ens.zarr/.zmetadata" "python -m \
    ai_models_ensembles.download_ifs_ensemble $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME"
proceed_if_not_exists "${MODEL_DIR}/ifs_control.zarr/.zmetadata" "python -m \
    ai_models_ensembles.download_ifs_control $OUTPUT_DIR $DATE_TIME $INTERVAL $NUM_DAYS $MODEL_NAME"
echo "*****DONE*****"
'
