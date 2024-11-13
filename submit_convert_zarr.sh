#!/usr/bin/bash -l
#SBATCH --job-name=ai_zarr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --output=logs/out_zarr_%j.log
#SBATCH --error=logs/err_zarr_%j.log
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --mem=200G

source ./config.sh

srun bash -c '
echo "Converting all grib files to zarr files for $MODEL_NAME and $DATE_TIME with $NUM_MEMBERS members"
python -u -m ai_models_ensembles.convert_grib_to_zarr "$PERTURBATION_DIR" --subdir_search True
echo "*****DONE*****"
'
