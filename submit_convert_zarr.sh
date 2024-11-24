#!/usr/bin/bash -l
#SBATCH --job-name=ai_zarr
#SBATCH --nodes=3
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --output=logs/out_zarr_%j.log
#SBATCH --error=logs/err_zarr_%j.log
#SBATCH --time=24:00:00
#SBATCH --no-requeue

source ./config.sh

export job1='echo "Converting all grib files to zarr files for $MODEL_NAME and $DATE_TIME"
echo "Converting the unperturbed forecast."
python -u -m ai_models_ensembles.convert_grib_to_zarr "$MODEL_DIR"'

export job2='echo "Converting the perturbed forecasts."
python -u -m ai_models_ensembles.convert_grib_to_zarr "$PERTURBATION_DIR" --subdir_search True
echo "*****DONE*****"
'

# Convert the unperturbed forecast
srun -N1 -n1 -c32 --mem 64G\
     --output=logs/out_zarr0_%j.log \
     --error=logs/err_zarr0_%j.log \
     bash -c "$job1"

wait

# Run with different perturbation values
for latent in 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.008 0.006 0.004 0.002; do
    export PERTURBATION_LATENT=$latent
    export PERTURBATION_DIR="${MODEL_DIR}/init_${PERTURBATION_INIT}_latent_${PERTURBATION_LATENT}_layer_${LAYER}"
    export job2
    srun -N1 -n1 -c32 --mem 64G\
         --output=logs/out_zarr${latent}_%j.log \
         --error=logs/err_zarr${latent}_%j.log \
         bash -c "$job2" &
done
wait
