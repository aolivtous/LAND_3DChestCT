#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_data_proc
#SBATCH -o jobs_preproc/job_%j.o         
#SBATCH -e jobs_preproc/job_%j.e
#SBATCH --partition=TO_FILL
#SBATCH --nodes=TO_FILL
#SBATCH --ntasks=TO_FILL
#SBATCH --cpus-per-task=TO_FILL
#SBATCH --mem=TO_FILL
#SBATCH --time=TO_FILL
#SBATCH --gres=gpu:TO_FILL      

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_MODULE='conda'
CONDA_ENV_PATH="TO_FILL/.conda/envs/land"

#------------------ Environment ------------------#
module load "${CONDA_MODULE}"
source activate "${CONDA_ENV_PATH}"

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export PYTHONPATH="$SCRIPT_DIR/../src"

DICOM_DIR=TO_FILL
NPY_DIR=TO_FILL/Datasets/LIDC_LAND
PNG_DIR=$NPY_DIR/image_seq

# Ensure ~/.pylidcrc points to the DICOM root for pylidc
echo -e "[dicom]\npath = $DICOM_DIR" > ~/.pylidcrc
echo "Set .pylidcrc to use DICOM path: $DICOM_DIR"

# Run full preprocessing
cd "$SCRIPT_DIR/../src"
python3 -m utils.preproc_lidc_npy --dicom_dir "$DICOM_DIR" --npy_dir "$NPY_DIR" --normalize --resample --central_crop