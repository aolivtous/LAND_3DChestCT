#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_inf_vae
#SBATCH -o jobs_inference/job_%j.o         
#SBATCH -e jobs_inference/job_%j.e
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

# ==== Configuration Variables ====
MODEL_DIR="$SCRIPT_DIR/../checkpoints/vae/TO_FILL/vae_best_epoch"
DATASET_PATH="TO_FILL/Datasets/LIDC_LAND"

OUTPUT_PATH="$SCRIPT_DIR/../outputs/vae/inference_micro/"
OUTPUT_CSV="$SCRIPT_DIR/../outputs/vae/metrics_inference.csv"
TRAIN_PORTION=0.9

# ==== Run the script ====
python -B "$SCRIPT_DIR/../src/vae/vae_inference.py" \
  --model_dir "$MODEL_DIR" \
  --dataset_path "$DATASET_PATH" \
  --train_portion "$TRAIN_PORTION" \
  --output_path "$OUTPUT_PATH" \
  --output_csv "$OUTPUT_CSV" \