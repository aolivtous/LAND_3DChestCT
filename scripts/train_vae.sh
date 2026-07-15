#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_train_vae
#SBATCH -o jobs_train/job_%j.o         
#SBATCH -e jobs_train/job_%j.e
#SBATCH --partition=TO_FILL
#SBATCH --nodes=TO_FILL
#SBATCH --ntasks=TO_FILL
#SBATCH --cpus-per-task=TO_FILL
#SBATCH --mem=TO_FILL
#SBATCH --time=TO_FILL
#SBATCH --gres=gpu:TO_FILL

# Get the directory where this .sh script is located
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
MODEL_CONFIG="$SCRIPT_DIR/../src/vae/configs/config_vae.json"
TRAIN_CONFIG="$SCRIPT_DIR/../src/vae/configs/config_vae_train.json"
DATASET_PATH="TO_FILL/Datasets/LIDC_LAND"

LOG_PATH="$SCRIPT_DIR/../logs/vae/logging/"
TENSORBOARD_LOG_DIR="$SCRIPT_DIR/../logs/vae/tensorboard/"
MODEL_DIR="$SCRIPT_DIR/../checkpoints/vae/"

TRAIN_PORTION=0.9

RUN_NAME="vaeLAND" # Name for this training run (used in logging)
export WANDB_MODE=disabled

# ==== Run the script ====
python -B "$SCRIPT_DIR/../src/vae/vae_train.py" \
  --tensorboard_log_path "$TENSORBOARD_LOG_DIR" \
  --model_dir "$MODEL_DIR" \
  --model_config_file "$MODEL_CONFIG" \
  --train_config_file "$TRAIN_CONFIG" \
  --dataset_path "$DATASET_PATH" \
  --train_portion "$TRAIN_PORTION" \
  --log_path "$LOG_PATH" \
  --run_name "$RUN_NAME" \
#   --enable_wandb  #comment it to not use it