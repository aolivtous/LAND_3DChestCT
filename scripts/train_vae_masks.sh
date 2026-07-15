#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_train_vae_masks
#SBATCH -o jobs_train/job_%j.o         
#SBATCH -e jobs_train/job_%j.e
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
MASK_MODE="nodule+lung"  # Options: "nodule", "nodule+lung", "nodule+lung+texture"
LATENT_SIZE="1"
MODEL_CONFIG="$SCRIPT_DIR/../src/vae/configs/config_vae_masks_${MASK_MODE}_latent${LATENT_SIZE}.json"
TRAIN_CONFIG="$SCRIPT_DIR/../src/vae/configs/config_vae_masks_train.json"
DATASET_PATH="TO_FILL/Datasets/LIDC_LAND"
LOG_PATH="$SCRIPT_DIR/../logs/vaeMasks/logging/"
TENSORBOARD_LOG_DIR="$SCRIPT_DIR/../logs/vaeMasks/tensorboard/"
MODEL_DIR="$SCRIPT_DIR/../checkpoints/vaeMasks/"

#set num_classes based on mask mode
if [ "$MASK_MODE" == "nodule" ]; then
  NUM_CLASSES=2
elif [ "$MASK_MODE" == "nodule+lung" ]; then
  NUM_CLASSES=3
elif [ "$MASK_MODE" == "nodule+lung+texture" ]; then
  NUM_CLASSES=7
else
  echo "Unknown MASK_MODE: $MASK_MODE"
  exit 1
fi

TRAIN_PORTION=0.9
RUN_NAME="vaeMasks_${MASK_MODE}_latent${LATENT_SIZE}" # Name for this training run (used in logging)

# ==== Run the script ====
python -B "$SCRIPT_DIR/../src/vae/vae_masks_train.py" \
  --tensorboard_log_path "$TENSORBOARD_LOG_DIR" \
  --model_dir "$MODEL_DIR" \
  --model_config_file "$MODEL_CONFIG" \
  --train_config_file "$TRAIN_CONFIG" \
  --dataset_path "$DATASET_PATH" \
  --train_portion "$TRAIN_PORTION" \
  --log_path "$LOG_PATH" \
  --run_name "$RUN_NAME" \
  --mask_mode "$MASK_MODE" \
  --num_classes "$NUM_CLASSES" \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.001 \
  #   --enable_wandb  #comment it to not use it


