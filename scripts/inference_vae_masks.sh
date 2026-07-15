#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_inf_vae_masks
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
MODEL_DIR="$SCRIPT_DIR/../checkpoints/vaeMasks/vaeMasks_nodule+lung_latent1_TO_FILL/vae_best_epoch"
DATASET_PATH="TO_FILL/Datasets/LIDC_LAND"

OUTPUT_PATH="$SCRIPT_DIR/../outputs/vae/inference_masks/"
OUTPUT_CSV="$SCRIPT_DIR/../outputs/vae/metrics_masks_inference.csv"
TRAIN_PORTION=0.9

get_mask_config() {
    local model_path="$1"
    local MASK_MODE=""
    local NUM_CLASSES=0
    
    if [[ "$model_path" == *"nodule+lung+texture"* ]]; then
        MASK_MODE="nodule+lung+texture"
        NUM_CLASSES=7
    elif [[ "$model_path" == *"nodule+lung"* ]]; then
        MASK_MODE="nodule+lung"
        NUM_CLASSES=3
    elif [[ "$model_path" == *"nodule"* ]]; then
        MASK_MODE="nodule"
        NUM_CLASSES=2
    else
        MASK_MODE="none"
        NUM_CLASSES=1
    fi

    echo "$MASK_MODE $NUM_CLASSES"
}
read MASK_MODE NUM_CLASSES <<< $(get_mask_config "$MODEL_DIR")

# ==== Run the script ====
python -B "$SCRIPT_DIR/../src/vae/vae_masks_inference.py" \
  --model_dir "$MODEL_DIR" \
  --dataset_name "$DATASET_PATH" \
  --train_portion "$TRAIN_PORTION" \
  --output_path "$OUTPUT_PATH" \
  --output_csv "$OUTPUT_CSV" \
  --num_classes "$NUM_CLASSES" \
  --mask_mode "$MASK_MODE" \
