#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_inf_ldm
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

VAE_PATH="$SCRIPT_DIR/../checkpoints/vae/TO_FILL/vae_best_epoch"
VAE_MASK_PATH="$SCRIPT_DIR/../checkpoints/vaeMasks/vaeMasks_nodule+lung_latent1_TO_FILL/vae_best_epoch/"

MASK_DATASET="TO_FILL/Datasets/NLST_LAND"
SAVE_FOLDER="$SCRIPT_DIR/../outputs/ldm/inference_ldm/"
LATENTS_DIR="" # if you want to use precomputed vae lantents to use the same across all experiments specify the path to the latents here
BATCH_SIZE=1

# Format: "MODEL_PATH NUM_SAMPLES_TO_GENERATE STEPS"
declare -a configs=(
    "$SCRIPT_DIR/../checkpoints/ldm/TO_FILL_nodule+lung_mask 3 1000" #example to generate 3 samples with 1000 steps from a specific model checkpoint
    # Add more lines here, for other experiments
)

if [[ -n "$LATENTS_DIR" ]]; then
    latents_arg="--latents_dir $LATENTS_DIR"
else
    latents_arg=""
fi

# Function to infer mask_mode and mask_dataset from model path
get_mask_config() {
    local model_path="$1"
    local mask_mode=""

    if [[ "$model_path" == *"nodule+lung+texture"* ]]; then
        mask_mode="nodule+lung+texture"
    elif [[ "$model_path" == *"nodule+lung"* ]]; then
        mask_mode="nodule+lung"    
    elif [[ "$model_path" == *"nodule"* ]]; then
        mask_mode="nodule"
    else
        mask_mode="none"
    fi

    echo "$mask_mode"
}

for config in "${configs[@]}"; do
    read -r model_path num_samples steps <<< "$config"

    model_name=$(basename "$model_path")
    save_dir="${SAVE_FOLDER}/${model_name}_${steps}steps"

    # Determine mask settings
    mask_mode=$(get_mask_config "$model_path")

    # Build mask args
    if [[ "$mask_mode" != "none" ]]; then
        mask_args="--mask_mode $mask_mode --mask_dataset $MASK_DATASET"
    else
      mask_args="--mask_mode $mask_mode"
    fi

    echo -e "\n[INFO] Model: $model_path"
    echo "[INFO] Save to: $save_dir"
    echo "[INFO] Mask mode: $mask_mode"
    echo "[INFO] Mask dataset: $MASK_DATASET"

    python -B "$SCRIPT_DIR/../src/inference_ldm.py" \
        --vae_path "$VAE_PATH" \
        --model_path "$model_path" \
        --save_dir "$save_dir" \
        --num_samples "$num_samples" \
        --steps "$steps" \
        --batch_size "$BATCH_SIZE" \
        --vae_mask_dir "$VAE_MASK_PATH" \
        $mask_args \
        $latents_arg      
done