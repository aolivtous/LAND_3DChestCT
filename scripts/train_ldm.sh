#!/bin/bash
#------------------------------------------------------------------
#SBATCH -J land_train_ldm
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

log_dir="$SCRIPT_DIR/../logs/ldm/"
model_dir="$SCRIPT_DIR/../checkpoints/ldm/"
vae_dir="$SCRIPT_DIR/../checkpoints/vae/TO_FILL/vae_best_epoch"
vae_mask_dir="$SCRIPT_DIR/../checkpoints/vaeMasks/vaeMasks_nodule+lung_latent1_TO_FILL/vae_best_epoch"
image_dataset="TO_FILL/Datasets/LIDC_LAND"
mask_dataset="TO_FILL/Datasets/LIDC_LAND"
# training configuration
num_epochs=500
num_inference_steps=1000
save_freq=10 # in epochs
save_freq_imgs=30 # in epochs
resolution=256
lr=1e-5
bsz=1
prediction_type="v_prediction"

########## to resume from a specific checkpoint set the path here ##########
ckpt_path=""

# experiment list (uncomment) # Format: "experiment_name:extra_args"
declare -a EXPERIMENT_CONFIGS=(
  #"unconditional:--mask_mode none --attention" # --resume_from_checkpoint $ckpt_path"
  #"nodule_mask:--mask_mode nodule --mask_dataset $mask_dataset --attention"
  "nodule+lung_mask:--mask_mode nodule+lung --mask_dataset $mask_dataset --attention"
  #"nodule+lung+texture_mask:--mask_mode nodule+lung+texture --mask_dataset $mask_dataset --attention" #--resume_from_checkpoint $ckpt_path"
)

for config in "${EXPERIMENT_CONFIGS[@]}"; do
  IFS=":" read -r suffix extra_args <<< "$config"

  # Prevent accidental leading colons
  suffix="${suffix//:/}"
  extra_args="${extra_args:-}"

  echo "Running $suffix experiment..."

  exp_id="${resolution}_bsz${bsz}_lr${lr}_$suffix"

  python -B "$SCRIPT_DIR/../src/train_ldm.py" \
    --dataset_name="$image_dataset" \
    --resolution="$resolution" \
    --output_dir="$model_dir/$exp_id" \
    --logging_dir="$log_dir/$exp_id" \
    --train_batch_size="$bsz" \
    --num_epochs="$num_epochs" \
    --gradient_accumulation_steps=1 \
    --learning_rate="$lr" \
    --lr_warmup_steps=10 \
    --checkpointing_steps=1000 \
    --save_images_epochs="$save_freq_imgs" \
    --save_model_epochs="$save_freq" \
    --ddpm_num_inference_steps="$num_inference_steps" \
    --mixed_precision=no \
    --dataloader_num_workers=8 \
    --prediction_type="$prediction_type" \
    --snr_gamma=5.0 \
    --vae_dir="$vae_dir" \
    --vae_mask_dir="$vae_mask_dir" \
    $extra_args
done



