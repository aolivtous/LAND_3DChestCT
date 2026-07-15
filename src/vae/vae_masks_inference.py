
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
import numpy as np
import imageio
import warnings
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from monai.config import print_config
from autoencoder_kl import AutoencoderKlReducedMaisi
from utils.masks_utils import LIDCMasks, vae_loss_sdf, vae_loss_segmentation, per_class_dice

warnings.filterwarnings("ignore")
print_config()

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def prepare_val_data(args):
    """Prepare LIDC masks validation dataloader."""
    val_dataset =  LIDCMasks(
        directory=args.dataset_name,
        mask_mode=args.mask_mode,
        num_classes=args.num_classes,
        split="val",
        val_ratio=1-args.train_portion,
        seed=args.seed if hasattr(args, "seed") else 42,
        use_onehot=True,  # Changed to False to use single-channel input to use less memory
        sdf_flag=args.sdf_flag,
        original_textures=args.original_textures,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    return val_dataloader


def main(args):
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset CUDA memory tracking if using GPU
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Load model
    autoencoder = AutoencoderKlReducedMaisi.from_pretrained(args.model_dir).to(device)
    autoencoder.eval()
    print(f"Loaded model from {args.model_dir}")

    # Prepare data
    dataloader_val = prepare_val_data(args)

    val_losses = []
    dice_per_class = [[] for _ in range(args.num_classes)]
    i = 0

    for batch in tqdm(dataloader_val, desc="Inference"):
        if args.sdf_flag:
            target_sdf = batch["mask_sdf"].to(device)  # [B,1,D,H,W]
        else:
            mask_input = batch["mask_input"].to(device)  # [B,C,D,H,W]
        
        masks_index = batch["mask_index"].to(device)    # [B,D,H,W]

        with autocast("cuda", enabled=True):
            with torch.no_grad():
                if args.sdf_flag:
                    recon, mu, sigma = autoencoder(target_sdf)
                    loss_g, sdf_l1, eik_loss, kl = vae_loss_sdf(recon, target_sdf, mu, sigma, beta=1e-4)
                    print("loss_g:" , loss_g.item(),
                    "SDF L1 loss: ", sdf_l1.item(), 
                    "Eikonal loss: ", eik_loss.item(),
                    " KL loss: ", kl.item()
                    )
                    val_losses.append(loss_g.item())
                    #print consumed memory
                    if device.type == "cuda":
                        current_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        print(f"Current GPU memory used: {current_mem:.2f} GB")
                    min_vals, min_idx = torch.min(recon, dim=1)   # [1, D, H, W]
                    pred_mask = min_idx + 1                              # [1, D, H, W]
                    pred_mask[min_vals > 0] = 0                          # background where both distances are positive
                    #print(f"pred_classes shape: {pred_mask.shape}, unique values: {torch.unique(pred_mask)}")
                    dices = per_class_dice(pred_mask, masks_index, args.num_classes, is_logits=False)

                else:
                    recon, mu, logvar = autoencoder(mask_input)
                    loss_val,  recon_loss_val, kl_val, ce_val, dice_val  = vae_loss_segmentation(recon, masks_index, mu, logvar, args.num_classes, args.mask_mode, beta=1e-4)
                    print("Reconstruction loss: ", recon_loss_val.item(),
                          " KL loss: ", kl_val.item(), 
                          " CE loss: ", ce_val.item(), 
                          " Dice loss: ", dice_val.item()
                          )
                    if device.type == "cuda":
                        current_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        print(f"Current GPU memory used: {current_mem:.2f} GB")
                    val_losses.append(loss_val.item())
                    dices = per_class_dice(recon, masks_index, args.num_classes)

                # Append each class dice to its own list
                for c in range(args.num_classes):
                    dice_per_class[c].append(dices[c].item())
                
        i += 1
  
    avg_val_loss = np.mean(val_losses)
    avg_dice_per_class = [float(np.nanmean(dice_per_class[c])) for c in range(args.num_classes)]
    print(f"Average validation VAE loss: {avg_val_loss:.4f}")
    print("Average Dice per class:")
    for c, dice in enumerate(avg_dice_per_class):
        print(f"  Class {c}: Dice = {dice:.4f}")


    # Print peak memory usage
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Peak GPU memory used: {peak_mem:.2f} GB")

    else:
        peak_mem = 0

    # Save to CSV
    if args.output_csv:
        results = {
            "model_dir": args.model_dir,
            "average_val_loss": round(avg_val_loss, 4),
            "dice_per_class": [round(d, 4) for d in avg_dice_per_class],
            "peak_memory_GB": round(peak_mem, 4),
            "num_classes": args.num_classes,
            "mask_mode": args.mask_mode,

        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(args.output_csv, mode="a", index=False)
        print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for VAE masks model.")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained VAE model directory.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Path to the dataset directory.")
    parser.add_argument("--train_portion", type=float, required=True,
                        help="Portion of dataset used for training (e.g., 0.9).")
    parser.add_argument("--mask_mode", type=str, default="nodule+lung",
                        choices=["nodule","lung", "nodule+lung", "nodule+lung+texture"])
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of segmentation classes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save output videos and metrics.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save results as CSV.")
    parser.add_argument('--sdf_flag', action='store_true', help="Use SDF regression loss instead of segmentation loss")
    parser.add_argument('--original_textures', action='store_true', help="Use original textures instead of synthesized ones")

    args = parser.parse_args()
    main(args)
