# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is based on code from the MONAI MAISI Project (https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi) 

import argparse
import glob
import os
import torch
import warnings
import cv2
import numpy as np
import imageio
import pandas as pd

from pathlib import Path
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm.auto import tqdm
from utils.transforms import VAE_Transform
from utils.losses import KL_loss
from autoencoder_kl import AutoencoderKlReducedMaisi 
from monai.metrics.regression import MultiScaleSSIMMetric, PSNRMetric
warnings.filterwarnings("ignore")

# Print Monai configuration
print_config()

closest_even_nb = lambda n:  n if n % 2 == 0 else n - 1

    
def get_heatmap(original_video, reconstructed_video):
    min = 0
    max = 0
    frames_with_heatmap = []
    print(f"original_video.shape: {original_video.shape}")
    for i in range(original_video.shape[0]):  # Iterate over each frame
        
        real_frame = original_video[i]
        recon_frame = reconstructed_video[i]

        # Compute the squared difference between the original and reconstructed frame
        diff_frame = np.sqrt((recon_frame - real_frame) ** 2)

        if i == 0:
            min = diff_frame.min()
            max = diff_frame.max()
        
        else:
            if diff_frame.min() < min:
                min = diff_frame.min()
                print("act min")
            if diff_frame.max() > max:
                max = diff_frame.max()
                print("act max")


        #convert to float 32
        diff_frame = diff_frame.astype(np.float32)

        #normalize the difference
        #diff_frame_normalized = cv2.normalize(diff_frame, None, 0, 1, cv2.NORM_MINMAX)
       
        # Ensure the difference is valid for visualization
        if np.any(np.isnan(diff_frame)) or np.any(np.isinf(diff_frame)):
            print(f"Invalid frame {i} due to invalid difference.")
            continue
    
        # Apply a single colormap for the absolute difference
        heatmap = cv2.applyColorMap((diff_frame * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap = np.float32(heatmap_rgb)

        # Append the blended frame to the list of frames
        frames_with_heatmap.append(heatmap)

    # Convert frames to uint8
    frames_with_heatmap = np.array(frames_with_heatmap, dtype=np.uint8)

    print(f"diff_frame min {min} and max {max}")

    return frames_with_heatmap
   
def prepare_data(args):
    """Prepare the dataset and dataloaders."""

    # Search for the CT volumes within the preprocessed .npy dataset layout:
    # <dataset_path>/<series_id>/chest_ct/<series_id>.npy
    train_images_1 = sorted(glob.glob(os.path.join(args.dataset_path, "**", "chest_ct", "*.npy"), recursive=True))

    data_dicts_1 = [{"image": image_name} for image_name in train_images_1]
    len_train = int(args.train_portion * len(data_dicts_1))
    train_files_1, val_files_1 = data_dicts_1[:len_train], data_dicts_1[len_train:]

    print(f"Dataset LIDC: number of val data is {len(val_files_1)}.")

    # Dataset dictionary for expandable datasets
    datasets = {
        1: {
            "data_name": "Dataset LIDC",
            "train_files": train_files_1,
            "val_files": val_files_1,
            "modality": "ct",
        }
    }

    # Initialize file lists for modalities
    train_files = {"ct": [], "mri":[]}
    val_files = {"ct": [], "mri":[]}

    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    # Process datasets and append modality labels
    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        val_files_i = dataset["val_files"]
    
        # Attach modality to each file
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
        val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

    
    # Combine training and validation data

    test_files_combined = val_files["ct"] + val_files["mri"]

    # Define transformations for training and validation datasets
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=None,  # if None, will validate on whole image volume
        output_dtype=torch.float16,  # final data type
        spacing_type="original",
        spacing=[1.0, 1.0, 1.0],
        val_crop_size=[256, 256, 256],
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    # Create DataLoader for validation

    dataset = CacheDataset(data=test_files_combined, transform=val_transform, cache_rate=0, num_workers=8)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    return dataloader

def setup_device():
    """Setup device for training."""
    return torch.device("cuda")

def main(args):

    #Set random seed for reproducibility.
    set_determinism(seed=0)
    dataloader_val = prepare_data(args)
    device = setup_device()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    autoencoder = AutoencoderKlReducedMaisi.from_pretrained(args.model_dir)
    num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    # config loss and loss weight
    
    intensity_loss = L1Loss(reduction="mean")

    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    )

    torch.cuda.reset_peak_memory_stats()
    
    val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
    
    i=0
   
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    psnr = PSNRMetric(max_val=1.0)

    ms_ssims = []
    psnrs = []

    for batch in tqdm(dataloader_val, unit="batch", leave=False):

        all_vid_tensors = []

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        autoencoder.to(device)
        autoencoder.eval()
        #select a random bath from the validation set
        with torch.no_grad():
            with autocast("cuda", enabled=True):
                images = batch["image"]
                #reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images) # sliding window inferer with patch size = image size was giving error
                reconstruction,z_mu, z_sigma= autoencoder(images.to(device))
                reconstruction = reconstruction.to(device)
                val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item() 
            
                val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()
                peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
                print(f"Max memory allocated in validation is {peak_memory_gb}")

        #calculate the heatmap for each view [num_frames, height, width, channels]).
        reconstruction = reconstruction.to(device)
        images = images.to(device)
        value = psnr(images, reconstruction).item()
        psnrs.append(value)
        
        score = ms_ssim(images, reconstruction).item()
        ms_ssims.append(score)

        i+=1
        
    
    for key in val_epoch_losses:
        val_epoch_losses[key] /= len(dataloader_val)

    for loss_name, loss_value in val_epoch_losses.items():
        print(f"Val_vae_loss: {loss_name} = {loss_value}")

    #save the losses and the metrics to a csv file
    results = {
        "model_dir": args.model_dir,
        "recons_loss": round(val_epoch_losses["recons_loss"], 3),
        "kl_loss": round(val_epoch_losses["kl_loss"], 3),
        "p_loss": round(val_epoch_losses["p_loss"], 3),
        "ms_ssim": round(np.mean(ms_ssims), 3),
        "psnr": round(np.mean(psnrs), 3),
    }

    if args.output_csv:
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(args.output_csv, mode='a', index=False)
        print(f"Results saved to {args.output_csv}")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate outputs from trained VAE model using specified dataset and configuration.")

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the directory containing the trained VAE model.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the preprocessed .npy dataset directory (as produced by '
                             'preproc_lidc_npy.py / preproc_nlst_npy.py), containing <series_id>/chest_ct/*.npy.')
    parser.add_argument('--train_portion', type=float, required=True,
                        help='Portion of the dataset to use (e.g., 0.9 means use 90%% of the dataset).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Directory to save the generated output videos or results.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save the output CSV file with results (optional).')

    args = parser.parse_args()

    main(args)