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
import json
import os
import time
import torch
import logging
import warnings
import wandb

from pathlib import Path
from monai.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils.transforms import VAE_Transform
from utils.losses import KL_loss
from autoencoder_kl import AutoencoderKlReducedMaisi

warnings.filterwarnings("ignore")

# Print Monai configuration
print_config()

def load_configurations(args,  timestamp):
    """Load configuration files and set up arguments."""

    # Load model path and setup directories

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    sub_folder_dir = os.path.join(args.model_dir, timestamp)
    Path(sub_folder_dir).mkdir(parents=True, exist_ok=True)
    trained_d_path = os.path.join(sub_folder_dir, "discriminator", "discriminator_last_epoch.pt")
    trained_d_path_best = os.path.join(sub_folder_dir, "discriminator", "discriminator_best_epoch.pt")
    Path(os.path.dirname(trained_d_path)).mkdir(parents=True, exist_ok=True)
    trained_g_path = os.path.join(sub_folder_dir, "vae_last_epoch")
    trained_g_path_best = os.path.join(sub_folder_dir, "vae_best_epoch")
    logging.info(f"Last Trained model will be saved as {trained_g_path} and {trained_d_path}.")
    logging.info(f"Best Trained model will be saved as {trained_g_path_best} and {trained_d_path_best}.")

    # Setup Tensorboard path with timestamp
    tensorboard_path = os.path.join(args.tensorboard_log_path, f"vae_{timestamp}")
    Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_path)
    logging.info(f"Tensorboard event will be saved as {tensorboard_path}.")

    layout = {
        "VAE Losses epoch": {
            "Reconstr_loss": ["Multiline", ["recons_loss/train", "recons_loss/val"]],
            "KL_loss": ["Multiline", ["kl_loss/train", "kl_loss/val"]],
            "Perceptual_loss": ["Multiline", ["p_loss/train", "p_loss/val"]],
        },
        "Discriminator Losses": {
            "Adv Loss": ["Multiline", ["train_adv_loss_iter", "train_fake_loss_iter", "train_real_loss_iter"]],
        },
        "Scale Factor": {
            "Scale Factor": ["Multiline", ["scale_factor/train", "scale_factor/val"]],
        },
        "VAE Losses Iter": {
            "Reconstr_loss": ["Multiline", ["train_recons_loss_iter"]],
            "KL_loss": ["Multiline", ["train_kl_loss_iter"]],
            "Perceptual_loss": ["Multiline", ["train_p_loss_iter"]],
        },
            "Scale Factor Iter": {
            "Scale Factor Iter": ["Multiline", ["scale_factor_train_iter"]],
        }
    }

    tensorboard_writer.add_custom_scalars(layout)

    # Load additional config files
    config_dict = json.load(open(args.model_config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    config_train_dict = json.load(open(args.train_config_file, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
        logging.info(f"{k}: {v}")
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
        logging.info(f"{k}: {v}")

    logging.info("Network definition and training hyperparameters have been loaded.")

    # Initialize WandB
    if args.enable_wandb:   
        wandb.init(project="Reduced Maisi VAE", config=args)  # Set your project name
        wandb.run.name = f"{args.run_name}_{timestamp}"
        logging.info(f"WandB logging initialized for {wandb.run.name}.")


    return trained_g_path, trained_d_path, trained_g_path_best, trained_d_path_best, tensorboard_writer

def prepare_data(args):
    """Prepare the dataset and dataloaders."""

    # Search for the CT volumes within the preprocessed .npy dataset layout:
    # <dataset_path>/<series_id>/chest_ct/<series_id>.npy
    train_images_1 = sorted(glob.glob(os.path.join(args.dataset_path, "**", "chest_ct", "*.npy"), recursive=True))

    print(len(train_images_1))
    data_dicts_1 = [{"image": image_name} for image_name in train_images_1]
    len_train = int(args.train_portion * len(data_dicts_1))
    train_files_1, val_files_1 = data_dicts_1[:len_train], data_dicts_1[len_train:]

    logging.info(f"Dataset LIDC: number of training data is {len(train_files_1)}.")
    logging.info(f"Dataset LIDC: number of val data is {len(val_files_1)}.")
    logging.info(f"Dataset LIDC: train data is {train_files_1}.")
    logging.info(f"Dataset LIDC: val data is {val_files_1}.")

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
        logging.info(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
        logging.info(f"{dataset['data_name']}: number of val data is {len(val_files_i)}.")
        
        # Attach modality to each file
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
        val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

    # Print total number of data for each modality
    for modality in train_files.keys():
        logging.info(f"Total number of training data for {modality} is {len(train_files[modality])}.")
        logging.info(f"Total number of val data for {modality} is {len(val_files[modality])}.")

    # Combine training and validation data
    train_files_combined = train_files["ct"] + train_files["mri"]
    val_files_combined = val_files["ct"] + val_files["mri"]

    # Define transformations for training and validation datasets
    train_transform = VAE_Transform(
        is_train=True,
        random_aug=args.random_aug,
        k=4,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=args.val_patch_size,  # if None, will validate on whole image volume
        output_dtype=torch.float16,  # final data type
        spacing_type="original",
        spacing=[1.0, 1.0, 1.0],
        val_crop_size=[256, 256, 256],
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    # Create DataLoader for training and validation
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=8)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    dataset_val = CacheDataset(data=val_files_combined, transform=val_transform, cache_rate=args.cache, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=4, shuffle=False)

    return dataloader_train, dataloader_val

def setup_device():
    """Setup device for training."""
    return torch.device("cuda")

def initialize_models(args, device):
    """Initialize autoencoder and discriminator models."""

    autoencoder = AutoencoderKlReducedMaisi.from_config(args.model_config_file).to(device)
    
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())

    logging.info(autoencoder)
    logging.info(f"number of parameters of the autoencoder = {total_params}")

    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)

    return autoencoder, discriminator

def main(args):

    #create folder for logs 
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    log_directory = args.log_path  # Parent directory where logs are stored
    log_filename = f"vae_{timestamp}.log"
    log_path = os.path.join(log_directory, log_filename)

    # Create the parent directory if it doesn't exist
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Example of logging an info message
    logging.info('Starting training...')

    trained_g_path, trained_d_path, trained_g_path_best, trained_d_path_best, tensorboard_writer = load_configurations(args, timestamp)
    dataloader_train, dataloader_val = prepare_data(args)
    device = setup_device()
    autoencoder, discriminator = initialize_models(args, device)

    # config loss and loss weight
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        logging.info("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        logging.info("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    )

    # config optimizer and lr scheduler
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)


    # please adjust the learning rate warmup rule based on your dataset and n_epochs
    def warmup_rule(epoch):
        # learning rate warmup rule --> The learning rate of each parameter group is set to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.
        if epoch < 10:
            return 0.01
        elif epoch < 20:
            return 0.1
        else:
            return 1.0


    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    # set AMP scaler
    if args.amp:
        # test use mean reduction for everything
        scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
        scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
            
        # Initialize variables
        best_val_recon_epoch_loss = 10000000.0
        total_step = 0
        start_epoch = 0

        # Setup validation inferer
        val_inferer = ( # UNUSED NOW BC WAS GIVING ERROR when no patches used
            SlidingWindowInferer(
                roi_size=args.val_sliding_window_patch_size,
                sw_batch_size=1,
                progress=False,
                overlap=0.0,
                device=torch.device("cpu"),
                sw_device=device,
            )
            if args.val_sliding_window_patch_size
            else SimpleInferer()
        )


    def loss_weighted_sum(losses):
        return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]

    scale_factor_train = 0

    print(f"args.amp is {args.amp}")

    # Training and validation loops
    for epoch in tqdm(range(start_epoch, args.n_epochs), desc="Training Epochs", unit="epoch"):
        torch.cuda.reset_peak_memory_stats()

        logging.info(f"Starting epoch {epoch}.")
        logging.info(f"lr: {scheduler_g.get_lr()}")
        autoencoder.train()
        discriminator.train()
        train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

        # Wrap the dataloader with tqdm to show progress per batch
        for batch in tqdm(dataloader_train, desc=f"Training Epoch {epoch}", unit="batch", leave=False):

            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                # Train Generator
                reconstruction, z_mu, z_sigma = autoencoder(images)
                losses = {
                    "recons_loss": intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": loss_perceptual(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

                if args.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                # Train Discriminator
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

            #scale factor 
            scale_factor_train_iter = 1.0 / z_mu.detach().flatten().std() 
            tensorboard_writer.add_scalar("scale_factor_train_iter", scale_factor_train_iter, total_step)

            scale_factor_train += scale_factor_train_iter

            # Log training loss
            total_step += 1
            for loss_name, loss_value in losses.items():
                tensorboard_writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), total_step)
                if args.enable_wandb:
                    wandb.log({f"train_{loss_name}_iter": loss_value.detach().item()})
                train_epoch_losses[loss_name] += loss_value.item()

            tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
            tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
            tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

            if args.enable_wandb:
                wandb.log({"scale_factor_train_iter": scale_factor_train_iter})
                wandb.log({"train_adv_loss_iter": generator_loss})
                wandb.log({"train_fake_loss_iter": loss_d_fake})
                wandb.log({"train_real_loss_iter": loss_d_real.detach().item()})


        scheduler_g.step()
        scheduler_d.step()

        scale_factor_train /= len(dataloader_train)
        tensorboard_writer.add_scalar("scale_factor/train", loss_value, epoch)

        if args.enable_wandb:
            wandb.log({"scale_factor/train": scale_factor_train})

        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)

        logging.info(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")

        for loss_name, loss_value in train_epoch_losses.items():
            tensorboard_writer.add_scalar(f"{loss_name}/train", loss_value, epoch)
            if args.enable_wandb:
                 wandb.log({f"{loss_name}/train" : loss_value})
        
        torch.save(discriminator.state_dict(), trained_d_path)
        autoencoder.save_pretrained(trained_g_path)
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        logging.info(f"Max memory allocated in training epoch {epoch} is {peak_memory_gb}")

        scale_factor_val = 0
        

        # Validation
        if epoch % args.val_interval == 0:

            torch.cuda.reset_peak_memory_stats()
            autoencoder.eval()
            val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
            
            #select a random bath from the validation set images to save
            batch_select = torch.randint(0, len(dataloader_val), (1,))
            i = 0
            for batch_index, batch in enumerate(tqdm(dataloader_val, desc=f"Val Epoch {epoch}", unit="batch", leave=False)):
               
                #select a random bath from the validation set

                with torch.no_grad():
                    with autocast("cuda", enabled=args.amp):
                        images = batch["image"]
                        #reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images) # sliding window inferer with patch size = image size was giving error
                        reconstruction,z_mu, z_sigma = autoencoder(images.to(device))
                        reconstruction = reconstruction.to(device)
                        val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                        val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item() 
                    
                        val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()

                        scale_factor_val += 1.0 / z_mu.detach().flatten().std() 
                        

                if batch_index == batch_select:
                    logging.info(f"Saving random batch {batch_index} images from validation set.")
               
                    img_video = torch.rot90(images.permute(0, 2, 1, 3, 4).to(device),k=1, dims=(3, 4))
                    recon_video = torch.rot90(reconstruction.permute(0, 2, 1, 3, 4).to(device),k=1, dims=(3, 4))

                    # Ensure both have 3 channels
                    if img_video.shape[2] == 1:  # Check if the channel dimension is 1
                        img_video = img_video.repeat(1, 1, 3, 1, 1)  # Repeat to make it 3 channels

                    if recon_video.shape[2] == 1:  # Check if the channel dimension is 1
                        recon_video = recon_video.repeat(1, 1, 3, 1, 1)  # Repeat to make it 3 channels

                    vid_tensor_sag = torch.cat([img_video, recon_video], dim=-1)
                    tensorboard_writer.add_video("saggital view rand", vid_tensor_sag, global_step=epoch, fps=4, walltime=None)
        
                    img_video_ax = torch.rot90(img_video.permute(0, 3, 2, 1, 4), dims=(3, 4))
                    recon_video_ax = torch.rot90(recon_video.permute(0, 3, 2, 1, 4), dims=(3, 4))
                    vid_tensor_ax = torch.cat([img_video_ax, recon_video_ax], dim=-1)
                    
                    tensorboard_writer.add_video("axial view rand", vid_tensor_ax, global_step=epoch, fps=4, walltime=None)

                    img_video_co = img_video.permute(0, 4, 2, 3, 1)
                    recon_video_co = recon_video.permute(0, 4, 2, 3, 1)
                    vid_tensor_co = torch.cat([img_video_co, recon_video_co], dim=-1)

                    tensorboard_writer.add_video("coronal view rand", vid_tensor_co, global_step=epoch, fps=4, walltime=None)
                i += 1 

            for key in val_epoch_losses:
                val_epoch_losses[key] /= len(dataloader_val)

            val_loss_g = loss_weighted_sum(val_epoch_losses)
            logging.info(f"Epoch {epoch} val_vae_loss {val_loss_g}: {val_epoch_losses}.")

            if val_loss_g < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_loss_g
                autoencoder.save_pretrained(trained_g_path_best)
                torch.save(discriminator.state_dict(), trained_d_path_best)
                logging.info("Got best val vae loss.")
                logging.info(f"Save trained autoencoder to {trained_g_path_best}")
                logging.info(f"Save trained discriminator to {trained_d_path_best}")


            for loss_name, loss_value in val_epoch_losses.items():
                tensorboard_writer.add_scalar(f"{loss_name}/val", loss_value, epoch)
                if args.enable_wandb:
                    wandb.log({f"{loss_name}/val": loss_value})
            
            scale_factor_val /= len(dataloader_val)
            tensorboard_writer.add_scalar("scale_factor/val", scale_factor_val, epoch)
            logging.info(f"Scale factor validation epoch {epoch} = {scale_factor_val}")
            if args.enable_wandb:
                wandb.log({"scale_factor/val": scale_factor_val})

            # Monitor reconstruction result
            img_video = torch.rot90(images.permute(0, 2, 1, 3, 4).to(device),k=1, dims=(3, 4))
            recon_video = torch.rot90(reconstruction.permute(0, 2, 1, 3, 4).to(device),k=1, dims=(3, 4))

            # Ensure both have 3 channels
            if img_video.shape[2] == 1:  # Check if the channel dimension is 1
                img_video = img_video.repeat(1, 1, 3, 1, 1)  # Repeat to make it 3 channels

            if recon_video.shape[2] == 1:  # Check if the channel dimension is 1
                recon_video = recon_video.repeat(1, 1, 3, 1, 1)  # Repeat to make it 3 channels

            vid_tensor_sag = torch.cat([img_video, recon_video], dim=-1)
            tensorboard_writer.add_video("saggital view last", vid_tensor_sag, global_step=epoch, fps=4, walltime=None)
   
            img_video_ax = torch.rot90(img_video.permute(0, 3, 2, 1, 4), dims=(3, 4))
            recon_video_ax = torch.rot90(recon_video.permute(0, 3, 2, 1, 4), dims=(3, 4))
            vid_tensor_ax = torch.cat([img_video_ax, recon_video_ax], dim=-1)
            
            tensorboard_writer.add_video("axial view last", vid_tensor_ax, global_step=epoch, fps=4, walltime=None)

            img_video_co = img_video.permute(0, 4, 2, 3, 1)
            recon_video_co = recon_video.permute(0, 4, 2, 3, 1)
            vid_tensor_co = torch.cat([img_video_co, recon_video_co], dim=-1)

            tensorboard_writer.add_video("coronal view last", vid_tensor_co, global_step=epoch, fps=4, walltime=None)

            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            logging.info(f"Max memory allocated in validation epoch {epoch} is {peak_memory_gb}")

    logging.info("Training completed !")

    # Close tensorboard writer
    tensorboard_writer.close()

    # Close WandB
    if args.enable_wandb:
        wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train VAE model with specified configuration files and dataset.")

    
    parser.add_argument('--model_config_file', type=str, required=True,
                        help='Path to the vae model configuration JSON file.')
    parser.add_argument('--train_config_file', type=str, required=True,
                        help='Path to the vae training configuration JSON file.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory to save the trained model checkpoints.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the preprocessed .npy dataset directory (as produced by '
                             'preproc_lidc_npy.py / preproc_nlst_npy.py), containing <series_id>/chest_ct/*.npy.')
    parser.add_argument('--train_portion', type=float, required=True,
                        help='Portion of the dataset to use for training (e.g., 0.9 for 90%%).')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Directory to save training logs and visualizations.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for this training run (used in logging).')
    parser.add_argument('--tensorboard_log_path', type=str, required=True,
                        help='Path to the folder where TensorBoard logs will be saved.')
    parser.add_argument('--enable_wandb', action='store_true', help="Enable Weights & Biases logging")


    args = parser.parse_args()

    main(args)