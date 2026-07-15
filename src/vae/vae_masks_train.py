# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# This file is based on code from the MONAI MAISI Project

import argparse
import json
import os
import time
import torch
import logging
import warnings
import wandb

from pathlib import Path
from monai.config import print_config
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from autoencoder_kl import AutoencoderKlReducedMaisi
from utils.masks_utils import *

warnings.filterwarnings("ignore")

# Print Monai configuration
print_config()

def save_checkpoint(epoch, autoencoder, optimizer, scheduler, scaler, best_val_loss, path):
    checkpoint = {
        "epoch": epoch,
        "autoencoder_state": autoencoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")

def load_configurations(args):
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        sub_folder_dir = os.path.dirname(args.resume_checkpoint)
        folder_name = os.path.basename(sub_folder_dir)
        
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        folder_name = f"{args.run_name}_{timestamp}"

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        sub_folder_dir = os.path.join(args.model_dir, folder_name)

    log_directory = args.log_path
    Path(log_directory).mkdir(parents=True, exist_ok=True)
    log_filename = f"{folder_name}.log"
    log_path = os.path.join(log_directory, log_filename)
    print(f"log_path is {log_path}")
        
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting mask VAE training...")

    if args.resume_checkpoint:
        logging.info("Resuming training...")
    else:
        logging.info("Starting new training ...")

    Path(sub_folder_dir).mkdir(parents=True, exist_ok=True)
    trained_g_path = os.path.join(sub_folder_dir, "vae_last_epoch")
    trained_g_path_best = os.path.join(sub_folder_dir, "vae_best_epoch")
    logging.info(f"Last Trained model will be saved as {trained_g_path} ")
    logging.info(f"Best Trained model will be saved as {trained_g_path_best}")

    tensorboard_path = os.path.join(args.tensorboard_log_path, folder_name)
    Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_path)
    logging.info(f"Tensorboard event will be saved as {tensorboard_path}.")

    config_dict = json.load(open(args.model_config_file, "r"))
    config_dict['in_channels'] = args.num_classes
    config_dict['out_channels'] = args.num_classes
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

    if args.enable_wandb:
        
        print(f"RUN ID IS {folder_name}")
        wandb.init(project="Reduced Maisi VAE", config=args, id=folder_name, resume="allow")
        logging.info(f"Resumed WandB run {folder_name}")
    
    return trained_g_path, trained_g_path_best, sub_folder_dir, log_path, tensorboard_writer

def prepare_data(args):
    train_dataset = LIDCMasks(
        directory=args.dataset_path,
        mask_mode=args.mask_mode,
        num_classes=args.num_classes,
        split="train",
        val_ratio=1-args.train_portion,
        seed=args.seed if hasattr(args, "seed") else 42,
        use_onehot=True,  # Changed to False to use single-channel input to use less memory
        sdf_flag=args.sdf_flag,
        original_textures=args.original_textures
    )

    val_dataset = LIDCMasks(
        directory=args.dataset_path,
        mask_mode=args.mask_mode,
        num_classes=args.num_classes,
        split="val",
        val_ratio=1-args.train_portion,
        seed=args.seed if hasattr(args, "seed") else 42,
        use_onehot=True,  # Changed to False to use single-channel input to use less memory
        sdf_flag=args.sdf_flag,
        original_textures=args.original_textures
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=8
    )

    return train_dataloader, val_dataloader

def setup_device():
    return torch.device("cuda")

def initialize_models(args, device):
    autoencoder = AutoencoderKlReducedMaisi.from_config(args.model_config_file).to(device)
    total_params = sum(p.numel() for p in autoencoder.parameters())
    logging.info(autoencoder)
    logging.info(f"number of parameters of the autoencoder = {total_params}")
    return autoencoder

def colorize_mask(mask_index, num_classes=7):
    colors = torch.tensor([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ], device=mask_index.device, dtype=torch.uint8)

    assert num_classes <= colors.shape[0], f"Palette only defined up to {colors.shape[0]} classes"
    color_mask = colors[mask_index.long()]
    return color_mask.permute(0, 1, 4, 2, 3)

def main(args):

    trained_g_path, trained_g_path_best, trained_checkpoint, log_path, tensorboard_writer = load_configurations(args)
    print(f"logpath is {log_path}")

    dataloader_train, dataloader_val = prepare_data(args)
    device = setup_device()
    autoencoder = initialize_models(args, device)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr, eps=1e-6 if args.amp else 1e-8)

    def warmup_rule(epoch):
        if epoch < 10: return 0.01
        elif epoch < 20: return 0.1
        else: return 1.0
    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)

    scaler_g = GradScaler(enabled=args.amp, init_scale=2.0**8) if args.amp else None
    best_val_loss = float("inf")

    no_improvement_epochs = 0
    start_epoch = 0
    if getattr(args, "resume_checkpoint", None):
        if os.path.exists(args.resume_checkpoint):
            logging.info(f"Resuming training from checkpoint {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            autoencoder.load_state_dict(checkpoint["autoencoder_state"])
            optimizer_g.load_state_dict(checkpoint["optimizer_state"])
            if scheduler_g and checkpoint["scheduler_state"]:
                scheduler_g.load_state_dict(checkpoint["scheduler_state"])
            if scaler_g and checkpoint["scaler_state"]:
                scaler_g.load_state_dict(checkpoint["scaler_state"])
            best_val_loss = checkpoint["best_val_loss"]
            start_epoch = checkpoint["epoch"] + 1
            logging.info(f"Resumed from epoch {start_epoch} with best_val_loss={best_val_loss}")

    for epoch in tqdm(range(start_epoch, args.n_epochs), desc="Training Epochs"):
        autoencoder.train()
        train_epoch_loss = 0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_ce_loss = 0.0
        train_dice = 0.0
        train_regression_loss = 0.0
        train_eik_loss = 0.0


        for batch in tqdm(dataloader_train, desc=f"Training Epoch {epoch}", leave=False):
            optimizer_g.zero_grad(set_to_none=True)
            
            with autocast("cuda", enabled=args.amp):
                
                if args.sdf_flag:
                    target_sdf = batch["mask_sdf"].to(device)
                    recon, mu, sigma = autoencoder(target_sdf)
                    loss_g, sdf_l1, eik_loss, kl = vae_loss_sdf(recon, target_sdf, mu, sigma, beta=args.kl_weight)
                else:
                    masks_input = batch["mask_input"].to(device)
                    recon, mu, sigma = autoencoder(masks_input)
                    masks_index  = batch["mask_index"].to(device)
                    loss_g, recon_loss, kl, ce, dice = vae_loss_segmentation(recon, masks_index, mu, sigma, args.num_classes, args.mask_mode, beta=args.kl_weight)

            if args.amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            if args.sdf_flag:
                train_epoch_loss += loss_g.item()
                train_regression_loss += sdf_l1.item()
                train_eik_loss += eik_loss.item()
                train_kl_loss += kl.item()
            else:
                train_epoch_loss += loss_g.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl.item()
                train_ce_loss += ce.item()
                train_dice += dice.item()


        train_epoch_loss /= len(dataloader_train)
        train_recon_loss /= len(dataloader_train)
        train_kl_loss /= len(dataloader_train)
        train_ce_loss /= len(dataloader_train)
        train_dice /= len(dataloader_train)
        train_regression_loss /= len(dataloader_train)
        train_eik_loss /= len(dataloader_train)

        tensorboard_writer.add_scalar("vae_loss/train", train_epoch_loss, epoch)
        if args.enable_wandb: wandb.log({"vae_loss/train": train_epoch_loss})
        logging.info(f"Epoch {epoch} train_vae_loss: {train_epoch_loss}")
        logging.info(f"Epoch {epoch} train_recon_loss: {train_recon_loss}, train_kl: {train_kl_loss}, train_ce: {train_ce_loss}, train_dice: {train_dice}, train_regression_loss: {train_regression_loss}, train_eik_loss: {train_eik_loss}")

        scheduler_g.step()

        # Log train visuals once per epoch
        if epoch % args.log_images_interval == 0 and not args.sdf_flag:
            print("Logging train reconstructions to tensorboard...")
            
            img_video   = colorize_mask(masks_index,args.num_classes)            # ground truth
            recon_video = colorize_mask(recon.argmax(1),args.num_classes)       # prediction

            tensorboard_writer.add_video("sagittal/train", torch.cat([img_video, recon_video], dim=-1), global_step=epoch, fps=4)
            tensorboard_writer.add_video("axial/train", torch.cat([torch.rot90(img_video.permute(0,3,2,1,4), dims=(3,4)),
                                                                        torch.rot90(recon_video.permute(0,3,2,1,4), dims=(3,4))], dim=-1), global_step=epoch, fps=4)
            tensorboard_writer.add_video("coronal/train", torch.cat([img_video.permute(0,4,2,3,1), recon_video.permute(0,4,2,3,1)], dim=-1), global_step=epoch, fps=4)
                

        if epoch % args.val_interval == 0:
            print("Validation...")
            autoencoder.eval()
            val_epoch_loss = 0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_ce_loss = 0.0
            val_dice = 0.0
            val_regression_loss = 0.0
            val_eik_loss = 0.0


            with torch.no_grad():
                with autocast("cuda", enabled=args.amp):
                    for batch in dataloader_val:
   
                        if args.sdf_flag:
                            target_sdf_val = batch["mask_sdf"].to(device)
                            pred_val, mu_val, sigma_val = autoencoder(target_sdf_val)
                            loss_val, sdf_l1_val, eik_loss_val, kl_val = vae_loss_sdf(pred_val, target_sdf_val, mu_val, sigma_val, beta=args.kl_weight)
                            val_epoch_loss += loss_val.item()
                            val_regression_loss += sdf_l1_val.item()
                            val_eik_loss += eik_loss_val.item()
                            val_kl_loss += kl_val.item()
                        else:
                            masks_input_val = batch["mask_input"].to(device)
                            masks_index_val  = batch["mask_index"].to(device)
                            recon_val, mu_val, sigma_val = autoencoder(masks_input_val) 
                            loss_val,  recon_loss_val, kl_val, ce_val, dice_val   = vae_loss_segmentation(recon_val, masks_index_val, mu_val, sigma_val, args.num_classes,args.mask_mode, beta=args.kl_weight)
                            val_epoch_loss += loss_val.item()
                            val_recon_loss += recon_loss_val.item()
                            val_kl_loss += kl_val.item()
                            val_ce_loss += ce_val.item()
                            val_dice += dice_val.item()

                        
                val_epoch_loss /= len(dataloader_val)
                val_recon_loss /= len(dataloader_val)
                val_kl_loss /= len(dataloader_val)
                val_ce_loss /= len(dataloader_val)
                val_dice /= len(dataloader_val)
                val_regression_loss /= len(dataloader_val)
                val_eik_loss /= len(dataloader_val)

                tensorboard_writer.add_scalar("vae_loss/val", val_epoch_loss, epoch)
                if args.enable_wandb: wandb.log({"vae_loss/val": val_epoch_loss})
                logging.info(f"Epoch {epoch} val_vae_loss: {val_epoch_loss}")
                logging.info(f"Epoch {epoch} val_recon_loss: {val_recon_loss}, val_kl: {val_kl_loss}, val_ce: {val_ce_loss}, val_dice: {val_dice}, val_regression_loss: {val_regression_loss}, val_eik_loss: {val_eik_loss}")

                if epoch % args.log_images_interval == 0 and not args.sdf_flag:
                    print("Logging val reconstructions to tensorboard...")
    
                    img_video_val   = colorize_mask(masks_index_val, args.num_classes)            # ground truth
                    recon_video_val = colorize_mask(recon_val.argmax(1),args.num_classes)       # prediction

                    tensorboard_writer.add_video("sagittal/val", torch.cat([img_video_val, recon_video_val], dim=-1), global_step=epoch, fps=4)
                    tensorboard_writer.add_video("axial/val", torch.cat([torch.rot90(img_video_val.permute(0,3,2,1,4), dims=(3,4)),
                                                                                torch.rot90(recon_video_val.permute(0,3,2,1,4), dims=(3,4))], dim=-1), global_step=epoch, fps=4)
                    tensorboard_writer.add_video("coronal/val", torch.cat([img_video_val.permute(0,4,2,3,1), recon_video_val.permute(0,4,2,3,1)], dim=-1), global_step=epoch, fps=4)
                
                if val_epoch_loss < best_val_loss - args.early_stopping_min_delta:
                    best_val_loss = val_epoch_loss
                    no_improvement_epochs = 0
                    autoencoder.save_pretrained(trained_g_path_best)
                    logging.info(f"Saved best model & checkpoint to {trained_g_path_best}")
                else:
                    no_improvement_epochs += 1
                    logging.info(f"No improvement for {no_improvement_epochs} epochs.")
                    if no_improvement_epochs >= args.early_stopping_patience:
                        logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                        save_checkpoint(epoch, autoencoder, optimizer_g, scheduler_g, scaler_g, best_val_loss, os.path.join(trained_checkpoint, f"last_checkpoint.pth"))
                        autoencoder.save_pretrained(trained_g_path)
                        tensorboard_writer.close()
                        if args.enable_wandb: wandb.finish()
                        return
                    
        save_checkpoint(epoch, autoencoder, optimizer_g, scheduler_g, scaler_g, best_val_loss, os.path.join(trained_checkpoint, f"last_checkpoint.pth"))

        autoencoder.save_pretrained(trained_g_path)

    logging.info("Training completed!")
    tensorboard_writer.close()
    if args.enable_wandb: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train VAE model with specified configuration files and dataset.")
    parser.add_argument('--model_config_file', type=str, required=True)
    parser.add_argument('--train_config_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--train_portion', type=float, required=True)
    parser.add_argument("--mask_mode", type=str, default="nodule+lung",
                        choices=["nodule", "lung", "nodule+lung", "nodule+lung+texture"])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--tensorboard_log_path', type=str, required=True)
    parser.add_argument('--enable_wandb', action='store_true', help="Enable Weights & Biases logging")

    parser.add_argument('--log_images_interval', type=int, default=1,
                        help='Log train/val images every N epochs.')
   
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of validation epochs with no improvement after which training will stop.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum change in val loss to be considered an improvement.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from.')
    parser.add_argument('--sdf_flag', action='store_true', help="Use SDF regression loss instead of segmentation loss")
    parser.add_argument('--original_textures', action='store_true', help="Use SDF regression loss instead of segmentation loss")

    args = parser.parse_args()
    main(args)
