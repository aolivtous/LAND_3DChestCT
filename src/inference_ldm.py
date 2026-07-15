import csv
import os
import torch
import argparse
from math import ceil
import sys

from tqdm import tqdm
from contextlib import contextmanager

from utils.preproc_lidc_npy import extract_centroids, crop_around_centroid
from utils.utils_lidc3D import *
from unet.unet import UNetModel
from vae.autoencoder_kl import AutoencoderKlReducedMaisi

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def load_pipeline(model_path, vae_path, mask_mode, mask_dataset):

    print("Loading Diffusion pipeline from:")
    print(f"    - {model_path}\n")
    
    unet = UNetModel.from_pretrained(model_path, subfolder="unet")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    unet.requires_grad_(False)
    unet = unet.cuda()

    print("U-Net model loaded eval mode")

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    #move it to cuda
    print("Loading VAE from", vae_path)
    vae = AutoencoderKlReducedMaisi.from_pretrained(vae_path)
    vae.requires_grad_(False)
    vae = vae.cuda()
    latent_channels = vae.latent_channels
    print("VAE loaded")

    params_vae = sum(p.numel() for p in vae.parameters())
    trainable_params_vae = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE Total params: {params_vae}")
    print(f"VAE Trainable params: {trainable_params_vae}")

    if args.vae_mask_dir is not None:
        print("Loading mask encoder from", args.vae_mask_dir)
        maskEncoder = AutoencoderKlReducedMaisi.from_pretrained(args.vae_mask_dir)
        maskEncoder.requires_grad_(False)
        print("Mask Encoder loaded eval mode")
        params_vae = sum(p.numel() for p in maskEncoder.parameters())
        trainable_params_vae = sum(p.numel() for p in maskEncoder.parameters() if p.requires_grad)
        print(f"VAE Total params: {params_vae}")
        print(f"VAE Trainable params: {trainable_params_vae}")
        maskEncoder = maskEncoder.cuda()
        
    else:
        maskEncoder = None

    scheduler_config_path = os.path.join(model_path, "scheduler/scheduler_config.json")
    noise_scheduler = DDPMScheduler.from_config(scheduler_config_path)

    pipeline = CondLatentDiffusionPipeline_LIDC3D(
        unet=unet,
        scheduler=noise_scheduler,
        vae=vae,
        maskEncoder=maskEncoder,
        latent_channels=latent_channels,
        patchbased=args.patchbased,
        mask_mode=args.mask_mode,
        mask_dataset=args.mask_dataset,
    )
    print("Diffusion pipeline is ready\n")
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--latents_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mask_mode", type=str, required=False)
    parser.add_argument("--mask_dataset", type=str, required=False)
    parser.add_argument("--vae_mask_dir", type=str, default=None)
    parser.add_argument("--save_nodule_crops", action="store_true")
    parser.add_argument(
        "--patchbased",
        action="store_true",
        help="Whether to use Patch-Based Diffusion instead of directly operating on the entire volume.",
    )
    
    args = parser.parse_args()


    torch.manual_seed(0)

    print(f"save dir is {args.save_dir}")

    # Create directory if needed
    os.makedirs(args.save_dir, exist_ok=True)

    # Count existing .npy files (for info only)
    existing_files = sorted([f for f in os.listdir(args.save_dir) if f.endswith(".npy")])
    print(f"Found {len(existing_files)} existing samples in {args.save_dir}")

    pipeline = load_pipeline(args.model_path, args.vae_path, args.mask_mode, args.mask_dataset)

    # Build a lookup so saved output files can be named after the mask that conditioned them.
    # LIDCVolumes always discovers samples via sorted(glob.glob(...)), so a freshly-built
    # instance here yields the exact same index -> mask mapping the pipeline itself used
    # internally (read_N_masks_for_inference, called with the same start_indx per batch below).
    mask_naming_dataset = None
    if args.mask_dataset is not None and args.mask_mode not in (None, "none"):
        mask_naming_dataset = LIDCVolumes(args.mask_dataset, mask_mode=args.mask_mode, masks_only=True)

    batch_size = args.batch_size
    num_batches = ceil(args.num_samples / batch_size)

    verbose = False

    for b in tqdm(range(num_batches), desc=f"Sampling from model: {os.path.basename(args.model_path)}"):

        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, args.num_samples)

        # Collect only samples NOT existing yet
        out_paths = []
        latents = []

        for i in range(start_idx, end_idx):
            if mask_naming_dataset is not None and i < len(mask_naming_dataset):
                mask_filename = os.path.basename(mask_naming_dataset[i]["filename"])
                out_path = os.path.join(args.save_dir, mask_filename)
            else:
                out_path = os.path.join(args.save_dir, f"image_{i:05d}.npy")

            # Skip if file already exists
            if os.path.exists(out_path):
                continue

            out_paths.append(out_path)

            # Optional latent loading
            if args.latents_dir is not None:
                latent = torch.load(os.path.join(args.latents_dir, f"latent_{i}.pt")).squeeze(0).to(pipeline.unet.device)
                latents.append(latent)

        # If nothing needs to be generated this batch → skip
        if len(out_paths) == 0:
            continue

        # Prepare latents if present
        if len(latents) > 0:
            latents = torch.stack(latents, dim=0)
        else:
            latents = None

        current_batch_size = len(out_paths)
        print(f"Sampling indices {start_idx}–{end_idx}, generating {current_batch_size} new samples.")

        # Run the model
        output = pipeline(
            latents=latents,
            height=256,
            width=256,
            batch_size=current_batch_size,
            num_inference_steps=args.steps,
            output_type="numpy",
            return_dict=False,
            renormalize=False,
            return_latents=False,
            start_indx=start_idx
        )

        # Unpack based on mode
        if args.mask_mode != "none":
            if "texture" in args.mask_mode:
                images_, _, _, classes = output
                csv_path = os.path.join(args.save_dir, "image_labels.csv")
                with open(csv_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for path, cls in zip(out_paths, classes):
                        writer.writerow([os.path.basename(path), cls])
            else:
                images_, _, _ = output
        else:
            images_ = output

        # Save images
        for img, out_path in zip(images_, out_paths):
            # img.shape is (3, 256, 256, 256) because output vals were replicated 3 times for RGB-like display
            if img.shape[0] == 3:
                img = img[0]
                assert img.shape[0] == 256
                assert img.shape[1] == 256
                assert img.shape[2] == 256
            np.save(out_path, img)

        del images_, latents
        torch.cuda.empty_cache()

    print(f"Finished inference for model: {args.model_path}")

    # OBTAIN CROPS AROUND THE NODULES
    if args.save_nodule_crops:
        inference_paths = sorted([f for f in os.listdir(args.save_dir) if f.endswith(".npy")])
        crops_dir = args.save_dir + "_Crops"

        if not os.path.exists(crops_dir):
            os.makedirs(crops_dir, exist_ok=True)

        print(f"mask dataset is {args.mask_dataset}")
        inference_dataset = LIDCVolumes(args.mask_dataset, mask_mode=args.mask_mode, masks_only=True)
        synth_samples = len(inference_paths)
        if synth_samples < args.num_samples:
            print(f"Warning: {args.num_samples - synth_samples} samples missing from {args.save_dir}")

        for i, volume in enumerate(inference_dataset):
            if i >= synth_samples:
                break
            name = volume["filename"]
            print(name)
            mask = volume["mask"]
            synt_path = os.path.join(args.save_dir, inference_paths[i])
            synt_vol = np.load(synt_path)
            print(f"Processing {synt_path}: {synt_vol.shape}")

            tmp_mask = (mask * 5 >= 1) * 1 if args.mask_mode == "nodule+lung+texture" else (mask >= 1) * 1

            if tmp_mask.any():
                nodule_centroids = extract_centroids(tmp_mask[0])
                nodule_count = 0
                print(f"found a total of {len(nodule_centroids)} nodules")
                for nodule_centroid in nodule_centroids:
                    cx, cy, cz = map(int, map(round, nodule_centroid))
                    new_name = inference_paths[i].split('.')[0] + f"_{nodule_count}.npy"
                    out_path = os.path.join(crops_dir, new_name)

                    if os.path.exists(out_path):
                        print(f"Skipping existing crop: {new_name}")
                        nodule_count += 1
                        continue

                    crop = crop_around_centroid(synt_vol, (cx, cy, cz), crop_size = 32)
                    np.save(out_path, crop)
                    print(f"Saved crop: {new_name}")
                    nodule_count += 1