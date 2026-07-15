import inspect
import logging
import math
import os
import shutil
import os
import accelerate
import datasets
import diffusers
import torch
import torch.nn.functional as F

from datetime import timedelta
from pathlib import Path
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, LoggerType
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipeline.pipeline import *
from utils.opts import *
from utils.preproc_lidc_npy import *
from utils.utils_lidc3D import *
from unet.unet import UNetModel
from vae.autoencoder_kl import AutoencoderKlReducedMaisi

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DEBUG = False
USE_VAE = True
CHECK_CUDA_MEM = False
SAVE_REAL_SAMPLES = False

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def main(args):
    
    USE_VAE = False if args.patchbased else True

    logging_dir = args.logging_dir #os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset

    print(f"args.logger is {args.logger}")
    if args.logger == "both":
        log_type = ["wandb", LoggerType.TENSORBOARD]
    else:
        log_type = args.logger

    print(f"log type is {log_type}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_type,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard" or args.logger == "both":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb" or args.logger == "both":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Set the random seed manually for reproducibility.
    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                model_names = ["unet", "emb"]
                for i, (model, folder_name) in enumerate(zip(models, model_names)):
                    model.save_pretrained(os.path.join(output_dir, folder_name))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            print("#########################")
            print(f"input dir is {input_dir}")
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            model_names = ["unet", "emb"]
            for i, (model, folder_name) in enumerate(zip(models, model_names)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetModel.from_pretrained(input_dir, subfolder=folder_name)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    if CHECK_CUDA_MEM:
        print("initial cuda mem:", torch.cuda.mem_get_info())
        #exit()

    # Load pretrained VAE model
    if USE_VAE:
        print("Loading VAE from", args.vae_dir)
        vae = AutoencoderKlReducedMaisi.from_pretrained(args.vae_dir)
        latent_channels = vae.latent_channels
        vae.requires_grad_(False)
        vae.to(accelerator.device)
        
    else:
        vae = None
        latent_channels = None

    if args.vae_mask_dir is not None:
        print("Loading mask encoder from", args.vae_mask_dir)
        maskEncoder = AutoencoderKlReducedMaisi.from_pretrained(args.vae_mask_dir)
        maskEncoder.requires_grad_(False)
        maskEncoder.to(accelerator.device)
        
    else:
        maskEncoder = None

    # Initialize the U-Net diffusion model
    if args.model_config_name_or_path is None:

        conditional_pipeline = (args.mask_mode != "none") | args.patchbased
        assert not ((args.mask_mode != "none") and args.patchbased), "patchbased and mask condition cannot be used simultaneously"
        condition_channels = 3 if args.patchbased else 1
        condition_channels = condition_channels*conditional_pipeline

        cross_attention_dim = condition_channels if (conditional_pipeline and args.attention) else None
        in_channels = 1 if not USE_VAE else latent_channels

        channel_mult = (1,3,4,4,4,4,4) if not USE_VAE else (1,3,4,4,4)
        model_channels = 128  #128 canals i (1,3,4,4,4) per 64x64x64 --> 32 canals i (1,3,3,4,4,4) per 128x128x128
        model = UNetModel(
            in_channels=in_channels + condition_channels,
            model_channels=64 if args.patchbased else model_channels,
            out_channels=in_channels,
            num_res_blocks=2,
            attention_resolutions=(16,),  # must be a tuple of numbers: 1, 2, 4, 8, 16
            dropout=0,
            channel_mult=(1,3,4,8,8,8) if args.patchbased else channel_mult,
            dims=3,
            use_checkpoint=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order = False,
            num_groups=32,
            resample_2d=False,
            additive_skips=True,
            self_attention_blocks = ["input", "middle", "output"] if (not conditional_pipeline and args.attention) else [],
            cross_attention_blocks = ["input", "middle", "output"],
            cross_attention_dim = cross_attention_dim,
        )
        print("Model successfully created")
        print("cross attention dim is", cross_attention_dim)
        #model.to(accelerator.device)

    else:

        conditional_pipeline = (args.mask_mode != "none") | args.patchbased
        assert not ((args.mask_mode != "none") and args.patchbased), "patchbased and mask condition cannot be used simultaneously"
        condition_channels = 3 if args.patchbased else 1
        condition_channels = condition_channels*conditional_pipeline
        cross_attention_dim = condition_channels if (conditional_pipeline and args.attention) else None
        model = UNetModel.from_pretrained(args.model_config_name_or_path, subfolder=f"unet")
        print(f"Succesfully loaded UNET from {args.model_config_name_or_path} subfolder unet")
        #model.to(accelerator.device)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNetModel, #UNet2DModel,
            model_config=model.config_,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            clip_sample=False if USE_VAE else True,
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(clip_sample=False, num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)


    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [*model.parameters()],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets
    renormalize = False if USE_VAE else True
    dataset = LIDCVolumes(args.dataset_name,
                     normalize=(lambda x: 2*x - 1) if renormalize else None,
                     mode='train',
                     concat_coords=args.patchbased,
                     patch_based=args.patchbased,
                     mask_mode=args.mask_mode,
                     useMaskEncoder = (True if (args.mask_mode != "none" and args.vae_mask_dir is not None) else False)
                     )

    logger.info(f"Dataset size: {len(dataset)}")

    print("\n\ndataset length is", len(dataset), "\n\n")

    #dataset.set_transform(parse_lidc_metadata)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )


    # Prepare everything with our `accelerator`
    #"""
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    #"""
    """
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    """

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training !!! *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
            
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(os.path.dirname(args.resume_from_checkpoint)).split("-")[1])
        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    if CHECK_CUDA_MEM:
        print("cuda mem before train loop:", torch.cuda.mem_get_info())


    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["image"].type(weight_dtype).to(accelerator.device)
            images = clean_images[:, :1]

            if args.mask_mode != "none":
                
                masks = batch["mask"].type(weight_dtype).to(accelerator.device) # (B, 1, D, H, W)
                mask_latents = mask_downsample(masks, maskEncoder=maskEncoder) 

            #if DEBUG:
            #    global_step += 1
            #    break

            # clean_images.shape = (batchsize, channels, D, H, W)
            # coords.shape = (batchsize, 3, D, H, W)

            if USE_VAE:
               
                with torch.amp.autocast("cuda", enabled=True):
                    #latents = vae.encode(images).latent_dist.sample()
                    z_mu, z_sigma = vae.encode(images)
                    latents = vae.sampling(z_mu, z_sigma)

            else:
                latents = images[:, 0:1, :, :, :]

            if args.patchbased:
                coord_latents = clean_images[:, 1:]
                if USE_VAE:
                    deltaD, deltaH, deltaW = np.array(clean_images.shape[-3:])/np.array(latents.shape[-3:]) # compression factors in spatial dims
                    coord_latents = coord_latents[:, :, ::int(deltaD), ::int(deltaH), ::int(deltaW)]

            # Sample noise that we'll add to the images
            noise = torch.randn(latents.shape, dtype=weight_dtype, device=accelerator.device)
            bsz = latents.shape[0]  # batch size
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Concatenate condition
            if args.mask_mode != "none":
                noisy_latents = torch.cat((noisy_latents, mask_latents), 1)
                cond_latents = mask_latents.reshape(bsz, -1, condition_channels)
            elif args.patchbased:
                noisy_latents = torch.cat((noisy_latents, coord_latents), 1)
                cond_latents = coord_latents.reshape(bsz, -1, condition_channels)
            else:
                cond_latents = None
                

            if CHECK_CUDA_MEM:
                print("cuda mem before train step:", torch.cuda.mem_get_info())

            with accelerator.accumulate(model):

                # Predict the noise residual
                if cross_attention_dim:
                    model_output = model(x=noisy_latents, timesteps=timesteps, context=cond_latents)
                else:
                    model_output = model(x=noisy_latents, timesteps=timesteps)

                if args.prediction_type == "epsilon": # this is the default loss that compares predicted vs ground truth noise
                    
                    if args.snr_gamma is None:
                        #print("gamma is none")
                        loss = F.mse_loss(model_output.float(), noise.float())
                    else:

                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        
                        mse_loss_weights = mse_loss_weights / snr
                      
                        loss = F.mse_loss(model_output.float(), noise.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()

                elif args.prediction_type == "v_prediction":
                    
                    #target = noise_scheduler.get_velocity(latents, noise, timesteps) changed for KARRAS MULTISTEP SCHEDULER
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
                    sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
                    sqrt_one_minus_alpha = (1.0 - alphas_cumprod[timesteps]) ** 0.5

                    target = sqrt_alpha[:, None, None, None] * noise - sqrt_one_minus_alpha[:, None, None, None] * latents

                    if args.snr_gamma is None:
                        
                        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
                    else:
                        
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]

                        mse_loss_weights = mse_loss_weights / (snr + 1)

                        loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")


                accelerator.backward(loss)
                #if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or DEBUG:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if DEBUG:
                break

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1 or DEBUG:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

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

                generator = torch.Generator(device=accelerator.device).manual_seed(0)

                #"""
                # run pipeline in inference (sample random noise and denoise)
                output = pipeline(
                    generator=generator,
                    height = args.resolution,
                    width = args.resolution, 
                    batch_size=1,
                    num_inference_steps=3 if DEBUG else args.ddpm_num_inference_steps,
                    output_type="numpy",
                    return_dict=False,
                    renormalize=renormalize,
                )
                if args.mask_mode != "none":
                    if len(output) == 4:
                        out_images, input_masks, output_masks, texture = output
                    else:
                        out_images, input_masks, output_masks = output
                        texture = None
                else:
                    out_images = output

                if not USE_VAE:
                    out_images = np.tile(out_images, (1, 3, 1, 1, 1))
                if args.mask_mode != "none":
                    
                    if input_masks.shape[1] > 1:

                        tmp_masks = np.argmax(input_masks, axis=1)  
                        tmp_masks = np.expand_dims(tmp_masks, axis=1)  
                    
                        if args.mask_mode == "nodule+lung":
                            tmp_masks = (tmp_masks == 1)
                        elif args.mask_mode == "nodule+lung+texture":
                            tmp_masks = ((tmp_masks >= 1) & (tmp_masks <= 5))  
                        
                    else:
                        if args.mask_mode == "nodule+lung+texture":
                            tmp_masks = ((input_masks * 5 >= 1)).astype(int)
                        else:
                            tmp_masks = ((input_masks >= 1)).astype(int)

                    out_images = merge_images_with_masks(out_images, tmp_masks)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (out_images * 255).round().astype("uint8")

                # save samples to disk for visualizaton
                out_dir = os.path.join(args.output_dir, f"images/step-{global_step}_epoch-{epoch}")
                save_vol_gifs(images_processed, out_dir)

                if args.logger == "tensorboard" or args.logger == "both":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    
                    vid_tensor = torch.from_numpy(images_processed).permute(0, 2, 1, 3, 4)
                    #vid_tensor = vid_tensor.clamp(min=0., max=1.)
                    # vid_tensor is expected to have shape (B,T,C,H,W), [0, 255] vals for uint8 or [0, 1] vals for float
                    # C - image channels, must be 3 because only rgb images are considered
                    # B - batch size
                    # T - time dimension (number of frames)
                    tracker.add_video("axial view", vid_tensor, global_step=global_step, fps=6, walltime=None)
                    vid_tensor_ = vid_tensor.permute(0, 3, 2, 1, 4)
                    tracker.add_video("coronal view", vid_tensor_, global_step=global_step, fps=6, walltime=None)
                    vid_tensor_ = vid_tensor.permute(0, 4, 2, 1, 3)
                    tracker.add_video("saggital view", vid_tensor_, global_step=global_step, fps=6, walltime=None)

                    nx, ny, nz = args.resolution//2, args.resolution//2, args.resolution//2
                    if args.mask_mode != "none":
                        if input_masks.shape[1] > 1:
                            print("input masks has multiple channels")
                            mask_vol = np.argmax(input_masks, axis=1)  # shape: (1, 256, 256, 256)
                        
                            #normalize to 0-1
                            mask_vol = (mask_vol - mask_vol.min())/(mask_vol.max() - mask_vol.min())
                        
                        else: 
                            mask_vol = input_masks[0, :, :, :, :]
                        
                        try:
                            # leave only the connected components corresponding to nodules
                            print(tmp_masks.shape)
                            print(f"unique vals in tmp_masks: {np.unique(tmp_masks)}")
                            
                            nodule_centroids = extract_centroids(tmp_masks[0])
                            nx, ny, nz = nodule_centroids[0]
                        except:
                            print(f"warning: could not find nodule in {args.mask_mode} mask")
                        
                    print(mask_vol.shape, images_processed.shape)
                    slice_x = images_processed[0, :, int(nx), :, :].transpose((1,2,0))
                    slice_y = images_processed[0, :, :, int(ny), :].transpose((1,2,0))
                    slice_z = images_processed[0, :, :, :, int(nz)].transpose((1,2,0))
                    ct_thumbnail_img = np.hstack([slice_x, slice_y, slice_z]).transpose(2, 0, 1)[np.newaxis, ...]
                    tracker.add_images("ct_thumbnail", ct_thumbnail_img, global_step=global_step)

                    if args.mask_mode != "none":
                        slice_x = mask_vol[:, int(nx), :, :].transpose((1,2,0))
                        slice_y = mask_vol[:, :, int(ny), :].transpose((1,2,0))
                        slice_z = mask_vol[:, :, :, int(nz)].transpose((1,2,0))
                        mask_thumbnail_img = np.hstack([slice_x, slice_y, slice_z]).transpose(2, 0, 1)[np.newaxis, ...]
                        mask_thumbnail_img = (mask_thumbnail_img*255).round().astype("uint8")
                        tracker.add_images("mask_thumbnail", mask_thumbnail_img, global_step=global_step)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1 or (DEBUG and epoch == 0):
                
                # save the model
                unet = accelerator.unwrap_model(model)
                #nodule_features_emb = accelerator.unwrap_model(nodule_features_emb)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = CondLatentDiffusionPipeline_LIDC3D(
                    unet=unet,
                    scheduler=noise_scheduler,
                    vae=vae
                )

                params_dict = {"unet_last": unet.parameters()} #"vae": vae.parameters()}  #, "emb": nodule_features_emb.params}
                print(args.output_dir,)
                pipeline.save_pretrained(args.output_dir, params=params_dict)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

            if epoch % 50 == 0:
                
                print("additional saving every 50 epochs")
                epoch_output = os.path.join(args.output_dir, f"epoch_ckpts/epoch-{epoch}")
                os.makedirs(os.path.dirname(epoch_output), exist_ok=True)
                
                # save the model
                unet = accelerator.unwrap_model(model)
                #nodule_features_emb = accelerator.unwrap_model(nodule_features_emb)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = CondLatentDiffusionPipeline_LIDC3D(
                    unet=unet,
                    scheduler=noise_scheduler,
                    vae=vae
                )

                params_dict = {"unet": unet.parameters()} #"vae": vae.parameters()}  #, "emb": nodule_features_emb.params}
                pipeline.save_pretrained(epoch_output, params=params_dict)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=epoch_output,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )


        if DEBUG:
            if epoch >= 5:
                print("all completed with success")
                exit()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
