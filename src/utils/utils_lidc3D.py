import glob
import os
import os.path
import nibabel
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, label, gaussian_filter
from scipy.interpolate import UnivariateSpline

from vae.autoencoder_kl import AutoencoderKlReducedMaisi
from unet.unet import UNetModel
from pipeline.pipeline import *

def fit_nparray_to_given_size(arr, size=256):
    h, w, d = arr.shape
    if w != size or h != size or d != size:
        arr_ = np.zeros((size,size,size))
        arr_[:h, :w, :d] = arr[:size, :size, :size]
    else:
        arr_ = arr
    return arr_

def mask_downsample(mask, maskEncoder=None, factor=4):
    # mask shape is BxCxDxHxW
    #mask_downsampled = mask[:, :, ::factor, ::factor, ::factor]
    if maskEncoder is None:
        print("Warning: maskEncoder is None, downsampling using maxPool")
        mask_downsampled = F.max_pool3d(mask, kernel_size=factor, stride=factor)
    else:
        with torch.amp.autocast("cuda", enabled=True):
            z_mu, z_sigma = maskEncoder.encode(mask)
            mask_downsampled = maskEncoder.sampling(z_mu, z_sigma)
            
            
    return mask_downsampled

def mask_upsample(mask, factor=4):
    # mask shape is BxCxDxHxW
    return mask.tile((1, 1, factor, factor, factor))

class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, normalize=None, mode='train', concat_coords=False, patch_based=False, useMaskEncoder=False, mask_mode="none", masks_only=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have the name: processed.nii.gz

        masks_only: set True when this dataset is only ever going to be used to supply
                    conditioning masks (e.g. as --mask_dataset). In that case chest_ct/*.npy
                    is never read at all -- samples are discovered directly via mask/*.npy,
                    and a zero-filled placeholder standing in for the real image is built
                    from the mask's own shape. Requires mask_mode != "none". Leave False
                    (the default) for the primary image dataset, where chest_ct is real data
                    that's actually used.
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.concat_coords = concat_coords
        self.patch_based = patch_based
        self.mask_mode = mask_mode
        self.useMaskEncoder = useMaskEncoder
        self.masks_only = masks_only
        self.database = []

        if masks_only:
            assert self.mask_mode != "none", "masks_only=True requires mask_mode != 'none'"

        if self.mask_mode == "none":
            self.num_classes = 1
        elif self.mask_mode == "nodule":
            self.num_classes = 2
        elif self.mask_mode == "nodule+lung" or self.mask_mode == "nodule+lung+t":
            self.num_classes = 3
        else:
            assert self.mask_mode == "nodule+lung+texture"
            self.num_classes = 7

        if self.patch_based and self.useMaskEncoder:
            print("Warning: patch_based=True and useMaskEncoder=True. This setting is not possible since the maskEncoder is not trained with patches. Use useMaskEncoder=False insetad to downsample the masks using maxPool!")
            raise ValueError("Invalid setting")
        
        self.all_images = []

        if masks_only:
            # Discover samples directly via their masks; chest_ct is never required to exist.
            all_paths = sorted(glob.glob(self.directory + "/**/mask/*.npy", recursive=True))
        else:
            # Your original list
            all_paths = sorted(glob.glob(self.directory + "/**/chest_ct/*.npy", recursive=True))

        #self.image_paths = sorted(glob.glob(self.directory + "/*.npy", recursive=True))
        self.image_paths = all_paths
        self.n_images = len(self.image_paths)

    def __getitem__(self, x, train=True):

        image_path = self.image_paths[x]

        if self.masks_only:
            # self.image_paths holds mask/*.npy paths directly in this mode (see __init__).
            # chest_ct is never read -- build a zero-filled placeholder of the mask's own
            # shape to stand in for the unused "image" key.
            mask_path = image_path
            assert os.path.exists(mask_path), "Error: mask path was not found!"
            mask_arr = np.load(mask_path)

            if not self.useMaskEncoder:
                if self.mask_mode == "nodule":
                    mask_arr = (mask_arr >= 1) * 1
                elif self.mask_mode == "nodule+lung":
                    mask_arr[mask_arr >= 1] = 1
                else:
                    assert self.mask_mode == "nodule+lung+texture"
                    mask_arr = mask_arr / 5.0  # normalize mask between 0 and 1
                mask_input = torch.from_numpy(mask_arr).unsqueeze(0).to(torch.float32)  # shape (1, 256, 256, 256)
            else:
                mask_arr = mask_arr.copy()
                if self.mask_mode == "nodule":
                    mask_arr = (mask_arr >= 1).astype(np.int64)  # 0=background, 1=nodule
                elif self.mask_mode == "nodule+lung":
                    mask_arr[mask_arr >= 1] = 1  # nodules
                    mask_arr[mask_arr == 0.5] = 2  # lungs
                else:
                    assert self.mask_mode == "nodule+lung+texture"
                    mask_arr[mask_arr == 0.5] = 6  # lungs
                    mask_arr = np.round(mask_arr).astype(int)
                mask_arr = mask_arr.astype(np.int64)
                mask_index = torch.from_numpy(mask_arr).long()  # [B,D,H,W]
                mask_input = F.one_hot(mask_index, num_classes=self.num_classes).permute(3, 0, 1, 2).float()  # [B,C,D,H,W]

            image = torch.zeros((1, *mask_input.shape[-3:]), dtype=torch.float32)  # unused placeholder, correct spatial shape
            return {"image": image, "y": 1, "offset": [0, 0, 0], "mask": mask_input, "filename": image_path}

        # ---- original behavior below, unchanged ----
        if os.path.splitext(image_path)[-1] == ".npy":
            arr = np.load(image_path)
            #arr = fit_nparray_to_given_size(arr, size=256)
            if self.mask_mode != "none" and not self.useMaskEncoder:
                
                mask_path = image_path.replace("/chest_ct/", "/mask/")
                print(f"Loading mask from {mask_path}")
                assert os.path.exists(mask_path), "Error: mask path was not found!"
                mask_arr = np.load(mask_path)
                # conditional mask:
                #     nodules have values greater or equal to 1, depending on texture
                #     lungs have value 0.5
                #     rest of structures have value 0
                if self.mask_mode == "nodule":
                    # preserve only the nodule, remove nodule texture and lung information
                    mask_arr = (mask_arr >= 1)*1
                elif self.mask_mode == "nodule+lung":
                    # remove nodule texture information
                    mask_arr[mask_arr >= 1] = 1
                else:
                    assert self.mask_mode == "nodule+lung+texture"
                    mask_arr = mask_arr/5.0 # normalize mask between 0 and 1

                #mask_arr = fit_nparray_to_given_size(mask_arr, size=256)
                #mask_arr = regularize_components_minimal(mask_arr)
                assert mask_arr.shape == arr.shape
                mask_arr = torch.from_numpy(mask_arr).unsqueeze(0) # shape (1, 256, 256, 256)
                mask_arr = mask_arr.to(torch.float32)
                mask_input = mask_arr
                #print(f"mask in put size is {mask_input.shape}")

            elif self.mask_mode != "none":
                
                mask_path = image_path.replace("/chest_ct/", "/mask/")
                assert os.path.exists(mask_path), "Error: mask path was not found!"
                mask_arr = np.load(mask_path)
                mask_arr = mask_arr.copy()
                # conditional mask:
                #     nodules have values greater or equal to 1, depending on texture
                #     lungs have value 0.5
                #     rest of structures have value 0
                if self.mask_mode == "nodule":
                    mask_arr = (mask_arr >= 1).astype(np.int64)  # 0=background, 1=nodule
                elif self.mask_mode == "nodule+lung":
                    mask_arr[mask_arr >= 1] = 1  # nodules
                    mask_arr[mask_arr == 0.5] = 2  # lungs

                else:
                    assert self.mask_mode == "nodule+lung+texture"
                    mask_arr[mask_arr == 0.5] = 6  # lungs
                    mask_arr = np.round(mask_arr).astype(int)

                mask_arr = mask_arr.astype(np.int64)
                mask_index = torch.from_numpy(mask_arr).long()  # [B,D,H,W]
                #print(f"mask index unique values: {torch.unique(mask_index)}")

                mask_input = F.one_hot(mask_index, num_classes=self.num_classes).permute(3,0,1,2).float()  # [B,C,D,H,W]
                
        else:
            nib_img = nibabel.load(image_path)
            arr = nib_img.get_fdata()
        image = torch.tensor(arr, dtype=torch.float32)
        image = image.unsqueeze(0)
        image_size = image.shape[-1]
        #image.shape = (1, 256, 256, 256)
        first_coords = [0, 0, 0]

        # normalization
        image = self.normalize(image)

        # normalized coordinates
        if self.concat_coords:
            dim = len(image.shape) - 1  # 2d or 3d
            self.coord_cache = torch.stack(torch.meshgrid(dim * [torch.linspace(-1, 1, image_size)], indexing='ij'), dim=0)
            image = torch.cat([image, self.coord_cache], dim=0)
            #image.shape = (4, 256, 256, 256)

        # half crop: crop to a 128x128[x128] image,
        if self.patch_based:
            patch_size = image_size // 2
            shape = (len(image.shape)-1,)

            original_sampling = False

            if original_sampling:
                ### OLD STUFF
                #first_coords = np.random.randint(0, 32+1, shape) + np.random.randint(0, 64+32+1, shape)
                #print(first_coords)
                first_coords = np.random.randint(0, image_size-patch_size+1, shape) # we will generate 64x64x64 crops
            else:
                anchors = np.meshgrid(np.arange(0, image_size, patch_size),
                          np.arange(0, image_size, patch_size),
                          np.arange(0, image_size, patch_size))
                anchors = np.vstack([anchors[0].ravel(), anchors[1].ravel(), anchors[2].ravel()]).T 
                # anchors.shape is (N, 3) array where N is the number of anchors and 3 is the 3-valued coordinate vector within the 3D structure
                # each anchor is the top corner of a cubic patch
                #print(anchors)
                n_anchors = len(anchors)
                anchor_idx = np.random.randint(0, n_anchors)
                std = patch_size//8 # std could be patch_size//8
                indices = np.random.normal(0, std, size=3) + anchors[anchor_idx]

                indices[indices < 0 ] = 0
                indices[indices >= image_size - patch_size] = image_size - patch_size
                first_coords = indices.astype(int)

            index = tuple([slice(None), *(slice(f, f+patch_size) for f in first_coords)])
            image = image[index]

        elif self.mask_mode != "none":
            return {"image": image, "y": 1, "offset": first_coords, "mask": mask_input,  "filename": image_path}

        else:
            return {"image": image, "y": 1, "offset": first_coords,  "filename": image_path}

    def __len__(self):
        return self.n_images


class CondLatentDiffusionPipeline_LIDC3D(LatentDiffusionPipelineBase):
    def __init__(
            self,
            scheduler: Union[
                DDIMScheduler,
                DDPMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,
            ],
            unet: UNetModel,
            vae: Optional[Union[AutoencoderKlReducedMaisi]] = None,
            maskEncoder: Optional[AutoencoderKlReducedMaisi] = None,
            latent_channels: Optional[int] = 16,
            patchbased: Optional[bool] = False,
            mask_mode: Optional[str] = "none",
            mask_dataset: Optional[str] = None,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            maskEncoder=maskEncoder,
        )
        
        self.vae_scale_factor = 1
        self.use_vae = True if vae is not None else False
        self.use_maskEncoder = True if maskEncoder is not None else False
        self.patchbased = patchbased
        self.latent_channels = latent_channels
        self.mask_mode = mask_mode
        self.mask_dataset = mask_dataset
        

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,  # default to generate a single image
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            renormalize: bool = False,
            return_latents: bool = False,
            start_indx: Optional[int] = None,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        device, dtype = self.unet.device, self.unet.dtype

        deltaD, deltaH, deltaW = 4, 4, 4
        if self.use_vae:
            latents = self.prepare_latents(batch_size, self.latent_channels, height//deltaH, width//deltaW,
                                        dtype, device, generator, latents, depth=height//deltaD)
        else:
            latents = self.prepare_latents(batch_size, 1, height, width,
                                        dtype, device, generator, latents, depth=height)
        #print(f"num inference steps is {num_inference_steps}")
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # init scheduler
        """timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device)
        _ = self.scheduler.add_noise(latents, noise, timesteps)"""
        
        # latents.shape = (batch_size, 3, latent_height, latent_width)

        debug = False

        if debug:
            print("before the loop", sum(latents[0].flatten() - latents[1].flatten()))
            #print("before the loop", sum(mask_latents[0].flatten() - mask_latents[1].flatten()))


        coord_latents = torch.stack(torch.meshgrid(3 * [torch.linspace(-1, 1, height)], indexing='ij'), dim=0)
        coord_latents = coord_latents.repeat(batch_size, 1, 1, 1, 1).to(device)
        if self.use_vae:
            coord_latents = coord_latents[:, :, ::deltaD, ::deltaH, ::deltaW]


        # Get masks for inference if needed
        if self.mask_mode != "none":
            assert os.path.exists(self.mask_dataset)
            #print(f"before reading N masks, useMaskEncoder is {self.use_maskEncoder}")
            input_masks, texture_scores = read_N_masks_for_inference(batch_size, self.mask_dataset, self.mask_mode, self.use_maskEncoder, start_indx)
            input_masks_tensor = torch.from_numpy(input_masks).to(latents.device).to(latents.dtype)
            mask_latents = mask_downsample(input_masks_tensor, self.maskEncoder)


        # run denoising iterative process
        for t in self.progress_bar(self.scheduler.timesteps):

            timesteps = t.repeat((batch_size,)).to(latents.device).long()

            latents = self.scheduler.scale_model_input(latents, t)

            noisy_latents = latents # this would be the unconditional scenario
            if self.patchbased:
                noisy_latents = torch.cat((noisy_latents, coord_latents), 1)
                condition_channels = coord_latents.shape[1]
                cond_latents = coord_latents.reshape(batch_size, -1, condition_channels)

            """
            noise_pred = self.unet(
                cond_latents,
                t,
            ).sample
            """

            #cond_latents = mask_latents.view(batch_size, -1)
            #if self.nodule_attributes:
            #    cond_latents = torch.cat([cond_latents, emb_vec], 1)
            #    #noisy_latents = merge_input_with_class_cond_embedding(noisy_latents, emb_vec)

            if self.mask_mode != "none":
                noisy_latents = torch.cat((noisy_latents, mask_latents), 1)
                condition_channels = mask_latents.shape[1]
                cond_latents = mask_latents.reshape(batch_size, -1, condition_channels)

            #predict epsiolon, v or xo (depending on scheduler.prediction_type)
            if self.unet.cross_attention_dim is None:
                pred = self.unet(x=noisy_latents, timesteps=timesteps)
            else:
                pred = self.unet(x=noisy_latents, timesteps=timesteps, context=cond_latents)

            if debug:
                print("after 1", sum(latents[0].flatten() - latents[1].flatten()))
                print("after 1", sum(mask_latents[0].flatten() - mask_latents[1].flatten()))
                print("pred", sum(pred[0].flatten() - pred[1].flatten()))

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        #images = latents.cpu().permute(0, 2, 3, 4, 1).float().numpy()
        if self.use_vae:
            with torch.amp.autocast("cuda", enabled=True):
                images = self.vae.decode(latents)
            images = images.repeat((1, 3, 1, 1, 1))
        else:
            images = latents

        if renormalize:
            images = images / 2 + 0.5

        images = images.cpu().float().numpy()
        images = np.clip(images, 0, 1)

        """
        images = self.decode_latents(latents)
        input_masks = masks.permute(0, 2, 3, 1).detach().cpu().numpy()
        if self.encode_mask:
            output_masks = self.decode_latents(mask_latents)
        else:
            mask_t = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST)
            output_masks = torch.cat([mask_t(m)[0].unsqueeze(0).unsqueeze(0) for m in mask_latents])
            output_masks = output_masks.permute(0, 2, 3, 1).detach().cpu().numpy()
        """
        #input_masks *= 255
        #output_masks *=255

        if output_type == "pil":
            images = self.numpy_to_pil(images)

        if not return_dict:
            #return (images, input_masks, output_masks, nodule_features)

            if return_latents:
                return images, latents
            if self.mask_mode != "none":
                output_masks = mask_latents.tile((1, 1, 4, 4, 4)).cpu().numpy()
                if "texture" in self.mask_mode: 
                    return images, input_masks, output_masks, texture_scores
                else:
                    return images, input_masks, output_masks
            return images

        return ImagePipelineOutput(images=images)

class RandomMaskGenerator3D():


    def __init__(self, nodule_masks_sum_npy_path, nodule_sizes_npy_path, verbose=False):

        assert os.path.exists(nodule_masks_sum_npy_path)
        if verbose:
            print("building random mask generator...")
        
        global_mask = np.load(nodule_masks_sum_npy_path)
        continuous_mask = global_mask.copy()/global_mask.max() # normalize probs
        self.prob_map = gaussian_filter(continuous_mask, sigma=5)

        x = np.sort(np.load(nodule_sizes_npy_path))
        self.n_points_spline = len(x)
        x_data = np.arange(self.n_points_spline)
        y_data = x
        self.size_spline = UnivariateSpline(x_data, y_data, s=1)

        if verbose:
            plt.plot(x)
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, self.size_spline(x_data), color='red', label='Fitted Exponential')
            plt.legend()
            plt.title('Exponential Fit')
            plt.show()

    def borders_are_zero(self, arr):
        if arr.ndim != 3:
            raise ValueError("Input must be a 3D array.")
    
        # Check all six faces of the 3D array
        return (
            np.all(arr[0, :, :] == 0) and     # front face
            np.all(arr[-1, :, :] == 0) and    # back face
            np.all(arr[:, 0, :] == 0) and     # top face
            np.all(arr[:, -1, :] == 0) and    # bottom face
            np.all(arr[:, :, 0] == 0) and     # left face
            np.all(arr[:, :, -1] == 0)        # right face
        )
    
    def generate_random_mask(self,):

        # sample nodule centroid
        prob_flat = self.prob_map.flatten()
        prob_flat /= prob_flat.sum()
        index_flat = np.random.choice(len(prob_flat), p=prob_flat)
        z, y, x = np.unravel_index(index_flat, self.prob_map.shape)
        position = (x, y, z)

        # sample nodule radius
        nodule_size = self.size_spline(np.random.uniform(1.0, self.n_points_spline))
        nodule_size = max(nodule_size, 64) # minimum size is 64 (4**3)
        D = int(np.cbrt(nodule_size)) # retrieve the radius of the cube (half of one size)
        pad = D % 2
        R = D // 2
        
        # setup the mask
        xlen, ylen, zlen = self.prob_map.shape
        generated_mask = np.zeros((xlen, ylen, zlen))
        x_start, y_start, z_start = max(0, x-R), max(0, y-R), max(0, z-R)
        x_end, y_end, z_end = min(xlen, x+R), min(ylen, y+R), min(zlen, z+R)
        generated_mask[x_start:x_end+pad, y_start:y_end+pad, z_start:z_end+pad] = 1

        # nodule cannot be at the border, build a new mask until this does not happen
        if not self.borders_are_zero(generated_mask):
            generated_mask, position, nodule_size = self.generate_random_mask()
        
        return generated_mask, position, nodule_size


def save_vol_gifs(data, out_dir):
    assert len(data.shape) == 5
    data = data.mean(1)[:, np.newaxis, :, :, :]
    batch_size, channels, depth, height, width = data.shape
    assert channels == 1
    os.makedirs(out_dir, exist_ok=True)
    for sample_idx in range(batch_size):
        data_x = [data[sample_idx, 0, i, :, :] for i in range(depth)]
        out_path = os.path.join(out_dir, "x.gif")
        imageio.mimsave(out_path, data_x, duration=0.1)
        data_y = [data[sample_idx, 0, :, i, :] for i in range(depth)]
        out_path = os.path.join(out_dir, "y.gif")
        imageio.mimsave(out_path, data_y, duration=0.1)
        data_z = [data[sample_idx, 0, :, :, i] for i in range(depth)]
        out_path = os.path.join(out_dir, "z.gif")
        imageio.mimsave(out_path, data_z, duration=0.1)


def load_pipeline(model_path, verbose=True):

    if verbose:
        print("Loading Diffusion pipeline from:")
        print(f"    - {model_path}\n")
    unet = UNetModel.from_pretrained(model_path, subfolder=f"unet")
    # Count all parameters
    unet.cuda()
    if verbose:
        print("U-Net model loaded")

    scheduler_config_path = model_path + "/scheduler/scheduler_config.json" 
    noise_scheduler = DDPMScheduler.from_config(scheduler_config_path)
    
    pipeline = CondLatentDiffusionPipeline_LIDC3D(
        unet=unet,
        scheduler=noise_scheduler,
    )
    print("Diffusion pipeline is ready\n")

    return pipeline


def regularize_components_minimal(binary_mask):
    """
    Converts each connected component in a 3D binary mask into the smallest
    odd-sized cube that contains it, and returns a new binary mask with those cubes.

    Parameters:
        binary_mask (np.ndarray): 3D binary mask.

    Returns:
        np.ndarray: New binary mask with regular cubes containing each component.
    """
    assert binary_mask.ndim == 3, "Input mask must be 3D"

    output_mask = np.zeros_like(binary_mask)
    labeled_mask, num_features = label(binary_mask, structure=np.ones((3,3,3)))

    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        bbox = find_objects(component)[0]  # tuple of slices per axis

        # Compute the bounding box extents
        min_coords = [s.start for s in bbox]
        max_coords = [s.stop for s in bbox]
        sizes = [stop - start for start, stop in zip(min_coords, max_coords)]

        # Determine minimal odd-sized cube side length N
        max_extent = max(sizes)
        N = max_extent if max_extent % 2 == 1 else max_extent + 1
        half_N = N // 2

        # Compute cube center
        center = [(start + stop) // 2 for start, stop in zip(min_coords, max_coords)]

        # Compute cube bounds, clipped to volume
        bounds = []
        for c, dim in zip(center, binary_mask.shape):
            start = max(0, c - half_N)
            end = min(dim, c + half_N + 1)
            # Adjust bounds to make cube size exactly N if clipped
            actual_size = end - start
            if actual_size < N:
                if start == 0:
                    end = min(dim, N)
                elif end == dim:
                    start = max(0, dim - N)
            bounds.append((start, end))

        z0, z1 = bounds[0]
        y0, y1 = bounds[1]
        x0, x1 = bounds[2]

        output_mask[z0:z1, y0:y1, x0:x1] = 1

    return output_mask


def extract_3d_contours(binary_mask, width=1):
    """
    Extracts the outer contour voxels of 3D connected components with specified thickness.

    Parameters:
        binary_mask (np.ndarray): 3D binary mask.
        width (int): Contour width in number of voxels (>=1).

    Returns:
        np.ndarray: Binary mask with contour voxels of specified width.
    """
    assert binary_mask.ndim == 3, "Input must be a 3D binary mask"
    assert width >= 1, "Contour width must be >= 1"

    current_mask = binary_mask.astype(bool)
    inner_mask = current_mask.copy()

    for _ in range(width):
        padded = np.pad(inner_mask, 1, mode='constant', constant_values=0).astype(bool)

        # Extract 6-connected neighbors
        xm = padded[1:-1, 1:-1, :-2]
        xp = padded[1:-1, 1:-1, 2:]
        ym = padded[1:-1, :-2, 1:-1]
        yp = padded[1:-1, 2:, 1:-1]
        zm = padded[:-2, 1:-1, 1:-1]
        zp = padded[2:, 1:-1, 1:-1]

        # Keep only voxels fully surrounded by neighbors
        surrounded = xm & xp & ym & yp & zm & zp
        inner_mask = inner_mask & surrounded

    contour_mask = current_mask & (~inner_mask)
    return contour_mask.astype(np.uint8)

def merge_images_with_masks(images, masks):
    #images and masks are expected to be np arrays with shape (B, channels, D, H, W)
    super_images = images.copy()
    if super_images.shape[1] == 1:
        super_images = np.tile(super_images, (1,3,1,1,1))
    super_mask = np.zeros_like(super_images).astype(bool)
    for idx in range(images.shape[0]):
        #mask = masks[idx].permute(1, 2, 0).detach().cpu().numpy().astype(bool)
        mask_ = masks[idx].astype(bool)
        print(f"mask shape before processing: {mask_.shape}")
        mask_ = regularize_components_minimal(mask_[0])
        mask = extract_3d_contours(mask_, width=2)
        mask = mask[np.newaxis, ...]
        assert len(mask.shape) == 4
        if mask.shape[0] == 1:
            mask = np.concatenate([mask, mask, mask], axis=0) 
        mask[1:, :, :, :] = False
        super_mask[idx] = mask
    #super_images[super_mask] += 0.5
    super_images = np.clip(super_images, 0, 1)
    return super_images

def read_N_masks_for_inference(N, dataset_dir, mask_mode, useMaskEncoder, start_indx=None):


    inference_dataset = LIDCVolumes(dataset_dir,
                                    mask_mode=mask_mode, useMaskEncoder=False, masks_only=True) # we always load the masks without one-hot encoding, even if useMaskEncoder=True because here we set the texture values for inference

    nodule_masks = []
    nodule_textures = []

    if mask_mode == "nodule":
        num_classes = 2  # background, nodule
    elif mask_mode == "nodule+lung":
        num_classes = 3  # background, nodule, lung
    elif mask_mode == "nodule+lung+texture":
        num_classes = 7  # background, nodule texture 1-5, lung
    else:
        raise ValueError("Invalid mask_mode")

    if start_indx == None:
        for i in range(N):

            mask_orig = inference_dataset[i]["mask"]
            
            if useMaskEncoder:
                
                nodule_texture = 0

                if mask_mode == "nodule":
                    mask_orig = (mask_orig >= 1).astype(np.int64)  # 0=background, 1=nodule
                elif mask_mode == "nodule+lung":
                    
                    mask_orig[mask_orig >= 1] = 1  # nodules
                    mask_orig[mask_orig == 0.5] = 2 # lungs
                   
                else:  # "nodule+lung+texture"
                    #LIDC_PROBS TEXTURE
                    mask_orig = mask_orig * 5.0
                    mask_orig = mask_orig * 5.0
                    probs = np.array([113, 93, 115, 379, 857])
                    normalized_probs = probs/sum(probs)
                    nodule_texture = np.random.choice(np.arange(1, 6), 1, p=normalized_probs)[0]
                    #nodule_texture = np.random.randint(1, 6)
                    #nodule_texture = np.random.choice([1, 3, 5])
                    mask_orig[mask_orig >= 1] = nodule_texture
                    mask_orig[mask_orig == 0.5] = 6  # lungs
               
                mask_orig = mask_orig.long()
                mask_input = F.one_hot(mask_orig.squeeze(0), num_classes=num_classes).permute(3,0,1,2).float()  
                nodule_textures.append(nodule_texture)
                nodule_masks.append(mask_input)

            else:

                nodule_texture = 0
                if mask_mode == "nodule+lung+texture":
                    
                    #LIDC_PROBS TEXTURE
                    mask_orig = mask_orig * 5.0
                    probs = np.array([113, 93, 115, 379, 857])
                    normalized_probs = probs/sum(probs)
                    nodule_texture = np.random.choice(np.arange(1, 6), 1, p=normalized_probs)[0]
                    #nodule_texture = np.random.randint(1, 6) 
                    #nodule_texture = np.random.choice([1, 3, 5])
                    mask_orig[mask_orig >= 1] = nodule_texture
                    # assign a random nodule texture on a scale of 1 (non-solid) to 5 (solid)
                    #mask_orig[mask_orig >= 1] = np.random.randint(1, 6)
                    mask_orig = mask_orig / 5.0 # normalize again
                nodule_masks.append(mask_orig)
                nodule_textures.append(nodule_texture)
            
    else:
        
        dataset_len = len(inference_dataset)
        for j in range(start_indx, min(start_indx + N, dataset_len)):
            mask_orig = inference_dataset[j]["mask"]
            
            nodule_texture = 0

            if useMaskEncoder:
                if mask_mode == "nodule":
                    mask_orig = (mask_orig >= 1).astype(np.int64)  # 0=background, 1=nodule
                elif mask_mode == "nodule+lung":
                    
                    mask_orig[mask_orig >= 1] = 1  # nodules
                    mask_orig[mask_orig == 0.5] = 2  # lungs
                
                else: 
                    #LIDC_PROBS TEXTURE
                    mask_orig = mask_orig * 5.0
                    mask_orig = mask_orig * 5.0
                    probs = np.array([113, 93, 115, 379, 857])
                    normalized_probs = probs/sum(probs)
                    nodule_texture = np.random.choice(np.arange(1, 6), 1, p=normalized_probs)[0]
                    #nodule_texture = np.random.randint(1, 6)
                    #nodule_texture = np.random.choice([1, 3, 5])
                    mask_orig[mask_orig >= 1] = nodule_texture
                    mask_orig[mask_orig == 0.5] = 6  # lungs
                   
                mask_orig = mask_orig.long()
                mask_input = F.one_hot(mask_orig.squeeze(0), num_classes=num_classes).permute(3,0,1,2).float()  # [C,D,H,W]
                nodule_textures.append(nodule_texture)
                nodule_masks.append(mask_input)

            else:
                if mask_mode == "nodule+lung+texture":
                    #LIDC_PROBS TEXTURE
                    mask_orig = mask_orig * 5.0
                    probs = np.array([113, 93, 115, 379, 857])
                    normalized_probs = probs/sum(probs)
                    nodule_texture = np.random.choice(np.arange(1, 6), 1, p=normalized_probs)[0]
                    #nodule_texture = np.random.randint(1, 6)
                    #nodule_texture = np.random.choice([1, 3, 5])
                    mask_orig[mask_orig >= 1] = nodule_texture
                    mask_orig = mask_orig / 5.0  # normalize again
                nodule_masks.append(mask_orig)
                nodule_textures.append(nodule_texture)
    
    input_masks = np.stack(nodule_masks)
    input_textures = np.stack(nodule_textures)
    return input_masks, input_textures

def squeeze_except_batch(tensor):
    # tensor[0] selects the first item in the batch, revealing the rest of the shape
    squeezed = tensor
    for dim in reversed(range(1, tensor.dim())):  # start from dim 1 (exclude batch)
        if squeezed.size(dim) == 1:
            squeezed = squeezed.squeeze(dim)
    return squeezed