import os
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
import glob


class LIDCMasks(torch.utils.data.Dataset):
    def __init__(self, directory, mask_mode="none", num_classes=7, 
                 use_onehot=False, split="train", val_ratio=0.1, seed=42, sdf_flag=False, original_textures=False, sdf_truncation=20, spacing=None):
        """
        directory: expected to contain masks as .npy
        mask_mode: controls preprocessing of the mask
        num_classes: total number of segmentation classes
        use_onehot: if True -> return one-hot mask [C,D,H,W], else single-channel [1,D,H,W]
        split: "train" or "val"
        val_ratio: fraction of data used for validation
        seed: random seed for reproducible split
        mask_part: "nodule", "lung", or "all" 
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.mask_mode = mask_mode
        self.num_classes = num_classes
        self.use_onehot = use_onehot
        self.sdf_flag = sdf_flag
        self.sdf_truncation = sdf_truncation
        self.spacing = spacing
        self.original_textures = original_textures

        # collect all mask paths
        all_paths = sorted(glob.glob(self.directory + "/**/mask/*.npy", recursive=True))
        self.n_images = len(all_paths)
        print("Number of matched masks:", len(all_paths))
        if self.n_images == 0:
            raise RuntimeError(f"No masks found in {self.directory}")

        # reproducible train/val split
        np.random.seed(seed)
        indices = np.arange(self.n_images)
        np.random.shuffle(indices)
        val_count = round(self.n_images * val_ratio)
        print(f"Using {self.n_images - val_count} images for training, {val_count} for validation")

        if split == "train":
            selected_indices = indices[val_count:]
        elif split == "val":
            selected_indices = indices[:val_count]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_paths = [all_paths[i] for i in selected_indices]
        self.n_images = len(self.image_paths)


    def compute_sdf(self, mask, truncation=20, spacing=None):
        dt_out = distance_transform_edt(mask == 0, sampling=spacing)
        dt_in  = distance_transform_edt(mask == 1, sampling=spacing)
        sdf = dt_out - dt_in
        sdf = np.clip(sdf, -truncation, truncation) / truncation
        return sdf.astype(np.float32)

    def get_foreground_classes(self, mask_mode):
        """Defines which labels to compute SDF for depending on mask_mode."""
        if mask_mode == "nodule":
            return [1]  # only nodule
        elif mask_mode == "lung":
            return [1]  # lung only (after remap)
        elif mask_mode == "nodule+lung":
            return [1, 2]
        elif mask_mode == "nodule+lung+texture":
            return [1, 2, 3, 4, 5, 6]  # textures + lung
        else:
            return []
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Mask path {image_path} not found!")

        mask_arr = np.load(image_path)  # shape [D,H,W]

        # remap depending on mask_mode
        if self.mask_mode != "none":
            mask_arr = mask_arr.copy()
            
            if self.mask_mode == "nodule":
                mask_arr = (mask_arr >= 1).astype(np.int64)  # 0=background, 1=nodule
            elif self.mask_mode == "nodule+lung":
                mask_arr[mask_arr >= 1] = 1  # nodules
                mask_arr[mask_arr == 0.5] = 2  # lungs
            elif self.mask_mode == "lung":
                mask_arr[mask_arr > 0.5] = 0 # background
                mask_arr[mask_arr < 0.5] = 0  # background
                mask_arr[mask_arr == 0.5] = 1  # lungs
            elif self.mask_mode=="nodule+lung+texture" and not self.original_textures:  # 
                #reassign nodule texture so it is balanced (check current value 1-5 and change it for a random between 1-5)
                # Convert lung values (0.5) to 6
                mask_arr[mask_arr == 0.5] = 6
                # Round the array to nearest integer and convert to int
                mask_arr = np.round(mask_arr).astype(int)
                # Find all unique nodule labels (1–5, or however many nodules you have)
                nodule_labels = np.unique(mask_arr)
                nodule_labels = nodule_labels[(nodule_labels >= 1) & (nodule_labels <= 5)]
                # Reassign each nodule label to a new random texture between 1–5
                for label in nodule_labels:
                    new_texture = np.random.randint(1, 6)
                    mask_arr[mask_arr == label] = new_texture
                    
            elif self.mask_mode=="nodule+lung+texture" and self.original_textures:  
                mask_arr[mask_arr == 0.5] = 6

            mask_arr = mask_arr.astype(np.int64)

        out_dict = {"filename": image_path}

        # === SDF generation ===
        if self.sdf_flag:
            class_list = self.get_foreground_classes(self.mask_mode)
            sdf_channels = []
            for cls_id in class_list:
                binary = (mask_arr == cls_id).astype(np.uint8)
                sdf_map = self.compute_sdf(binary, truncation=self.sdf_truncation, spacing=self.spacing)
                sdf_channels.append(sdf_map[None])  # [1,D,H,W]
            if len(sdf_channels) > 0:
                sdf_tensor = torch.from_numpy(np.concatenate(sdf_channels, axis=0))  # [C,D,H,W]
            else:
                # no foreground — return empty channel or zeros
                sdf_tensor = torch.zeros((1,) + mask_arr.shape, dtype=torch.float32)
            out_dict["mask_sdf"] = sdf_tensor

        # === Original mask outputs ===
        #if not self.sdf_flag :
        mask_index = torch.from_numpy(mask_arr).long()
        if self.use_onehot:
            # One-hot encoding for model input/output (high memory)
            mask_input = F.one_hot(mask_index, num_classes=self.num_classes).permute(3,0,1,2).float()  # [C,D,H,W]
        else:
            # Single channel input (low memory)
            mask_input = mask_index.unsqueeze(0).float()  # [1,D,H,W]
        out_dict["mask_input"] = mask_input
        out_dict["mask_index"] = mask_index

        return out_dict
    
    def __len__(self):
        return self.n_images



def generalized_dice_loss_ignore_background(probs, targets, epsilon=1e-6, ignore_index=0): 
    """
    probs: softmax probabilities, shape (B, C, D, H, W)
    targets: integer labels, shape (B, D, H, W)
    """
    num_classes = probs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0,4,1,2,3).float()

    dims = (0,2,3,4)

    # Exclude the background class
    valid_classes = [i for i in range(num_classes) if i != ignore_index]
    one_hot = one_hot[:, valid_classes, ...]
    probs = probs[:, valid_classes, ...]
    
    # Intersection and union
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)

    # Class weights (inverse squared of ground-truth volume)
    gt_sum = one_hot.sum(dims)
    weights = 1.0 / (gt_sum**2 + epsilon)

    numerator = 2 * (weights * intersection).sum()
    denominator = (weights * union).sum() + epsilon

    dice = numerator / denominator

    return 1 - dice



def generalized_dice_loss_from_logits(logits, targets, num_classes, epsilon=1e-6):
    """
    logits: (B, C, D, H, W)
    targets: (B, D, H, W) integer labels
    """
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3, 4)

    numerator = 0.0
    denominator = 0.0
    for c in range(num_classes):
        probs_c = probs[:, c, ...]
        target_c = (targets == c).float()
        intersection = (probs_c * target_c).sum(dims)
        union = probs_c.sum(dims) + target_c.sum(dims)

        # class weight (inverse squared volume)
        w_c = 1.0 / (target_c.sum(dims) ** 2 + epsilon)

        numerator += w_c * (2 * intersection)
        denominator += w_c * union

    dice = (numerator + epsilon) / (denominator + epsilon)
    return 1 - dice


def generalized_dice_loss(probs, targets, epsilon=1e-6):
    """
    probs: softmax probabilities, shape (B, C, D, H, W)
    targets: integer labels, shape (B, D, H, W)
    """
    num_classes = probs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)

    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)

    gt_sum = one_hot.sum(dims)
    weights = 1.0 / (gt_sum**2 + epsilon)

    # Mask out absent classes
    present_mask = gt_sum > 0
    weights = weights * present_mask

    numerator = 2 * (weights * intersection).sum()
    denominator = (weights * union).sum() + epsilon

    dice = numerator / denominator

    return 1 - dice

def vae_loss_segmentation(logits, targets, mu, sigma, num_classes, mask_part, beta=1e-7):

    # --- dice loss (generalized) ---
    probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1-1e-7)
    
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 to apply class weighting")
    if mask_part == "lung" and num_classes != 2:
        raise ValueError("For mask_part='lung', num_classes must be 2")
    if num_classes==2 and mask_part == "lung":
        class_weights = torch.tensor([0.5, 0.5], device=logits.device)  # Example weights for binary case
    elif mask_part == "nodule" and num_classes == 2:
        class_weights = torch.tensor([0.5, 10.0], device=logits.device)  # Example weights for binary case
    # Define class weights to handle class imbalance
    elif num_classes == 3:
        # Example: [background, nodule, lung] weights
        class_weights = torch.tensor([0.5, 10.0, 1.0], device=logits.device)
    elif num_classes == 7:
        # Example: [background, nodule texture 1, nodule texture 2, nodule texture 3, nodule texture 4, nodule texture 5, lung] weights
        class_weights = torch.tensor([0.5, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0], device=logits.device)
    else:
        raise ValueError(f"Class weights not defined for num_classes={num_classes}")
    
    # --- cross entropy ---
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    #ce = F.cross_entropy(logits, targets)

    dice = generalized_dice_loss(probs, targets)
    #dice = generalized_dice_loss_from_logits(logits, targets, num_classes)

    # --- reconstruction ---
    recon_loss = ce + dice

    # --- KL divergence ---
    # sigma = torch.clamp(sigma, min=1e-6)
    # kl = -0.5 * torch.mean(1 + 2*torch.log(sigma) - mu.pow(2) - sigma.pow(2))

    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(sigma.shape))),
    )
    kl= torch.sum(kl_loss) / kl_loss.shape[0]

    return recon_loss + beta * kl, recon_loss, kl, ce, dice

import torch
import torch.nn.functional as F

def per_class_dice(inputs, targets, epsilon=1e-6, is_logits=True):
    """
    Computes Dice score per class.
    If is_logits=True, applies argmax to get hard predictions.
    Absent classes in GT are set to NaN.
    """
    if is_logits:
        pred = torch.argmax(inputs, dim=1)  # (B, D, H, W)
        inputs = F.one_hot(pred, num_classes=inputs.shape[1]).permute(0, 4, 1, 2, 3).float()
    else:
        num_classes = inputs.max().item() + 1
        inputs = F.one_hot(inputs, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    num_classes = inputs.shape[1]
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (inputs * one_hot).sum(dims)
    union = inputs.sum(dims) + one_hot.sum(dims)
    dice_per_class = (2 * intersection + epsilon) / (union + epsilon)

    # Identify classes absent in GT
    gt_sum = one_hot.sum(dims)
    absent_mask = gt_sum == 0

    # Assign NaN to absent classes
    dice_per_class = dice_per_class.masked_fill(absent_mask, float('nan'))

    print("Dice per class:", dice_per_class)

    return dice_per_class

def sdf_regression_loss_weighted(pred_sdf, target_sdf, weights=None):
    """
    pred_sdf, target_sdf: (B, C, D, H, W)
    weights: (C,) tensor or None
    """
    l1_per_channel = torch.mean(torch.abs(pred_sdf - target_sdf), dim=(0,2,3,4))  # mean over batch+voxels
    if weights is not None:
        weights = weights / (weights.sum() + 1e-8)
        loss = (l1_per_channel * weights).sum()
    else:
        loss = l1_per_channel.mean()
    return loss

def eikonal_loss(sdf, class_weights=None):
    """
    Computes the eikonal loss for multi-class SDFs in 3D.
    
    Args:
        sdf: torch.Tensor of shape [B, C, D, H, W], predicted SDF
        class_weights: optional torch.Tensor of shape [C] to weight each channel (Normally not used)
        
    Returns:
        eik_loss: scalar tensor
    """
    # Compute gradients along spatial dimensions
    # torch.gradient returns a tuple of gradients along each dimension
    dx, dy, dz = torch.gradient(sdf, dim=(2,3,4))  # shape [B, C, D, H, W]

    # Gradient magnitude
    grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)  # [B, C, D, H, W]

    # Squared deviation from 1
    loss_per_voxel = (grad_norm - 1)**2  # [B, C, D, H, W]

    # Apply optional class weights
    if class_weights is not None:
        # reshape to broadcast: [1, C, 1, 1, 1]
        weights = class_weights.view(1, -1, 1, 1, 1)
        loss_per_voxel = loss_per_voxel * weights

    # Final eikonal loss: mean over batch, channels, and spatial dims
    eik_loss = loss_per_voxel.mean()
    return eik_loss


def vae_loss_sdf(pred_sdf, target_sdf, mu, sigma, beta=1e-7, eikonal_weight=0.1):
    """
    pred_sdf: (B,C,D,H,W)
    target_sdf: (B,C,D,H,W)
    """
    # Reconstruction term
    class_weights = torch.tensor([10, 1], device=pred_sdf.device)
    sdf_l1 = sdf_regression_loss_weighted(pred_sdf, target_sdf, class_weights)

    # Optional: add Eikonal regularization
    eik_loss = eikonal_loss(pred_sdf) * eikonal_weight

    # KL divergence
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(sigma.shape))),
    )
    kl = torch.sum(kl_loss) / kl_loss.shape[0]

    total = sdf_l1 + eik_loss + beta * kl
    return total, sdf_l1, eik_loss, kl