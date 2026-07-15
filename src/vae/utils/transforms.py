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

# slightly modified from the MONAI project since the intensity transforms used in MAISI ended up with image values not being bounded

# NOTE: this pipeline expects pre-normalized `.npy` volumes (as produced by preproc_lidc_npy.py /
# preproc_nlst_npy.py): already windowed to [0, 1] and axis-ordered. There is no NIfTI/raw-HU support
# here, so no orientation step (no affine metadata available on `.npy`) and no HU intensity windowing.

import warnings
from typing import List, Optional

import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    SelectItemsd,
    SpatialPadd,
    CenterSpatialCropd,
)


def define_vae_transform(
    is_train: bool,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    val_crop_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> tuple:
    """
    Define the MAISI VAE transform pipeline for training or validation, for pre-normalized `.npy` input.

    Args:
        is_train (bool): Whether it's for training or not. If True, the output transform will consider random_aug, the cropping will use "patch_size" for random cropping. If False, the output transform will alwasy treat "random_aug" as False, will use "val_patch_size" for central cropping.
        random_aug (bool): Whether to apply random data augmentation.
        k (int, optional): Patches should be divisible by k. Defaults to 4.
        patch_size (List[int], optional): Size of the patches. Defaults to [128, 128, 128]. Will random crop patch for training.
        val_patch_size (Optional[List[int]], optional): Size of validation patches. Defaults to None. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
        output_dtype (torch.dtype, optional): Output data type. Defaults to torch.float32.
        spacing_type (str, optional): Type of spacing. Defaults to "original". Choose from ["original", "rand_zoom"].
            ("fixed" is not supported: it requires physical-spacing/affine metadata that `.npy` volumes don't carry.)
        spacing (Optional[List[float]], optional): Unused (kept for signature compatibility). Defaults to None.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].
        label_keys (List[str], optional): List of label keys. Defaults to [].
        additional_keys (List[str], optional): List of additional keys. Defaults to [].
        select_channel (int, optional): Channel to select for multi-channel MRI. Defaults to 0.

    Returns:
        tuple: A tuple containing Composed Transform train_transforms or val_transforms depending on 'is_train'.
    """

    if spacing_type not in ["original", "rand_zoom"]:
        raise ValueError(
            f"spacing_type has to be chosen from ['original', 'rand_zoom']. Got {spacing_type}. "
            "('fixed' is not supported for .npy input, which has no physical-spacing metadata.)"
        )

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
    ]

    random_transform = []
    if is_train and random_aug:

        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]

        )

        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )
    roi_crop = None
    if spacing_type == "original" and val_crop_size is not None:
        #Do a center crop
        roi_crop = [CenterSpatialCropd(keys=keys, roi_size=val_crop_size, allow_missing_keys=True)]


    if is_train:
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            ),
        ]
    else:
        val_crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True)]

    if is_train:
        train_transforms = Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
        return train_transforms
    else:
        if roi_crop is not None:
            val_transforms = Compose(common_transform + roi_crop + val_crop + final_transform)
        else:
            val_transforms = Compose(common_transform + val_crop + final_transform)
        return val_transforms


class VAE_Transform:
    """
    A class to handle MAISI VAE transformations for CT, for pre-normalized `.npy` input.
    """

    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        val_crop_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
    ):
        """
        Initialize the VAE_Transform.

        Args:
            is_train (bool): Whether it's for training or not. If True, the output transform will consider random_aug, the cropping will use "patch_size" for random cropping. If False, the output transform will alwasy treat "random_aug" as False, will use "val_patch_size" for central cropping.
            random_aug (bool): Whether to apply random data augmentation for training.
            k (int, optional): Patches should be divisible by k. Defaults to 4.
            patch_size (List[int], optional): Size of the patches. Defaults to [128, 128, 128]. Will random crop patch for training.
            val_patch_size (Optional[List[int]], optional): Size of validation patches. Defaults to None. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
            output_dtype (torch.dtype, optional): Output data type. Defaults to torch.float32.
            spacing_type (str, optional): Type of spacing. Defaults to "original". Choose from ["original", "rand_zoom"].
            spacing (Optional[List[float]], optional): Unused (kept for signature compatibility). Defaults to None.
            image_keys (List[str], optional): List of image keys. Defaults to ["image"].
            label_keys (List[str], optional): List of label keys. Defaults to [].
            additional_keys (List[str], optional): List of additional keys. Defaults to [].
            select_channel (int, optional): Channel to select for multi-channel MRI. Defaults to 0.
        """
        if spacing_type not in ["original", "rand_zoom"]:
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'rand_zoom']. Got {spacing_type}. "
                "('fixed' is not supported for .npy input, which has no physical-spacing metadata.)"
            )

        self.is_train = is_train


        self.transform = define_vae_transform(
            is_train=is_train,
            random_aug=random_aug,
            k=k,
            patch_size=patch_size,
            val_patch_size=val_patch_size,
            val_crop_size = val_crop_size,
            output_dtype=output_dtype,
            spacing_type=spacing_type,
            spacing=spacing,
            image_keys=image_keys,
            label_keys=label_keys,
            additional_keys=additional_keys,
            select_channel=select_channel,
        )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        """
        Apply the appropriate transform to the input image.

        Args:
            img (dict): Input image dictionary.

        Returns:
            Composed Transform

        Raises:
            ValueError: If the modality is not 'ct' or 'mri'.
        """

        transform = self.transform

        return transform(img)