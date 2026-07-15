"""
Script for preprocessing the LIDC-IDRI dataset.
based on: https://github.com/pfriedri/wdm-3d/blob/main/utils/preproc_lidc-idri.py
"""
import os
import argparse
import logging
import rasterio
import pylidc as pl
import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom
from pylidc.utils import consensus
from lungmask import LMInferer
from scipy.ndimage import label, center_of_mass
from .utils_lidc3D import regularize_components_minimal, extract_3d_contours, fit_nparray_to_given_size

# --- pylidc / numpy compatibility shim -------------------------------------
# pylidc (last released 2021, lightly maintained) still calls the deprecated
# `np.int` alias internally (see pylidc/Contour.py: `.astype(np.int)`), which
# was fully removed in numpy>=1.24. Rather than downgrading numpy for the
# whole pipeline (torch/monai/etc. expect a modern numpy), restore just the
# alias pylidc needs, at runtime, without touching numpy's installed version
# or pylidc's own files.
if not hasattr(np, "int"):
    np.int = int
# ----------------------------------------------------------------------------

def extract_sorted_centroids(mask):
    """
    Extract centroids of connected components in a mask, sorted by component size (largest first).

    Parameters:
        mask (ndarray): A 2D or 3D NumPy array with 0 as background and non-zero values as foreground.

    Returns:
        List of tuples: Centroids (in float coordinates) of the connected components, sorted by size descending.
    """
    binary_mask = mask > 0
    labeled_array, num_features = label(binary_mask)

    sizes = np.array([(labeled_array == i).sum() for i in range(1, num_features + 1)])
    centroids = center_of_mass(binary_mask, labeled_array, range(1, num_features + 1))

    # Sort by size descending
    sorted_indices = np.argsort(-sizes)
    sorted_centroids = [centroids[i] for i in sorted_indices]

    return sorted_centroids

def extract_centroids(mask):
    """
    Extract centroids of connected components in a mask.

    Parameters:
        mask (ndarray): A 2D or 3D NumPy array with 0 as background and non-zero values as foreground.

    Returns:
        List of tuples: Centroids (in float coordinates) of the connected components.
    """
    # Create a binary mask
    binary_mask = mask > 0

    # Label connected components
    labeled_array, num_features = label(binary_mask)

    # Calculate centroids
    centroids = center_of_mass(binary_mask, labeled_array, range(1, num_features + 1))

    return centroids

def crop_around_centroid(volume, centroid, crop_size=64):
    if volume.ndim == 4:
        volume = volume[0,:,:,:]
    cx, cy, cz = centroid
    half = crop_size // 2
    x_start = max(cx - half, 0)
    y_start = max(cy - half, 0)
    z_start = max(cz - half, 0)

    x_end = min(x_start + crop_size, volume.shape[0])
    y_end = min(y_start + crop_size, volume.shape[1])
    z_end = min(z_start + crop_size, volume.shape[2])

    # Re-adjust start if we hit the boundary
    x_start = x_end - crop_size
    y_start = y_end - crop_size
    z_start = z_end - crop_size

    return volume[x_start:x_end, y_start:y_end, z_start:z_end]

def uint8(array):
    # the input npy array is assumed to be normalized in [0, 1]
    array = (array*255).astype(np.uint8)
    array = np.clip(array, 0, 255)
    return array


def save_ct_npy_as_image_seq(npy_path, out_dir, extension="png", prefix="ct"):

    assert os.path.exists(npy_path)
    array = np.load(npy_path)
    image_size = array.shape[0]
    os.makedirs(f"{out_dir}/{prefix}_axial", exist_ok=True)
    os.makedirs(f"{out_dir}/{prefix}_sagittal", exist_ok=True)
    os.makedirs(f"{out_dir}/{prefix}_coronal", exist_ok=True)

    if extension == "tif":
        profile = {
            'driver': 'GTiff',
            'interleave': 'band',
            'tiled': True,
            'height': image_size,
            'width': image_size,
            'compress': 'lzw',
            'nodata': 0,
            'dtype': np.float32,
            'count': 1
        }
        for i in range(array.shape[0]):
            with rasterio.open(f'{out_dir}/{prefix}_axial/slice_{i:03d}.tif', 'w', **profile) as dst:
                dst.write(array[i, :, :], 1)
        for i in range(array.shape[1]):
            with rasterio.open(f'{out_dir}/{prefix}_coronal/slice_{i:03d}.tif', 'w', **profile) as dst:
                dst.write(array[:, i, :], 1)
        for i in range(array.shape[2]):
            with rasterio.open(f'{out_dir}/{prefix}_sagittal/slice_{i:03d}.tif', 'w', **profile) as dst:
                dst.write(array[:, :, i], 1)
    else:
        #extension is png and the input npy array is assumed to be normalized in [0, 1]
        array = uint8(array)
        for i in range(array.shape[0]):
            img = Image.fromarray(array[i, :, :])
            img.save(f'{out_dir}/{prefix}_axial/slice_{i:03d}.png')
        for i in range(array.shape[1]):
            img = Image.fromarray(array[:, i, :])
            img.save(f'{out_dir}/{prefix}_coronal/slice_{i:03d}.png')
        for i in range(array.shape[2]):
            img = Image.fromarray(array[:, :, i])
            img.save(f'{out_dir}/{prefix}_sagittal/slice_{i:03d}.png')

def extract_central_crop(array, crop_size=256, fill=False):
    d, h, w = array.shape
    d_start = max(0, (d - crop_size) // 2)
    h_start = max(0, (h - crop_size) // 2)
    w_start = max(0, (w - crop_size) // 2)
    cropped_arr = array[d_start:d_start + crop_size, h_start:h_start + crop_size, w_start:w_start + crop_size]
    if fill:
        output = np.zeros((crop_size, crop_size, crop_size))
        d, h, w = cropped_arr.shape
        d_start = crop_size//2-d//2
        h_start = crop_size//2-h//2
        w_start = crop_size//2-w//2
        output[d_start:d_start + d, h_start:h_start + h, w_start:w_start + w] = cropped_arr
        return output
    return cropped_arr


def normalize_ct_vol_wdm(ct_array, min_hu_val=-1000, max_hu_val=2500):
    #from https://github.com/pfriedri/wdm-3d/blob/main/utils/preproc_lidc-idri.py
    ct_array[ct_array < -1000] = -1000
    out_clipped = np.clip(ct_array, -1000, np.quantile(ct_array, 0.999))
    out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
    return out_normalized

def normalize_ct_vol(ct_array, min_hu_val=-1000, max_hu_val=2500):
    """
    min_hu_val = -1000 # = "air" in Hounsfield unit (HU) https://en.wikipedia.org/wiki/Hounsfield_scale
    max_hu_val = 2500 # in the LDIC data the largest HU in a nodule is ~2400 (however values above 1500 are exceptional)
    """
    ct_array = (ct_array - min_hu_val)/(max_hu_val - min_hu_val)
    ct_array = np.clip(ct_array, 0, 1)
    return ct_array

def denormalize_ct_vol(ct_array, min_hu_val=-1000, max_hu_val=2500):
    ct_array = ct_array * (max_hu_val - min_hu_val) + min_hu_val
    return ct_array


def compute_lung_mask(ct_array, padding=False):
    # input ct_array is a DxHxW numpy array with HU values --> important! HU range expected
    # uses https://github.com/JoHof/lungmask
    # output mask contains 1 where there is lung 0 elsewhere
    # default model is U-net(R231)

    inferer = LMInferer(tqdm_disable=True)
    logger = logging.getLogger("lungmask")
    logger.disabled = True # avoid verbose
    
    if padding:
        # the segmenter only works if the full lung is surrounded by tissue 
        # adding some padding with tissue-like HU values makes it more robust
        pad = 10
        d, h, w = ct_array.shape
        input_image = np.ones((d + pad*2, h + pad*2, w + pad*2))*1000 
        input_image[pad:pad+d, pad:pad+h, pad:pad+w] = ct_array
        mask = inferer.apply(input_image)
        mask = mask[pad:pad+d, pad:pad+h, pad:pad+w]
    else:
        mask = inferer.apply(ct_array)  
    mask[mask>0] = 1
    return mask

def get_ct_thumbnail(ct_array, mask, nodule_centroid):
    # input ct_array is a (256, 256, 256) numpy array with range 0-1
    # input mask is a (256, 256, 256) numpy array with range 0-1

    x, y, z = nodule_centroid

    # convert ct array to RGB
    #ct_array = uint8(ct_array)
    ct_array = np.stack([ct_array, ct_array, ct_array], axis=0) # ct_array.shape = (3, D, H, W)

    # get the ct slices corresponding to the nodule centroid
    slice_x = ct_array[:, int(x), :, :].transpose((1,2,0))
    slice_y = ct_array[:, :, int(y), :].transpose((1,2,0))
    slice_z = ct_array[:, :, :, int(z)].transpose((1,2,0))

    # add a bounding box around the nodule
    bbx_mask = regularize_components_minimal(mask)
    bbx_mask = extract_3d_contours(bbx_mask, width=2)
    bbx_mask = np.stack([bbx_mask, bbx_mask, bbx_mask], axis=0)  # bbx_mask.shape = (3, D, H, W)
    bbx_mask[1:, :, :, :] = 0
    mask_x = bbx_mask[:, int(x), :, :].transpose((1,2,0))
    mask_y = bbx_mask[:, :, int(y), :].transpose((1,2,0))
    mask_z = bbx_mask[:, :, :, int(z)].transpose((1,2,0))

    im = np.hstack([slice_x, slice_y, slice_z])
    im_mask = np.hstack([mask_x, mask_y, mask_z])
    im[im_mask > 0] += 0.5
    im = np.clip(im, 0, 1)
    return im

def save_ct_thumbnail(ct_array, mask, nodule_centroid, out_path):
    # save ct thumbnail
    im = get_ct_thumbnail(ct_array, mask, nodule_centroid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(uint8(im)).save(out_path)

def get_mask_thumbnail(mask, nodule_centroid):
    # input mask is a (256, 256, 256) numpy array with range 0-5
    #            mask > 1   --> nodule
    #            mask = 0.5 --> lung
    #            mask = 0   --> elsewhere

    x, y, z = nodule_centroid

    # convert mask to RGB
    #ct_array = uint8(ct_array)
    mask = np.stack([mask, mask, mask], axis=0) # ct_array.shape = (3, D, H, W)

    # normalize_mask
    mask = np.clip(mask/5, 0, 1)

    # get the ct slices corresponding to the nodule centroid
    mask_x = mask[:, int(x), :, :].transpose((1,2,0))
    mask_y = mask[:, :, int(y), :].transpose((1,2,0))
    mask_z = mask[:, :, :, int(z)].transpose((1,2,0))

    im_mask = np.hstack([mask_x, mask_y, mask_z])
    return im_mask

def save_mask_thumbnail(mask, nodule_centroid, out_path):
    im_mask = get_mask_thumbnail(mask, nodule_centroid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(uint8(im_mask)).save(out_path)


def preprocess_dicom(dicom_dir_path, npy_dir_path,
                     normalize=True,
                     resample=False, target_res=1., 
                     central_crop=False, crop_size=256,
                     verbose=False):

    # e.g. patient_id = 'LIDC-IDRI-0078'
    # clevel = 0.25 means if 1/4 doctors marked it then we keep that part
    # amount of padding in each direction
    clevel = 0.25
    
    print(f"\nProcessing {dicom_dir_path} ...")

    # Query for a scan, and convert it to an array volume.
    assert os.path.exists(dicom_dir_path)
    assert "LIDC-IDRI-" in dicom_dir_path
    patient_id = dicom_dir_path.split("/")[-1]
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    vol = scan.to_volume(verbose=False)

    # Cluster the annotations for the scan
    nods = scan.cluster_annotations(verbose=False)

    # Extract mask array -> 1 if nodule, 0 else
    mask_vol = np.zeros_like(vol, dtype=np.float32)
    for nod_idx, anns in enumerate(nods):
        cmask, cbbox, _ = consensus(anns, clevel=clevel) #, crop_size=(crop_size, crop_size))
        texture = np.array([ann.texture for ann in anns]).mean()
        cmask = cmask.astype(np.float32)
        mask_vals = np.maximum(mask_vol[cbbox[0].start:cbbox[0].stop, cbbox[1].start:cbbox[1].stop, cbbox[2].start:cbbox[2].stop], cmask * texture)
        mask_vol[cbbox[0].start:cbbox[0].stop, cbbox[1].start:cbbox[1].stop, cbbox[2].start:cbbox[2].stop] = mask_vals
        if texture < 1:
            print(patient_id)
        assert texture >= 1

    # Resample
    if resample:
        zoom_factors = [scan.pixel_spacing/target_res, scan.pixel_spacing/target_res, scan.slice_spacing/target_res]
        resampled_vol = zoom(vol, zoom_factors, order=3, mode='nearest')
        resampled_mask_vol = zoom(mask_vol, zoom_factors, order=0, mode='nearest') #order = 0 to simulate nearest neighbor interpolation 
    else:
        resampled_vol = vol
        resampled_mask_vol = mask_vol

    # Reorganize so first axis is axial, second is coronal and third is saggital
    resampled_vol = resampled_vol.transpose((2, 0, 1))[::-1, :, :]
    resampled_mask_vol = resampled_mask_vol.transpose((2, 0, 1))[::-1, :, :]

    # Compute lung mask and merge it with nodule mask
    final_mask = compute_lung_mask(resampled_vol).astype(np.float32)
    final_mask[final_mask>0] = 0.5
    final_mask[resampled_mask_vol >= 1] = resampled_mask_vol[resampled_mask_vol >= 1]

    # Extract central crop
    if central_crop:
        resampled_vol = extract_central_crop(resampled_vol, crop_size=crop_size)
        resampled_mask_vol = extract_central_crop(resampled_mask_vol, crop_size=crop_size)
        final_mask = extract_central_crop(final_mask, crop_size=crop_size)

    # Normalize HU values between 0 and 1
    if normalize:
        resampled_vol = normalize_ct_vol_wdm(resampled_vol)

    # Make sure everything has shape (256, 256, 256)
    Nsize = crop_size
    resampled_vol = fit_nparray_to_given_size(resampled_vol, size=Nsize)
    resampled_mask_vol = fit_nparray_to_given_size(resampled_mask_vol, size=Nsize)
    final_mask = fit_nparray_to_given_size(final_mask, size=Nsize)

    # Write output npys
    npy_ct_path = os.path.join(npy_dir_path, f"chest_ct/{patient_id}.npy")
    npy_nodules_mask_path = os.path.join(npy_dir_path, f"mask/{patient_id}.npy")
    os.makedirs(os.path.dirname(npy_ct_path), exist_ok=True)
    os.makedirs(os.path.dirname(npy_nodules_mask_path), exist_ok=True)
    np.save(npy_ct_path, resampled_vol.astype(np.float32))
    np.save(npy_nodules_mask_path, final_mask.astype(np.float32))

    # Save thumbnails
    nodule_centroids = extract_centroids(resampled_mask_vol)
    nodule_mask = (resampled_mask_vol >= 1)
    if not nodule_centroids:
        nodule_centroids = [(Nsize//2, Nsize//2, Nsize//2)] # exception: no nodule, probably due to downsampling
    save_ct_thumbnail(resampled_vol, nodule_mask, nodule_centroids[0], f"{npy_dir_path}/{patient_id}-ct-thumbnail.png")
    save_mask_thumbnail(final_mask, nodule_centroids[0], f"{npy_dir_path}/{patient_id}-mask-thumbnail.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', type=str, required=True,
                        help='Input directory containing the original dicom data')
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Ouput directory to store the processed npy files')
    parser.add_argument('--png_dir', type=str, default=None,
                        help='Ouput directory to store the processed npy files as png seqs')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize HU values between 0 and 1')
    parser.add_argument('--resample', action='store_true',
                        help='All output samples will be resampled according to the --resolution value')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Output resolution in the x, y, z axes. Default: 1 mm/px')
    parser.add_argument('--central_crop', action='store_true',
                        help='All output samples will be cropped according to the --crop_size value')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Length N of a regular (N, N, N) central crop)')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Integer specifying a single sample index to process from the LIDC-IDRI data.' \
                        'If None, all samples will be processed. Default: None')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose info')
    args = parser.parse_args()

    # Convert DICOM to NPY
    print(f"     --dicom_dir     {args.dicom_dir}")
    print(f"     --npy_dir       {args.npy_dir}")
    print(f"     --png_dir       {args.png_dir}")
    print(f"     --normalize     {args.normalize}")
    print(f"     --resample      {args.resample}")
    print(f"     --resolution    {args.resolution}")
    print(f"     --central_crop  {args.central_crop}")
    print(f"     --crop_size     {args.crop_size}")
    print(f"     --sample_idx    {args.sample_idx}")

    if args.sample_idx is None:
        patients_to_process = sorted(os.listdir(args.dicom_dir))
        print(f"\nFound {len(patients_to_process)} patients in dicom_dir\n")
    else:
        patients_to_process = [f"LIDC-IDRI-{int(args.sample_idx):04d}"]
        print(f"\nProcessing only {patients_to_process[0]}\n")

    for patient in tqdm(patients_to_process):
        if "LIDC-IDRI-" not in patient:
            continue

        dicom_dir_path = os.path.join(args.dicom_dir, patient)
        npy_dir_path = os.path.join(args.npy_dir, patient)
        npy_path = os.path.join(npy_dir_path, f"chest_ct/{patient}.npy")
        if os.path.exists(npy_path) and os.path.exists(f"{npy_dir_path}/{patient}-ct-thumbnail.png"):
            print(f"\nWarning: {npy_path} already exists! Skipping...\n")
        else:
            preprocess_dicom(dicom_dir_path, npy_dir_path,
                            normalize=args.normalize,
                            resample=args.resample, target_res=args.resolution,
                            central_crop=args.central_crop, crop_size=args.crop_size,
                            verbose=args.verbose)

        if args.png_dir is not None:
            png_dir_path = os.path.join(args.png_dir, f"{patient}/png_seq")
            npy_path = os.path.join(npy_dir_path, f"chest_ct/{patient}.npy")
            save_ct_npy_as_image_seq(npy_path, png_dir_path, prefix="ct", extension="tif")
            npy_path = os.path.join(npy_dir_path, f"mask/{patient}.npy")
            save_ct_npy_as_image_seq(npy_path, png_dir_path, prefix="mask", extension="tif")

    print("\nAll done!\n")