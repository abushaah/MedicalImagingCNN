import os
import nibabel as nib
import numpy as np
from glob import glob
import psutil

"""
    Get data inputs, assumes CT volumes and segmentation masks have corresponding names and indices.
    Analyze the nifti datasets for MONAI parameter adjustments
    :param str in_dir: file path of data.
"""
def process(in_dir):
    volume_dict = {}
    segmentation_dict = {}

    # find all .nii files under in_dir
    nii_files = glob(os.path.join(in_dir, "**", "*.nii"), recursive=True)

    for filepath in nii_files:
        filename = os.path.basename(filepath)
        if filename.startswith("volume-"):
            idx = int(filename.split("-")[1].split(".")[0])
            volume_dict[idx] = filepath
        elif filename.startswith("segmentation-"):
            idx = int(filename.split("-")[1].split(".")[0])
            segmentation_dict[idx] = filepath

    # match volume and segmentation by idx
    matched_keys = sorted(set(volume_dict.keys()) & set(segmentation_dict.keys()))
    all_files = [{"vol": volume_dict[k], "seg": segmentation_dict[k]} for k in matched_keys]

    # split 80% train / 20% validation
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    validation_files = all_files[split_idx:]
    
    # analyze voxel sizes and shapes
    voxel_sizes = []
    shapes = []
    for k in matched_keys:
        img = nib.load(volume_dict[k])
        data = img.get_fdata()
        voxel_sizes.append(img.header.get_zooms())
        shapes.append(data.shape)

    # pixdim based on variables in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
    mean_spacing = np.mean(voxel_sizes, axis=0)
    mean_shape = np.mean(shapes, axis=0)
    pixdim = tuple(round(s, 2) for s in mean_spacing)

    # default for soft tissue
    a_min, a_max = -200, 250

    # detect GPU & RAM memory
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        mem_free_gpu = max([gpu.memoryFree for gpu in gpus])  # in MB
    except Exception:
        mem_free_gpu = 0  # fallback to CPU

    mem_free_ram = psutil.virtual_memory().available // (1024 * 1024)

    # adjust preprocessing resolution based on memory
    # values are randomized based on https://docs.monai.io/en/stable/transforms.html
    if mem_free_gpu >= 20000:
        spatial_size = [256, 256, 256]
        batch_size = 2
    elif mem_free_gpu >= 10000:
        spatial_size = [192, 192, 128]
        batch_size = 1
    elif mem_free_gpu >= 4000:
        spatial_size = [128, 128, 64]
        batch_size = 1
    else:
        spatial_size = [96, 96, 64]
        batch_size = 1

    return {
        "train_files": train_files,
        "validation_files": validation_files,
        "pixdim": pixdim,
        "a_min": a_min,
        "a_max": a_max,
        "spatial_size": spatial_size,
        "batch_size": batch_size,
        "mem_free_gpu": mem_free_gpu,
        "mem_free_ram": mem_free_ram,
    }