import re
from glob import glob
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

"""
    Use MONAI transforms to prepares data for segmentation.
    Voxel: 3D grid representation of data.
    
    :param tuple pixdim: standard voxel spacing (in millimeters) for resampling the images in the x, y, and z dimensions.
    :param int a_min: intensity voxel min for CT scans (less are clipped before scaling).
    :param int a_max: intensity voxel max for CT scans (more are clipped before scaling).
    :param int array spatial_size: output size (in voxel) to which each image and label volume will be resized. AKA input size for the neural network.
    :param int batch_size: adjyst batch size, default is 1.
    :return PyTorch DataLoader objects: used to train neural network.
"""
def transforms(pixdim, a_min, a_max, spatial_size, batch_size, cache, train_files, validation_files):

    # reproduce training results
    set_determinism(seed=0)

    # and apply transformations to them
    # parameters from https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["vol", "seg"],
            label_key="seg",
            spatial_size=spatial_size,  # use your configured size here
            pos=1, neg=1,
            num_samples=4,
            image_key="vol",
            image_threshold=0,
        ),
        ToTensord(keys=["vol", "seg"]),
    ])

    # transforms for validation data
    validation_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
    ])

    if cache >= 16000:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)

        # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
        # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        validation_ds = Dataset(data=validation_files, transform=validation_transforms)

        # train_ds = Dataset(data=train_files, transform=train_transforms, num_workers=4)
        # validation_ds = Dataset(data=validation_files, transform=validation_transforms, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size)

    # use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    # validation_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=4)

    return train_loader, validation_loader