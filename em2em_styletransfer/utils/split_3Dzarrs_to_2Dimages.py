"""

Program for splitting Zarr and HDF datasets into 2D image slices.

This program reads multiple Zarr and HDF datasets from the specified paths and splits them into 2D image slices
along the first dimension (Z). The 2D slices are then saved in separate folders within the output directory.

Usage:
The program expects you to provide the paths to the Zarr and HDF datasets and the output directory (out_f).

Notes:
- The program assumes that the datasets are organized in the form of Z x Y x X (depth x height x width).
- The program saves the 2D slices as JPEG/TIFF/PNG images with the '.jpg'/ '.png' / 'tif' file extension.
- The cropping size for the 2D slices can be adjusted using the 'crop_size' parameter in the split functions.

Author: Samia Mohinta
Affiliation: Cardona Lab, Cambridge University, UK
"""
import os
from glob import glob
import shutil
import zarr
import h5py
from typing import Union
from pathlib import Path
from list_ds_keys import *
import argparse
import operator
import skimage.io as io
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from copy import deepcopy
import cv2


def cropND(img, bounding):
    """
     Center crops an ND volume to the preferred bounding box size.

     This function performs a center crop on an N-dimensional (ND) volume (e.g., 3D, 4D, etc.) to the specified
     bounding box size. The bounding box is centered on the volume, and the image is cropped accordingly.

     Parameters:
         img (numpy.ndarray): The ND volume to be cropped.
         bounding (tuple): The target size for the bounding box in each dimension.

     Returns:
         numpy.ndarray: The center-cropped ND volume.

     Raises:
         ValueError: If the bounding box size is not compatible with the input image.

     Notes:
         - The bounding box should have the same number of dimensions as the input volume.
         - The bounding box size should be smaller than or equal to the corresponding dimension of the input volume.
    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def read_zarr(z_path: Union[str, Path], mode='r'):
    """
    Reads a Zarr dataset from the specified path.

    This function opens a Zarr dataset from the given file path and returns it in the specified mode.

    Parameters:
        z_path (Union[str, Path]): The path to the Zarr dataset.
        mode (str, optional): The file mode to open the dataset (e.g., 'r', 'r+', 'w', etc.). Default is 'r'.

    Returns:
        zarr.core.Array: The Zarr dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    return zarr.open(z_path, mode=mode)


def read_hdf(h_path: Union[str, Path], mode='r'):
    """
    Reads an HDF dataset from the specified path.

    This function opens an HDF dataset from the given file path and returns it in the specified mode.

    Parameters:
        h_path (Union[str, Path]): The path to the HDF dataset.
        mode (str, optional): The file mode to open the dataset (e.g., 'r', 'r+', 'w', etc.). Default is 'r'.

    Returns:
        h5py.File: The HDF dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """

    return h5py.File(h_path, mode=mode)


def split_hdf(f, out_f, crop_size=(256, 256), file_ext='.tif'):
    """
        Splits a high-dimensional HDF dataset into 2D image slices and saves them in individual folders.

        This function takes a high-dimensional HDF dataset and creates 2D image slices along the first dimension (Z)
        of the dataset. The slices are then saved as individual image files in corresponding sub-folders
        within the specified output directory.

        Parameters:
            f (h5py.File): The HDF dataset file.
            out_f (str): The output directory where the 2D image slices will be saved.
            crop_size (tuple, optional): The size of the 2D slices to be extracted. Default is (256, 256).
            file_ext (str, optional): The file extension of the saved images. Default is '.tif'.

        Returns:
            None

        Raises:
            TypeError: If the 'f' parameter is not a valid h5py.File object.
            OSError: If the output directory cannot be created.

        Notes:
            - The function assumes that the dataset is organized in the form of Z x Y x X (depth x height x width).
            - The function skips saving slices with the word 'mask' in their dataset key.
            - If the dataset contains 16-bit unsigned integer values, they will be converted to 8-bit uint format.
        """

    dataset_keys = list_keys(f)
    if not os.path.exists(out_f):
        os.makedirs(out_f, exist_ok=True)
    for k in dataset_keys:
        rf = f[k][:]

        # lets skip saving labels_mask for now
        if 'mask' in k:
            continue

        if 'raw' in k and rf.dtype == 'uint16':
            rf = (rf // 255).astype(np.uint8)

        # this creates the subsets inside the main folder based on the dataset key
        out_ff = os.path.join(out_f, os.path.basename(k))
        if not os.path.exists(out_ff):
            os.makedirs(out_ff, exist_ok=True)

        # note: we treat this shape as Z x  Y x X
        shape_rf = rf.shape
        # extract the z_slices too from the middle of the volume
        # hemi - matrix shape 776^3 the labels start from 128 and go till 648, so there are 520 slices
        rf_cropped = cropND(rf, bounding=crop_size)  # bounding=(120,) + crop_size

        # save each z_slice now as a 2D image
        for z_slice in tqdm(range(rf_cropped.shape[0]), desc=k):
            tifffile.imwrite(os.path.join(out_ff, f"img_{z_slice}{file_ext}"), rf_cropped[z_slice, ...])


def seg_to_affgraph(seg, nhood):
    """ Segmentation to affinities. These will be 3D arrays in range of 0-1 (0 = boundaries, 1= cytoplasm."""
    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            ]
                            > 0
                    )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                            ]
                            > 0
                    )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff


def binarize_boundaries(rf_arr, background=0):
    print("All labels are converted to binary maps where 1 = Foreground, 0 = Background")

    def __grow(gt, gt_mask=None, background_val=0, steps=1, binarise=True, only_xy=False):
        if gt_mask is not None:
            assert (
                    gt.shape == gt_mask.shape
            ), "GT_LABELS and GT_MASK do not have the same size."

        if only_xy:
            assert len(gt.shape) == 3
            for z in range(gt.shape[0]):
                __grow(gt[z], None if gt_mask is None else gt_mask[z])
            return

        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=gt.shape, dtype=bool)
        masked = None
        if gt_mask is not None:
            masked = np.equal(gt_mask, 0)
        for label in np.unique(gt):
            if label == background_val:
                continue
            label_mask = gt == label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(
                label_mask, iterations=steps, border_value=1
            )
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = background_val
        if binarise:
            gt_bin = deepcopy(gt)
            gt_bin[gt_bin > background_val] = 1  # membranes are denoted by background val
            return gt, gt_bin
        return gt, None

    rf_arr, rf_arr_bin = __grow(rf_arr, background_val=background)
    rf_arr_bin = (rf_arr_bin * 255).astype(np.uint8)
    return rf_arr_bin


def expand_dims_rgb(arr):
    # Expand dimensions and then tile to create 3 channels

    print("Adding 3 channels at the end of each image arr!")
    # Expand dimensions to create a new axis for channels
    expanded_array = np.expand_dims(arr, axis=-1)  # Shape: (120, 256, 256, 1)

    # Tile along the new axis to create 3 channels
    arr = np.tile(expanded_array, (1, 1, 1, 3))  # Shape: (120, 256, 256, 3)

    return arr


def split_zarr(f, out_f, crop_size=(256, 256), file_ext='.tif', binarize=False, save_aff=True, save_rgb=True):
    """
   Splits a high-dimensional Zarr dataset into 2D image slices and saves them in individual folders.

   This function takes a high-dimensional Zarr dataset and creates 2D image slices along the first dimension (Z)
   of the dataset. The slices are then saved as individual image files in corresponding sub-folders
   within the specified output directory.

   Parameters:
       f (zarr.core.Array): The Zarr dataset array.
       out_f (str): The output directory where the 2D image slices will be saved.
       crop_size (tuple, optional): The size of the 2D slices to be extracted. Default is (256, 256).
       file_ext (str, optional): The file extension of the saved images. Default is '.tif'.
       binarize (bool, optional): Should the labels be converted to binary maps. Default is False.
       save_aff (bool, optional): Should the labels be saved as affinities maps. Default is True.
       save_rgb (bool, optional): Should you save the images as rgb images with 3 channels. Default is True.

   Returns:
       None

   Raises:
       TypeError: If the 'f' parameter is not a valid zarr.core.Array object.
       OSError: If the output directory cannot be created.

   Notes:
       - The function assumes that the dataset is organized in the form of Z x Y x X (depth x height x width).
       - The function skips saving slices with the word 'mask' in their dataset key.
       - If the dataset contains 16-bit unsigned integer values, they will be converted to 8-bit uint format.
   """

    dataset_keys = list_keys(f)
    if not os.path.exists(out_f):
        os.makedirs(out_f, exist_ok=True)
    for k in dataset_keys:
        rf = f[k][:]

        # lets skip saving labels_mask for now
        if 'mask' in k:
            continue

        if 'raw' in k and rf.dtype == 'uint16':
            rf = (rf // 255).astype(np.uint8)

        # this creates the subsets inside the main folder based on the dataset key
        out_ff = os.path.join(out_f, os.path.basename(k))
        if not os.path.exists(out_ff):
            os.makedirs(out_ff, exist_ok=True)

        # note: we treat this shape as Z x  Y x X
        shape_rf = rf.shape
        # extract the z_slices too from the middle of the volume
        # hemi - matrix shape 776^3 the labels start from 128 and go till 648, so there are 520 slices
        rf_cropped = cropND(rf, bounding=crop_size)  # bounding=(120,) + crop_size

        # binarise the labels or save them as is
        if binarize and 'raw' not in k:  # Todo add control in main as an argparse arg
            rf_cropped = binarize_boundaries(rf_cropped, background=0)

        if save_aff and 'raw' not in k:
            rf_cropped = seg_to_affgraph(rf_cropped, nhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if rf_cropped.shape[0] == 3:  # has 3 channels at first, we have to transpose this
            rf_cropped = rf_cropped.transpose((1, 2, 3, 0))

        # make it ZxYxXxChannels
        if save_rgb and len(rf_cropped.shape) != 4:
            rf_cropped = expand_dims_rgb(rf_cropped)

        # save each z_slice now as a 2D image
        for z_slice in tqdm(range(rf_cropped.shape[0]), desc=k):
            cv2.imwrite(os.path.join(out_ff, f"img_{str(z_slice).zfill(5)}{file_ext}"), rf_cropped[z_slice, ...])
            # tifffile.imwrite(os.path.join(out_ff, f"img_{str(z_slice).zfill(5)}{file_ext}"), rf_cropped[z_slice, ...])


def sanity_check(path_2d_raw, path_2d_labels):
    """ This is not a good effort. Instead regenerate a zarr from tifs and visualize them via napari """

    # plot 20 raw overlayed with labels
    fig, axs = plt.subplots(20, 5, figsize=(12, 30))
    max_imgs = 100
    counter_x = counter_y = 0

    for rf, lf in zip(sorted(glob(f"{path_2d_raw}/*")), sorted(glob(f"{path_2d_labels}/*"))):
        r = io.imread(rf)
        l = io.imread(lf)

        max_imgs -= 1
        if max_imgs < 0:
            break

        axs[counter_x][counter_y].imshow(r, cmap='gray')
        axs[counter_x][counter_y].imshow(l, cmap='prism', alpha=0.2)
        axs[counter_x][counter_y].axis('off')

        counter_y += 1
        if counter_y > 4:
            counter_y = 0
            counter_x += 1

    fig.subplots_adjust(wspace=0.05, hspace=0.005)
    plt.show()


if __name__ == '__main__':
    # TODO - make it argparse

    # You can provide a path to multiple zarrs
    z_path = '/media/samia/DATA/mounts/cephfs/img2img-turbo/data/3d_zarrs'
    out_f = '/media/samia/DATA/mounts/cephfs/img2img-turbo/data/2d_pngs'
    split_hdf_bool = False  # set this to true if you have h5 files in the same path
    l_z_paths = sorted(glob(f"{z_path}/*.zarr"))
    l_h_paths = sorted(glob(f"{z_path}/*.h5"))

    assert len(l_z_paths) > 0, "No zarr files found in this path!"
    assert not split_hdf_bool or len(l_h_paths) > 0, "No h5 files found in this path!"

    for z in l_z_paths:
        f = read_zarr(z)
        split_zarr(f, out_f=os.path.join(out_f, os.path.basename(z).split('.')[0]), crop_size=(120, 256, 256),
                   # (z,y,x)
                   file_ext='.png', binarize=False, save_rgb=True)

    # TODO - activate this when user hits a button perhaps

    if split_hdf_bool:
        for h in l_h_paths:
            f = read_hdf(h)
            split_hdf(f, out_f=os.path.join(out_f, os.path.basename(h).split('.')[0]), crop_size=(120, 256, 256),
                      # (z,y,x)
                      file_ext='.png')

    # sanity checking
    # sanity_check(
    #     '/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr_2D/eb-inner-groundtruth-with-context-x20172-y2322-z14332/raw',
    #     '/media/samia/DATA/ark/dan-samia/lsd/funke/hemi/training/zarr_2D/eb-inner-groundtruth-with-context-x20172-y2322-z14332/neuron_ids')
