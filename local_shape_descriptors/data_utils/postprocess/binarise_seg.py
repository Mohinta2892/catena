import copy

import zarr
import numpy as np
import argparse
import sys
import os
import tifffile as t
from scipy import ndimage
import copy
from tqdm import tqdm


def binarise_seg():
    pass


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
    for label in tqdm(np.unique(gt), desc='masking'):
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
        gt_bin = copy.deepcopy(gt)
        gt_bin[gt_bin > background_val] = 1  # membranes are denoted by background val
        return gt, gt_bin
    return gt


def main():
    seg_path_mito = "/media/samia/DATA/mounts/zstore1/lsd_outputs/OCTO_3CUBES_ZYX/MTLSD/3d/clahed_hemi_mito_unproof/model_checkpoint_300000/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163_seg.tiff"
    seg_path = "/media/samia/DATA/mounts/zstore1/lsd_outputs/MTLSD/3d/run-aclsd-together/model_checkpoint_latest/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.zarr"
    ds_zarr = "volumes/final_segmentation_hist_quant_45_70"
    # read zarr
    if seg_path.endswith(".zarr"):
        assert ds_zarr is not None, "pass dataset if your segmentation is in a zarr"
        fz_seg = zarr.open(seg_path)[ds_zarr]
    elif seg_path.endswith((".tif", ".tiff")):
        fz_seg = t.imread(seg_path)

    if seg_path_mito.endswith(".zarr"):
        assert ds_zarr is not None, "pass dataset if your segmentation is in a zarr"
        fz_mito = zarr.open(seg_path_mito)[ds_zarr]
    elif seg_path_mito.endswith((".tif", ".tiff")):
        # fz_mito = t.imread(seg_path_mito)
        pass
    # mask = fz_seg[...] > 0

    gt, boundary_mask = __grow(fz_seg[130:526])
    print(np.unique(boundary_mask))
    print(np.unique(gt))

    # out_folder = os.path.dirname(seg_path)
    outfile = f"{os.path.splitext(seg_path)[0]}_binary.tiff"
    outfile_gt = f"{os.path.splitext(seg_path)[0]}_gt.tiff"

    t.imwrite(outfile, boundary_mask)
    # t.imwrite(outfile_gt, gt)


    print()


if __name__ == '__main__':
    main()
