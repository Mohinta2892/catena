"""Put this in post-processing volumes.
This can take a lot of time as the data is loaded into memory as ndarrays.
So make sure your file has a size such that fits into your RAM.
"""

import zarr
import tifffile as t
import argparse
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data_utils.preprocess_volumes.utils import *


def binarise_aff_maps(args):
    filename = args.f

    # read the zarr file
    zf = read_zarr(filename, mode='r+')
    ds = "volumes/pred_affs"
    pred_affs = zf[ds]  # this dataset key is preset, generated when inference is run

    # sum the affinity channels
    pred_affs_summed = np.squeeze(pred_affs[0, ...] + pred_affs[1, ...] + pred_affs[2, ...])
    # sanity check the values
    max_val = np.max(pred_affs_summed)
    min_val = np.min(pred_affs_summed)
    mean_val = np.mean(pred_affs_summed)

    print(f"max of affinities after summing: {max_val}")
    print(f"min of affinities after summing: {min_val}")
    print(f"mean of affinities after summing: {mean_val}")

    # normalise it
    pred_affs_summed = (pred_affs_summed - min_val) / (max_val - min_val)

    print(f"max of affinities after normalising: {np.max(pred_affs_summed)}")
    print(f"min of affinities after normalising: {np.min(pred_affs_summed)}")
    print(f"mean of affinities after normalising: {np.mean(pred_affs_summed)}")

    pred_affs_summed = pred_affs_summed * 255
    pred_affs_summed = pred_affs_summed.astype(np.uint8)

    zf[f"volumes/binary_{os.path.basename(ds)}"] = pred_affs_summed

    # t.imwrite(
    #     f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/tif/labels_pedro/{os.path.basename(filename)}.tif",
    #     data=pred_affs_summed,
    #     bigtiff=True, compression='zlib')

    # write the raw too, because it will be needed to overlay the preds on
    # t.imwrite(
    #     f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/tif/labels_pedro/raw_{os.path.basename(filename)}.tif",
    #     data=zf["volumes/raw"][...],
    #     bigtiff=True, compression='zlib')


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
        gt_bin = deepcopy.copy(gt)
        gt_bin[gt_bin > background_val] = 1  # membranes are denoted by background val
        return gt, gt_bin
    return gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="/path/to/zarr/file.zarr to binarise")
    args = parser.parse_args()
    binarise_aff_maps(args)
