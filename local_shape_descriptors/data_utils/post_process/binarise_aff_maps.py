"""Put this in post-processing volumes.
This can take a lot of time as the data is loaded into memory as ndarrays.
So make sure your file has a size such that fits into your RAM.
"""

import zarr
import tifffile as t
from utils import *
import argparse
import numpy as np
import os


def binarise_aff_maps(args):
    filename = args.f

    # read the zarr file
    zf = read_zarr(filename, mode='r')
    pred_affs = zf["volumes/pred_affs"]  # this dataset key is preset, generated when inference is run

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

    t.imwrite(
        f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/tif/labels_pedro/{os.path.basename(filename)}.tif",
        data=pred_affs_summed,
        bigtiff=True, compression='zlib')

    # write the raw too, because it will be needed to overlay the preds on
    t.imwrite(
        f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/tif/labels_pedro/raw_{os.path.basename(filename)}.tif",
        data=zf["volumes/raw"][...],
        bigtiff=True, compression='zlib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="/path/to/zarr/file.zarr to binarise")
    args = parser.parse_args()
    binarise_aff_maps(args)
