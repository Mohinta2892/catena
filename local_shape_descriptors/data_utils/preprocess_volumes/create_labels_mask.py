import numpy as np
import os
import zarr
import argparse
from glob import glob


def create_mask(labels, background_value):
    labels_mask = np.ones_like(labels).astype(np.uint8)
    assert background_value in labels, f"Background value {background_value} must exist in labels array!"
    background_mask = labels == background_value
    labels_mask[background_mask] = 0
    # example: reset the labels background from 18446744073709551613 to 0 for hemi-brain
    # asuumed that the backgroound is 0. TODO: expand
    # labels[background_mask] = 0

    return labels_mask


def pad_array(input_shape, output_shape, input_arr, bg_value):
    pad_sizes = []
    for dim_in, dim_out in zip(input_shape, output_shape):
        pad_total = dim_out - dim_in
        pad_sizes.append((pad_total // 2, pad_total - pad_total // 2))

    padded_arr = np.pad(input_arr, pad_sizes, mode='constant', constant_values=bg_value)
    return padded_arr


def main():
    parser = argparse.ArgumentParser(description="creates and saves a labels mask dataset in the input zarr")
    parser.add_argument('-dir', help="directory of zarrs")
    parser.add_argument('-labels_ds', help="Label dataset, example: volumes/labels/neuron_ids")
    parser.add_argument('-raw_ds', default="volumes/raw", help="Label dataset, example: volumes/raw")
    parser.add_argument('-bg_val', type=int, help="Background label value (can be 0 or any other large number)")

    args = parser.parse_args()

    list_of_zarrs = glob(f"{args.dir}/*zarr")

    for file_ in list_of_zarrs:
        print(f"Processing file: {file_}")
        zf = zarr.open(file_, "a")
        raw = zf[args.raw_ds][...]
        labels = zf[args.labels_ds][...]  # must fit in ram

        offset = zf[args.labels_ds].attrs["offset"]
        resolution = zf[args.labels_ds].attrs["resolution"]

        # resize labels to have same size as raw if they are different
        if labels.shape != raw.shape:
            labels = pad_array(input_shape=labels.shape, output_shape=raw.shape, input_arr=labels, bg_value=args.bg_val)
            zf["volumes/labels/neuron_ids"] = labels
            zf["volumes/labels/neuron_ids"].attrs["offset"] = offset
            zf["volumes/labels/neuron_ids"].attrs["resolution"] = resolution

        labels_mask = create_mask(labels, background_value=args.bg_val)
        zf["volumes/labels/labels_mask"] = labels_mask
        try:
            zf["volumes/labels/labels_mask"].attrs["offset"] = offset
            zf["volumes/labels/labels_mask"].attrs["resolution"] = resolution
        except Exception as e:
            zf["volumes/labels/labels_mask"].attrs["offset"] = (0, 0, 0)  # accept args, for now set defaults
            zf["volumes/labels/labels_mask"].attrs["resolution"] = (8, 8, 8)


if __name__ == '__main__':
    main()
