import os.path
import sys
import zarr
import numpy as np
import tifffile as t
import dask.array as da
import argparse
from glob import glob
import operator

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().cwd()))

from utils import list_keys


def crop_nd(vol, bounding):
    """
    Typically should crop and the raw EM based on the shape of the labels.
    Copied from: https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    :param img: ndarray
    :param bounding: n-dimensional tuple; should match the `shape` of `img`
    :return: a center-cropped ndarray of shape `bounding`
    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, vol.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return vol[slices]


def write_zarr(filename, mode='r'):
    return zarr.open(filename, component=dataset)


def main():
    parser = argparse.ArgumentParser(description="This script crops raw EM volumes to the size of labels."
                                                 "It saves the cropped volumes with their labels"
                                                 " and mask in the same folder "
                                                 "with a filename prefix `cropped_*.zarr`.")
    parser.add_argument('-d', help='Directory containing zarrs to be cropped!')
    parser.add_argument('-b', default=0, type=int,
                        help='Pass the value that denotes background in the volumes. Default 0')
    args = parser.parse_args()

    zarr_list = glob(f"{os.path.join(args.d, '*.zarr')}")

    for z in zarr_list:
        zf = zarr.open(z)
        ds_in_zarr = list_keys(zf)

        # NB: the below are list(str)
        raw_ds = [element for element in ds_in_zarr if 'raw' in element]
        label_ds = [element for element in ds_in_zarr if 'neuron_ids' in element or 'mito_ids' in element]

        try:
            if not len(raw_ds):
                raise Exception(f"raw ds does not exist in {z}.")
            if not len(label_ds):
                raise Exception(f"raw ds does not exist in {z}.")
        except Exception as e:
            print(e)

        # sanity check - shapes of raw and labels differ??
        try:

            raw = zf[raw_ds[0]]
            labels = zf[label_ds[0]]

            print(f"Shapes of raw EM and labels {raw.shape}, {labels.shape}")
            if raw.shape == labels.shape:
                # Todo: Change this to a more actionable user interaction?
                print("RAW shape == Label shape. Are you sure you wish to crop??")

        except Exception as e:
            print(e)

        # crop the raw EM
        cropped_raw = crop_nd(vol=raw[...], bounding=labels.shape)

        # create a mask for the labels??
        labels_mask_ds = 'volumes/labels/labels_mask'

        labels_mask = np.ones_like(labels).astype(np.uint8)
        background_mask = labels == int(args.b)  # background value
        labels_mask[background_mask] = 0

        out_filename = f"cropped_{os.path.basename(z)}"

        # save the new zarr
        out_zf = zarr.open(os.path.join(os.path.dirname(z), out_filename), mode='a')
        out_zf["volumes/raw"] = cropped_raw
        out_zf["volumes/labels/neuron_ids"] = labels
        out_zf[labels_mask_ds] = labels_mask

        # must set the resolutions/offsets

        out_zf["volumes/raw"].attrs['resolution'] = zf[raw_ds[0]].attrs['resolution']
        out_zf["volumes/raw"].attrs['offset'] = (0, 0, 0)  # zero after cropping right?

        out_zf["volumes/labels/neuron_ids"].attrs['resolution'] = zf[label_ds[0]].attrs['resolution']
        out_zf["volumes/labels/neuron_ids"].attrs['offset'] = (0, 0, 0)  # zero after cropping right?

        out_zf[labels_mask_ds].attrs['resolution'] = zf[raw_ds[0]].attrs['resolution']
        out_zf[labels_mask_ds].attrs['offset'] = (0, 0, 0)  # zero after cropping right?


if __name__ == '__main__':
    main()
