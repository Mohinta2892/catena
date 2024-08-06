import os.path

import numpy as np
import pandas as pd
import zarr
import argparse
import dask.array as da
from typing import Union, List
from pathlib import Path
from tqdm import tqdm


def make_gt_seg2link(seg: np.ndarray, ids: List):
    filtered_seg = np.zeros_like(seg)
    for x in tqdm(ids):
        mask = seg == x
        filtered_seg[mask] += seg[mask]

    return filtered_seg


def parse_tuple(string):
    return tuple(map(float, string.split(',')))


def save_as_npy(seg: np.ndarray, out_path: Union[str, Path], args):
    # make a npy file
    np_out_path = os.path.splitext(out_path)[0] + '.npy'
    np.save(np_out_path, seg)
    print(f"npy file saved at: {np_out_path}")


def save_as_zarr(seg: np.ndarray, out_path: Union[str, Path], args, component: str = "volumes/labels/neuron_ids"):
    # the output dataset is prefixed because all lsd models expect labels to be in this dataset
    # always overwrite the existing dataset
    da.to_zarr(da.from_array(seg), url=out_path, component=component, overwrite=True)

    # we must set the resolution and offset for the labels from user input
    f = zarr.open(out_path, "a")
    f[component].attrs["resolution"] = args.resolution
    f[component].attrs["offset"] = args.offset
    print(f"zarr file saved at: {out_path}")


def main(args):
    seg = np.load(args.f)
    if args.df.endswith('.csv'):
        df = pd.read_csv(args.df, skiprows=args.skip_rows if args.skip_rows is not None else None)
    elif args.df.endswith('.xlsx'):
        df = pd.read_excel(args.df, sheet_name=args.sheet_name,
                           skiprows=args.skip_rows if args.skip_rows is not None else None)

    # Todo: remove hard-coding
    # Currently this is biased with Shi-Yan's way of annotation
    bbox2 = df["ID_BBOX2"]
    bbox3 = df["ID_BBOX3"]
    bbox4 = df["ID_BBOX4"]

    ids = pd.concat([bbox2, bbox3, bbox4]).dropna().astype(int)
    print(f"Num of ids: {len(ids)}")
    ids = set(ids.iloc[:].to_list())  # uniques

    filtered_seg = make_gt_seg2link(seg=seg, ids=ids)
    save_as_zarr(seg=filtered_seg, out_path=args.out_path, args=args, component=args.dataset)
    save_as_npy(seg=filtered_seg, out_path=args.out_path, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This program takes a segmentation.npy file generated via seg2link"
        " containing corrections, and a list of ids which have been proofread,"
        " to generate a filtered zarr file as ground-truth"
        "for training LSDs.")
    parser.add_argument("-f", help="pass a seg-modified.py file from seg2link")
    parser.add_argument("-df", help="pass a csv/excel with ids you wish to pull which have been corrected.")
    parser.add_argument("-sheet_name", default="Sheet2", help="pass sheet name if it's an excel file.")
    parser.add_argument("-skip_rows", default=[0], type=list, help="mention any rows you wish to skip")
    parser.add_argument("-op", "--out_path", help="Path to save the output zarr")
    parser.add_argument("-res", "--resolution", type=parse_tuple, help="Voxel resolution")
    parser.add_argument("-off", "--offset", type=parse_tuple, help="Voxel Offset")
    parser.add_argument("-ds", "--dataset", default="volumes/labels/neuron_ids",
                        help="Dataset name inside the zarr, where the filtered segmentation will be saved.")

    args = parser.parse_args()
    main(args)
