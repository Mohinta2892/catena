"""
Merges neuron_ids with mito_ids to form a singualr labels dataset.
Primarily necessary for training on sparse segmentation data, as the Funkelab pipeline may not have been optimised to
reject blank regions efficiently, which leads to slow and spurious training.

NB: This is RAM intensive as relabelling will load entire datasets as numpy arrays into memory.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University, UK
"""
import os.path
import sys
import numpy as np
import skimage
import zarr
from glob import glob
import argparse
import dask.array as da
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_utils.preprocess_volumes.clahe_gconn import copy_datasets_from_multiple_sources


def merge_labels(file_, ds1, ds2, ds_to_copy=None):
    ds1_ids = da.from_zarr(file_, component=ds1)
    ds2_ids = da.from_zarr(file_, component=ds2)

    ds1_ids = np.asarray(ds1_ids)
    ds2_ids = np.asarray(ds2_ids)

    relab_ds1, _, _ = skimage.segmentation.relabel_sequential(ds1_ids)
    # assign unique ids to ds2 from max+1 of ds1, ensures no repeats
    relab_ds2, _, _ = skimage.segmentation.relabel_sequential(ds2_ids, offset=np.max(ds1_ids) + 1)

    relab_merge_ids = da.add(relab_ds1, relab_ds2)

    outfile = f"{os.path.splitext(file_)[0]}_merged_ids.zarr"
    da.to_zarr(da.from_array(relab_merge_ids), outfile, component="volumes/labels/neuron_ids",
               overwrite=True)
    # add attrs
    f = zarr.open(outfile, "a")
    zf = zarr.open(file_)[ds1]
    f["volumes/labels/neuron_ids"].attrs["resolution"] = zf.attrs["resolution"]
    f["volumes/labels/neuron_ids"].attrs["offset"] = zf.attrs["offset"]

    if ds_to_copy is not None:
        copy_datasets_from_multiple_sources(in_zarr=file_, out_zarr=outfile, datasets_to_copy=ds_to_copy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', help="Directory with zarrs")
    parser.add_argument('-ds1', help="Dataset containing a label class (e.g., neuron_ids)")
    parser.add_argument('-ds2', help="Dataset containing another label class (e.g., mito_ids)")
    parser.add_argument('-ds_copy', nargs='+',
                        default=None,
                        help='Datasets to copy from source zarr to out zarr. These are either labels/mask.')

    args = parser.parse_args()

    list_of_zarrs = glob(f"{args.dir}/*zarr")

    for file_ in list_of_zarrs:
        print(f"Processing {file_}")
        merge_labels(file_=file_, ds1=args.ds1, ds2=args.ds2, ds_to_copy=args.ds_copy)


if __name__ == "__main__":
    main()
