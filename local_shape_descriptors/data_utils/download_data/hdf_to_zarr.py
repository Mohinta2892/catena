import h5py
import numpy as np
import zarr
import os
import sys

from glob import glob
import argparse
import numcodecs
from tqdm import tqdm

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_utils.preprocess_volumes.utils import list_keys


def hdf_to_zarr(hdf, out_dir):
    hf = h5py.File(hdf)
    datasets = list_keys(hf)

    # create out zarr
    zf_path = os.path.join(out_dir, os.path.basename(hdf).split('.hdf')[0] + '.zarr')
    zf = zarr.open(zf_path, mode='a')
    for ds in (pbar := tqdm(datasets, desc='Iterating over datasets hdf:')):
        pbar.set_postfix_str(f" {hdf}")
        if hf[ds].dtype == object:
            # print(list(hf[ds][...]))
            zf.create_dataset(name=ds, data=tuple(hf[ds][...]), shape=hf[ds].shape, dtype=object, object_codec=numcodecs.VLenBytes(),
                              overwrite=True)
        else:
            zf[ds] = hf[ds]
        for k in hf[ds].attrs.keys():
            zf[ds].attrs[k] = tuple(hf[ds].attrs[k])  # casting prevents - `object cannot be serialized error`

    # make a labels mask for the labels - required by lsds
    labels_mask_name = 'volumes/labels/labels_mask'
    labels_mask = np.ones_like(zf["volumes/labels/neuron_ids"][...])
    background = zf["volumes/labels/neuron_ids"] == 0
    labels_mask[background] = 0

    zf[labels_mask_name] = labels_mask
    zf[labels_mask_name].attrs['offset'] = zf["volumes/labels/neuron_ids"].attrs['offset']
    zf[labels_mask_name].attrs['resolution'] = zf["volumes/labels/neuron_ids"].attrs['resolution']


def main():
    parser = argparse.ArgumentParser(description="Converts hdf to zarr")
    parser.add_argument('-d', help="/directory/path/to/hdf")
    parser.add_argument('-od', help="/output/directory/path/to/zarr")

    args = parser.parse_args()

    all_hdfs = sorted(glob(f"{args.d}/*.h*"))
    for h in all_hdfs:
        hdf_to_zarr(h, args.od)


if __name__ == '__main__':
    main()
