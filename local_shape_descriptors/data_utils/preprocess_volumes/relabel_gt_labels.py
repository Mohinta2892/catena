"""
This is an ad-hoc script that enable relabelling the segmentations in GT sub-volumes to number that potentially lie
in a range that is much smaller than the uint64 values in the original volumes, from where the subvolumes have been
cropped.
The script was conceived after observing affinity maps were not being generated from subvolumes with `large` values.
This is a pre-processing script, which:
- Takes the input zarrs from the input data directories
- Preserves the original labels by renaming `volumes/labels/neuron_ids` to `volumes/labels/neuron_ids_original`
- Creates a relabelled dataset `volumes/labels/neuron_ids`
- Saves the forward map of original --> new labels as a dict attribute in `volumes/labels/neuron_ids_original`

Author: Samia Mohinta, Cardona Lab, Cambridge University

"""
import argparse

from skimage.segmentation import relabel_sequential
import zarr
import h5py
import numpy as np
from utils import read_zarr
from glob import glob
from tqdm import tqdm


def relabel_segmentations(seg, offset=1):
    relab, fw_map, inv_map = relabel_sequential(seg, offset=offset)

    return relab, fw_map, inv_map


def main(args):
    data_dir = args.data_dir

    # all input zarrs
    samples = glob(f"{data_dir}/*.zarr")

    for sample in tqdm(samples):
        print(f"Opened {sample} in append mode...")
        f = read_zarr(sample, mode='a')  # must open in append mode to be able to edit the files
        # the datasets are preset - volumes/labels/neuron_ids must exist
        if int(args.dim_2d):
            slices = len(f["volumes/labels"].items())
            for sl in range(slices):
                seg = f[f"volumes/labels/{sl}"]  # this is still a zarr.Array object, not the actual ndarray data
                # grab the original attributes before overwriting
                vol_offset = seg.attrs["offset"]
                resolution = seg.attrs["resolution"]

                # call relabelling the ndarray data
                relab, fw_map, inv_map = relabel_segmentations(seg[...], offset=1)
                # convert the fw_map into a dict
                relabel_fw_map = {}
                for i, j in zip(fw_map.in_values, fw_map.out_values):
                    relabel_fw_map.update({int(i): int(j)})

                # preserve the original dataset
                f[f"volumes/labels_original/{sl}"] = seg[...]
                f[f"volumes/labels_original/{sl}"].attrs["offset"] = vol_offset
                f[f"volumes/labels_original/{sl}"].attrs["resolution"] = resolution
                f[f"volumes/labels_original/{sl}"].attrs["fw_map"] = relabel_fw_map

                # make a new relabeled dataset
                f[f"volumes/labels/{sl}"] = relab[...]
                f[f"volumes/labels/{sl}"].attrs["offset"] = vol_offset
                f[f"volumes/labels/{sl}"].attrs["resolution"] = resolution
        else:
            seg = f["volumes/labels/neuron_ids"]  # this is still a zarr.Array object, not the actual ndarray data
            # grab the original attributes before overwriting
            vol_offset = seg.attrs["offset"]
            resolution = seg.attrs["resolution"]

            # call relabelling the ndarray data
            relab, fw_map, inv_map = relabel_segmentations(seg[...], offset=1)
            # convert the fw_map into a dict
            relabel_fw_map = {}
            for i, j in zip(fw_map.in_values, fw_map.out_values):
                relabel_fw_map.update({int(i): int(j)})

            # preserve the original dataset
            f["volumes/labels/neuron_ids_original"] = seg[...]
            f["volumes/labels/neuron_ids_original"].attrs["offset"] = vol_offset
            f["volumes/labels/neuron_ids_original"].attrs["resolution"] = resolution
            f["volumes/labels/neuron_ids_original"].attrs["fw_map"] = relabel_fw_map

            # make a new relabeled dataset
            f["volumes/labels/neuron_ids"] = relab[...]
            f["volumes/labels/neuron_ids"].attrs["offset"] = vol_offset
            f["volumes/labels/neuron_ids"].attrs["resolution"] = resolution
            
        print(f"Finished relabelled segmentations in {sample}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Relabels the ids using skimage from large values to smaller number in subvolumes")
    parser.add_argument('-d', '--data_dir', help='Absolute path where you have kept your .zarrs.'
                                                 ' Ensure the zarr have a dataset `volumes/labels/neuron_ids` ',
                        required=True)
    parser.add_argument('-dim', '--dim_2d', help='Is data 2D or 3D? Options: 1 = 2D, 0 = 3D',
                        required=True)
    args = parser.parse_args()

    main(args)
